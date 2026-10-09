"""
Build structured KG knowledge for K-M3FEND from CN-DBpedia-style local files.

Outputs (aligned with pkl row index):
  - {split}_kg.jsonl
  - {split}_knowledge.npy  [N, 768]
  - {split}_kg_trust.npy   [N, 1]

Requires: transformers, torch, numpy, pandas (optional jieba for better entity recall).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import jieba
    import jieba.posseg as pseg

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class KGIndex:
    """Local entity list + triple store (head, relation, tail) tab-separated."""

    def __init__(self, entity_file: str, triple_file: str, max_triples_per_entity: int = 8):
        self.entities: Set[str] = set()
        self.entity_list: List[str] = []
        if os.path.exists(entity_file):
            with open(entity_file, "r", encoding="utf-8") as f:
                for line in f:
                    e = line.strip()
                    if e and e not in self.entities:
                        self.entities.add(e)
                        self.entity_list.append(e)
        self.entity_list.sort(key=len, reverse=True)
        self._max_ent_len = max((len(e) for e in self.entity_list), default=0)

        self.triples_by_head: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.triples_by_tail: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        if os.path.exists(triple_file):
            with open(triple_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    h, r, t = [p.strip() for p in parts]
                    if not h or not r or not t:
                        continue
                    self.entities.add(h)
                    self.entities.add(t)
                    trip = (h, r, t)
                    if len(self.triples_by_head[h]) < max_triples_per_entity:
                        self.triples_by_head[h].append(trip)
                    if len(self.triples_by_tail[t]) < max_triples_per_entity:
                        self.triples_by_tail[t].append(trip)

        self.max_triples_per_entity = max_triples_per_entity

    def link_entities(self, text: str, max_entities: int = 10) -> List[str]:
        if not text:
            return []
        found: List[str] = []
        seen: Set[str] = set()

        if _HAS_JIEBA:
            for w, flag in pseg.cut(text):
                if len(w) < 2:
                    continue
                if flag in ("nr", "ns", "nt", "nz", "n", "vn") or flag.startswith("n"):
                    if w in self.entities and w not in seen:
                        found.append(w)
                        seen.add(w)
                if len(found) >= max_entities:
                    return found[:max_entities]

        if len(found) < max_entities and self._max_ent_len > 0:
            for ent in self.entity_list:
                if len(ent) > len(text):
                    continue
                if ent in text and ent not in seen:
                    found.append(ent)
                    seen.add(ent)
                if len(found) >= max_entities:
                    break

        if len(found) < max_entities:
            for m in re.findall(r"[\u4e00-\u9fff]{2,10}", text):
                if m in self.entities and m not in seen:
                    found.append(m)
                    seen.add(m)
                if len(found) >= max_entities:
                    break
        return found[:max_entities]

    def retrieve_triples(self, entities: List[str], max_total: int = 12) -> List[Dict[str, str]]:
        trips: List[Dict[str, str]] = []
        seen_t: Set[Tuple[str, str, str]] = set()
        for e in entities:
            for h, r, t in self.triples_by_head.get(e, []) + self.triples_by_tail.get(e, []):
                key = (h, r, t)
                if key in seen_t:
                    continue
                seen_t.add(key)
                trips.append({"head": h, "relation": r, "tail": t, "anchor": e})
                if len(trips) >= max_total:
                    return trips
        return trips

    def sentence_tree_text(
        self,
        text: str,
        entities: List[str],
        triples: List[Dict[str, str]],
        max_content_chars: int = 128,
    ) -> str:
        """
        KG-MFEND-style sentence tree (linearized for BERT encoding):
        root news snippet -> per-entity branches -> triple clauses.
        """
        root = text[:max_content_chars].replace("\n", " ").strip()
        if not entities and not triples:
            return root
        by_anchor: Dict[str, List[str]] = defaultdict(list)
        for tr in triples:
            anchor = tr.get("anchor") or tr["head"]
            clause = f"{tr['head']}{tr['relation']}{tr['tail']}"
            by_anchor[anchor].append(clause)
        branches: List[str] = []
        for ent in entities:
            clauses = by_anchor.get(ent, [])
            if clauses:
                branches.append(f"{ent}:" + "，".join(clauses[:4]))
            else:
                branches.append(ent)
        return " [SEP] ".join([root] + branches)

    def triple_text(self, text: str, entities: List[str], triples: List[Dict[str, str]]) -> str:
        return self.sentence_tree_text(text, entities, triples)

    def compute_trust(
        self,
        text: str,
        entities: List[str],
        triples: List[Dict[str, str]],
    ) -> float:
        if not entities and not triples:
            return 0.0
        ent_score = min(1.0, len(entities) / 3.0)
        trip_score = min(1.0, len(triples) / 4.0)
        overlap = sum(1 for e in entities if e in text)
        overlap_score = overlap / max(1, len(entities))
        # Down-weight noisy co-occurrence-only triples
        structured = sum(
            1 for tr in triples if tr.get("relation") not in ("文本共现",)
        )
        struct_score = min(1.0, structured / 3.0) if triples else 0.0
        return float(0.35 * ent_score + 0.35 * trip_score + 0.15 * overlap_score + 0.15 * struct_score)


class _SQLiteKV:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    def get(self, k: str) -> Optional[str]:
        cur = self.conn.execute("SELECT v FROM kv WHERE k = ?", (k,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (k, v))
        self.conn.commit()


class CNDBpediaClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 8.0,
        cache_path: Optional[str] = None,
        apikey: Optional[str] = None,
    ):
        cleaned = (base_url or "").strip()
        for _ in range(3):
            if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("`", '"', "'"):
                cleaned = cleaned[1:-1].strip()
        self.base_url = cleaned
        self.timeout = float(timeout)
        self.cache = _SQLiteKV(cache_path) if cache_path else None
        self.apikey = (apikey or "").strip() or None

    def avpair(self, entity: str) -> object:
        key = f"avpair:{entity}"
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                return json.loads(cached)
        sep = "&" if "?" in self.base_url else "?"
        params = {"q": entity}
        if self.apikey:
            params["apikey"] = self.apikey
        url = self.base_url + sep + urlencode(params)
        req = Request(url, headers={"Accept": "application/json"})
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(raw)
                if self.cache:
                    self.cache.set(key, json.dumps(data, ensure_ascii=False))
                return data
            except HTTPError as e:
                last_err = e
                if e.code == 404:
                    raise RuntimeError(
                        f"CN-DBpedia HTTP 404 for entity={entity}. "
                        f"Endpoint may require apikey or be unreachable: {url}"
                    )
                time.sleep(0.5)
            except URLError as e:
                last_err = e
                time.sleep(0.5)
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        raise RuntimeError(f"CN-DBpedia request failed for entity={entity}: {last_err}")


class OwnThinkClient:
    def __init__(
        self,
        base_url: str = "https://api.ownthink.com/kg",
        timeout: float = 8.0,
        cache_path: Optional[str] = None,
    ):
        cleaned = (base_url or "").strip()
        for _ in range(3):
            if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("`", '"', "'"):
                cleaned = cleaned[1:-1].strip()
        self.base_url = cleaned
        self.timeout = float(timeout)
        self.cache = _SQLiteKV(cache_path) if cache_path else None

    def knowledge(self, entity: str) -> object:
        key = f"ownthink:{entity}"
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                return json.loads(cached)
        encoded_entity = urlencode({"entity": entity})
        url = f"{self.base_url}/knowledge?{encoded_entity}"
        req = Request(url, headers={"Accept": "application/json"})
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(raw)
                if self.cache:
                    self.cache.set(key, json.dumps(data, ensure_ascii=False))
                return data
            except HTTPError as e:
                last_err = e
                time.sleep(0.5)
            except URLError as e:
                last_err = e
                time.sleep(0.5)
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        print(f"[ownthink] request failed for entity={entity}: {last_err}", flush=True)
        return {"message": "error", "data": {}}


_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要",
    "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "这个", "那个", "什么", "怎么", "为什么", "因为",
    "所以", "但是", "如果", "虽然", "还是", "或者", "以及", "等等", "通过", "根据", "按照", "关于", "对于",
    "可以", "可能", "应该", "必须", "需要", "已经", "正在", "将要", "曾经", "一直", "经常", "偶尔", "从来",
    "只是", "不过", "却", "而", "又", "再", "还", "更", "最", "太", "非常", "十分", "特别", "比较", "稍微",
    "几乎", "简直", "好像", "仿佛", "如同", "比如", "例如", "包括", "包含", "属于", "不是", "不能", "不要",
    "不用", "没关系", "没问题", "谢谢", "对不起", "您好", "请", "请问", "一下", "一会儿", "现在", "今天",
    "明天", "昨天", "后天", "前天", "今年", "去年", "明年", "最近", "刚才", "马上", "立刻", "过", "呢", "吗",
    "吧", "啊", "呀", "哇", "哦", "嗯", "哈", "嘿", "喂", "嗨",
}


def _extract_entity_candidates(text: str, max_entities: int) -> List[str]:
    if not text:
        return []

    entities: List[str] = []
    seen: Set[str] = set()

    known_entities = [
        '万象', '国际物理学', '中国科研', '欧洲物理学会', '物理世界',
        '中科大', '中国科学技术大学', '潘建伟', '陆朝阳',
        '多自由度量子隐形传态', '量子隐形传态',
        '人民日报', '新华社', '央视', '凤凰网', '新浪', '搜狐', '网易', '腾讯', '今日头条',
        '微博', '微信', '抖音', '快手', 'B站', '小红书', '淘宝', '京东', '拼多多', '美团', '滴滴',
        '华为', '小米', '苹果', '三星', '谷歌', '微软', '亚马逊', '阿里巴巴', '百度', '字节跳动',
        '清华大学', '北京大学', '浙江大学', '复旦大学', '上海交通大学',
        '中国科学院', '中国工程院', '国家自然科学基金', '科技部', '教育部',
        '钟南山', '袁隆平', '屠呦呦', '钱学森', '邓稼先',
    ]
    for entity in known_entities:
        if entity in text and entity not in seen:
            seen.add(entity)
            entities.append(entity)
            if len(entities) >= max_entities:
                return entities

    entity_patterns = [
        (r'(?:大学|学院|研究所|实验室|研究院|中心)', 2, 10),
        (r'(?:公司|集团|科技|技术|网络|软件|系统|平台|有限责任)', 2, 10),
        (r'(?:银行|保险|证券|基金|投资|控股)', 2, 10),
        (r'(?:协会|学会|基金会|委员会|联合会)', 2, 10),
        (r'(?:出版社|报社|电视台|网站|新闻网|新闻网站)', 2, 10),
        (r'(?:省|市|区|县|镇|村|街道|社区)', 2, 10),
        (r'(?:公园|广场|车站|机场|港口|码头)', 2, 10),
        (r'(?:山脉|河流|湖泊|海洋|岛屿|森林|草原|沙漠|高原|盆地|平原|丘陵)', 2, 10),
        (r'(?:教授|博士|院士|专家|研究员|科学家|工程师)', 2, 6),
        (r'(?:主席|总理|总统|部长|省长|市长|局长|会长|理事长|秘书长)', 2, 6),
        (r'(?:医生|律师|设计师|分析师|建筑师|会计师)', 2, 6),
        (r'(?:作家|艺术家|运动员|明星|演员|导演|主持人)', 2, 6),
        (r'(?:互联网|移动|通信|电信|运营商|芯片|手机|电脑|汽车|机器人)', 2, 10),
        (r'(?:智能|人工智能|机器学习|深度学习|大数据|云计算|物联网|区块链)', 2, 10),
        (r'(?:物理|化学|生物|数学|天文学|地理学|地质学|医学|工程|计算机|科学)', 2, 10),
        (r'(?:新闻|媒体|报道|文章|资讯|信息|数据)', 2, 10),
        (r'(?:国际|中国|美国|日本|韩国|英国|法国|德国|俄罗斯|印度|巴西)', 2, 10),
    ]

    for suffix_pattern, min_len, max_len in entity_patterns:
        for l in range(max_len, min_len - 1, -1):
            pattern = r'[\u4e00-\u9fff]{' + str(l) + '}' + suffix_pattern
            matches = re.findall(pattern, text)
            for m in matches:
                if m in seen:
                    continue
                seen.add(m)
                entities.append(m)
                if len(entities) >= max_entities:
                    return entities

    person_patterns = [
        r'[\u4e00-\u9fff]{2,4}(?:教授|博士|院士|专家|研究员|科学家)',
        r'[\u4e00-\u9fff]{2,4}(?:主席|总理|总统|部长|省长|市长|局长)',
        r'[\u4e00-\u9fff]{2,4}(?:医生|律师|工程师|设计师|分析师)',
        r'[\u4e00-\u9fff]{2,4}(?:作家|艺术家|运动员|明星|演员|导演|主持人)',
    ]
    title_suffixes = ('教授', '博士', '院士', '专家', '研究员', '科学家', '主席', '总理', '总统', '部长', '省长', '市长', '局长', '医生', '律师', '工程师', '设计师', '分析师', '作家', '艺术家', '运动员', '明星', '演员', '导演', '主持人')
    for pattern in person_patterns:
        for m in re.findall(pattern, text):
            name = m[:-2] if len(m) > 2 and m[-2:] in title_suffixes else m
            if len(name) >= 2 and name not in seen:
                seen.add(name)
                entities.append(name)
                if len(entities) >= max_entities:
                    return entities

    for m in re.findall(r'[\u4e00-\u9fff]{3,4}', text):
        if m in seen:
            continue
        if m in _STOPWORDS:
            continue
        if m[-1] in ('的', '了', '在', '是', '有', '和', '就', '不', '都', '也', '很', '到', '说', '要', '去', '会', '着', '过', '呢', '吗', '吧', '啊', '呀', '哇', '哦', '嗯', '哈', '嘿', '喂', '嗨', '之', '与', '及', '等', '所', '以', '而', '却', '则', '即', '乃', '其', '此', '彼', '某', '各', '每', '诸', '众', '几', '多', '少', '全', '凡', '共', '仅', '只', '才', '刚', '已', '将', '曾', '正'):
            continue
        seen.add(m)
        entities.append(m)
        if len(entities) >= max_entities:
            break

    final = []
    for ent in entities:
        is_subset = False
        for existing in list(final):
            if ent in existing:
                is_subset = True
                break
            if existing in ent:
                final.remove(existing)
        if not is_subset:
            final.append(ent)

    return final[:max_entities]


def _cndbpedia_to_triples(entity: str, payload: object, max_total: int, anchor: str) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    data = payload
    if isinstance(payload, dict) and "ret" in payload:
        data = payload.get("ret")
    if isinstance(data, dict):
        items = list(data.items())
        for k, v in items:
            if isinstance(v, list):
                for vv in v[:2]:
                    triples.append({"head": entity, "relation": str(k), "tail": str(vv), "anchor": anchor})
                    if len(triples) >= max_total:
                        return triples
            else:
                triples.append({"head": entity, "relation": str(k), "tail": str(v), "anchor": anchor})
                if len(triples) >= max_total:
                    return triples
        return triples
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                p = x.get("predicate") or x.get("prop") or x.get("property") or x.get("p")
                o = x.get("object") or x.get("value") or x.get("o")
                if p is not None and o is not None:
                    triples.append({"head": entity, "relation": str(p), "tail": str(o), "anchor": anchor})
                    if len(triples) >= max_total:
                        return triples
                if len(x) == 1:
                    kk = next(iter(x.keys()))
                    vv = x[kk]
                    triples.append({"head": entity, "relation": str(kk), "tail": str(vv), "anchor": anchor})
                    if len(triples) >= max_total:
                        return triples
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                triples.append({"head": entity, "relation": str(x[0]), "tail": str(x[1]), "anchor": anchor})
                if len(triples) >= max_total:
                    return triples
    return triples


class CNDBpediaIndex:
    def __init__(
        self,
        base_url: str,
        timeout: float = 8.0,
        cache_path: Optional[str] = None,
        apikey: Optional[str] = None,
        max_triples_per_entity: int = 8,
    ):
        self.client = CNDBpediaClient(
            base_url=base_url,
            timeout=timeout,
            cache_path=cache_path,
            apikey=apikey,
        )
        self.max_triples_per_entity = int(max_triples_per_entity)

    def link_entities(self, text: str, max_entities: int = 10) -> List[str]:
        return _extract_entity_candidates(text, max_entities=max_entities)

    def retrieve_triples(self, entities: List[str], max_total: int = 12) -> List[Dict[str, str]]:
        trips: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for ent in entities:
            payload = self.client.avpair(ent)
            ent_trips = _cndbpedia_to_triples(
                ent,
                payload,
                max_total=self.max_triples_per_entity,
                anchor=ent,
            )
            for t in ent_trips:
                key = (t["head"], t["relation"], t["tail"])
                if key in seen:
                    continue
                seen.add(key)
                trips.append(t)
                if len(trips) >= max_total:
                    return trips
        return trips

    def sentence_tree_text(
        self,
        text: str,
        entities: List[str],
        triples: List[Dict[str, str]],
        max_content_chars: int = 128,
    ) -> str:
        root = text[:max_content_chars].replace("\n", " ").strip()
        if not entities and not triples:
            return root
        by_anchor: Dict[str, List[str]] = defaultdict(list)
        for tr in triples:
            anchor = tr.get("anchor") or tr["head"]
            clause = f"{tr['head']}{tr['relation']}{tr['tail']}"
            by_anchor[anchor].append(clause)
        branches: List[str] = []
        for ent in entities:
            clauses = by_anchor.get(ent, [])
            if clauses:
                branches.append(f"{ent}:" + "，".join(clauses[:4]))
            else:
                branches.append(ent)
        return " [SEP] ".join([root] + branches)

    def triple_text(self, text: str, entities: List[str], triples: List[Dict[str, str]]) -> str:
        return self.sentence_tree_text(text, entities, triples)

    def compute_trust(
        self,
        text: str,
        entities: List[str],
        triples: List[Dict[str, str]],
    ) -> float:
        if not entities and not triples:
            return 0.0
        ent_score = min(1.0, len(entities) / 3.0)
        trip_score = min(1.0, len(triples) / 4.0)
        overlap = sum(1 for e in entities if e in text)
        overlap_score = overlap / max(1, len(entities))
        structured = sum(1 for tr in triples if tr.get("relation") not in ("文本共现",))
        struct_score = min(1.0, structured / 3.0) if triples else 0.0
        return float(0.35 * ent_score + 0.35 * trip_score + 0.15 * overlap_score + 0.15 * struct_score)


def _ownthink_to_triples(entity: str, payload: object, max_total: int, anchor: str) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except:
            return triples
    if not isinstance(payload, dict):
        return triples
    data = payload.get("data", {})
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return triples
    if not isinstance(data, dict):
        return triples
    if not data or payload.get("message") != "success":
        return triples
    avp = data.get("avp", [])
    for attr, value in avp[:max_total]:
        if isinstance(value, (list, tuple)):
            for vv in value[:2]:
                triples.append({"head": entity, "relation": str(attr), "tail": str(vv), "anchor": anchor})
                if len(triples) >= max_total:
                    return triples
        else:
            triples.append({"head": entity, "relation": str(attr), "tail": str(value), "anchor": anchor})
            if len(triples) >= max_total:
                return triples
    tags = data.get("tag", [])
    for tag in tags[:min(3, max_total - len(triples))]:
        triples.append({"head": entity, "relation": "标签", "tail": str(tag), "anchor": anchor})
    return triples


class OwnThinkIndex:
    def __init__(
        self,
        base_url: str = "https://api.ownthink.com/kg",
        timeout: float = 8.0,
        cache_path: Optional[str] = None,
        max_triples_per_entity: int = 8,
    ):
        self.client = OwnThinkClient(
            base_url=base_url,
            timeout=timeout,
            cache_path=cache_path,
        )
        self.max_triples_per_entity = int(max_triples_per_entity)

    def link_entities(self, text: str, max_entities: int = 10) -> List[str]:
        return _extract_entity_candidates(text, max_entities=max_entities)

    def retrieve_triples(self, entities: List[str], max_total: int = 12) -> List[Dict[str, str]]:
        trips: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for ent in entities:
            payload = self.client.knowledge(ent)
            ent_trips = _ownthink_to_triples(
                ent,
                payload,
                max_total=self.max_triples_per_entity,
                anchor=ent,
            )
            for t in ent_trips:
                key = (t["head"], t["relation"], t["tail"])
                if key in seen:
                    continue
                seen.add(key)
                trips.append(t)
                if len(trips) >= max_total:
                    return trips
        return trips

    def sentence_tree_text(
        self,
        text: str,
        entities: List[str],
        triples: List[Dict[str, str]],
        max_content_chars: int = 128,
    ) -> str:
        root = text[:max_content_chars].replace("\n", " ").strip()
        if not entities and not triples:
            return root
        by_anchor: Dict[str, List[str]] = defaultdict(list)
        for tr in triples:
            anchor = tr.get("anchor") or tr["head"]
            clause = f"{tr['head']}{tr['relation']}{tr['tail']}"
            by_anchor[anchor].append(clause)
        branches: List[str] = []
        for ent in entities:
            clauses = by_anchor.get(ent, [])
            if clauses:
                branches.append(f"{ent}:" + "，".join(clauses[:4]))
            else:
                branches.append(ent)
        return " [SEP] ".join([root] + branches)

    def triple_text(self, text: str, entities: List[str], triples: List[Dict[str, str]]) -> str:
        return self.sentence_tree_text(text, entities, triples)

    def compute_trust(
        self,
        text: str,
        entities: List[str],
        triples: List[Dict[str, str]],
    ) -> float:
        if not entities and not triples:
            return 0.0
        ent_score = min(1.0, len(entities) / 3.0)
        trip_score = min(1.0, len(triples) / 4.0)
        overlap = sum(1 for e in entities if e in text)
        overlap_score = overlap / max(1, len(entities))
        structured = sum(1 for tr in triples if tr.get("relation") not in ("文本共现",))
        struct_score = min(1.0, structured / 3.0) if triples else 0.0
        return float(0.35 * ent_score + 0.35 * trip_score + 0.15 * overlap_score + 0.15 * struct_score)


def _embed_texts(
    texts: List[str],
    dataset: str,
    model_name: Optional[str],
    max_len: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if model_name is None:
        model_name = "hfl/chinese-bert-wwm-ext" if dataset == "ch" else "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            v = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            vecs.append(v.detach().cpu().numpy().astype("float32"))
    return np.concatenate(vecs, axis=0)


def process_split(
    df,
    text_field: str,
    kg: KGIndex,
    out_jsonl: str,
    out_npy: str,
    out_trust_npy: str,
    dataset: str,
    embed_model: Optional[str],
    embed_max_len: int,
    embed_batch_size: int,
    device: str,
    limit: Optional[int],
    max_entities: int = 10,
    max_triples: int = 16,
    min_trust_to_embed: float = 0.08,
) -> None:
    indices = list(df.index)
    if limit is not None:
        indices = indices[: int(limit)]

    n_rows = int(df.index.max()) + 1 if len(df.index) else len(indices)
    trust_full = np.zeros((n_rows, 1), dtype=np.float32)
    emb_full = np.zeros((n_rows, 768), dtype=np.float32)

    records: List[dict] = []
    triple_texts: List[str] = []
    row_order: List[int] = []

    for it, idx in enumerate(indices):
        if (it + 1) % 50 == 0:
            print(f"[kg] processed {it + 1}/{len(indices)}", flush=True)
        text = str(df.loc[idx, text_field])
        entities = kg.link_entities(text, max_entities=max_entities)
        triples = kg.retrieve_triples(entities, max_total=max_triples)
        ttext = kg.triple_text(text, entities, triples)
        trust = kg.compute_trust(text, entities, triples)
        visible_rules = {
            "entities": entities,
            "allow_cls_all_linked": True,
            "content_only_self": True,
        }
        rec = {
            "row_index": int(idx),
            "entities": entities,
            "triples": triples,
            "triple_text": ttext,
            "visible_rules": visible_rules,
            "kg_trust": trust,
        }
        records.append(rec)
        row_order.append(int(idx))
        trust_full[int(idx), 0] = trust
        if trust >= min_trust_to_embed and (entities or triples):
            triple_texts.append(ttext)
        else:
            triple_texts.append("")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    texts_to_embed = [t for t in triple_texts if t]
    embed_indices = [i for i, t in enumerate(triple_texts) if t]
    if texts_to_embed:
        emb = _embed_texts(
            texts_to_embed,
            dataset=dataset,
            model_name=embed_model,
            max_len=embed_max_len,
            batch_size=embed_batch_size,
            device=device,
        )
        for j, i in enumerate(embed_indices):
            emb_full[row_order[i]] = emb[j]

    np.save(out_npy, emb_full)
    np.save(out_trust_npy, trust_full)
    print(f"[kg] wrote {out_jsonl} ({len(records)} rows)")
    print(f"[kg] wrote {out_npy} shape={emb_full.shape}")
    nz = float((np.abs(emb_full).sum(axis=1) > 0).mean()) if len(row_order) else 0.0
    print(f"[kg] wrote {out_trust_npy} mean_trust={trust_full[row_order].mean():.4f} nonzero_emb={nz:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Build structured KG knowledge npy/jsonl for KM3FEND")
    parser.add_argument("--pkl", required=True, help="Path to train/val/test.pkl")
    parser.add_argument("--dataset", choices=["ch", "en"], default="ch")
    parser.add_argument("--text_field", default="content")
    parser.add_argument("--kg_backend", choices=["local", "cndbpedia", "ownthink"], default="local")
    parser.add_argument("--entity_file", default="./data/kg/entities.txt")
    parser.add_argument("--triple_file", default="./data/kg/triples.tsv")
    parser.add_argument("--cndbpedia_base_url", default="http://shuyantech.com/api/cndbpedia/avpair")
    parser.add_argument("--cndbpedia_apikey", default=None)
    parser.add_argument("--cndbpedia_cache", default=None)
    parser.add_argument("--cndbpedia_timeout", type=float, default=8.0)
    parser.add_argument("--ownthink_base_url", default="https://api.ownthink.com/kg")
    parser.add_argument("--ownthink_cache", default=None)
    parser.add_argument("--ownthink_timeout", type=float, default=8.0)
    parser.add_argument("--out_npy", default=None, help="Default: <pkl_stem>_kg_knowledge.npy")
    parser.add_argument("--also_legacy_npy", action="store_true", help="Also write <stem>_knowledge.npy")
    parser.add_argument("--out_jsonl", default=None, help="Default: <pkl_stem>_kg.jsonl")
    parser.add_argument("--out_trust_npy", default=None, help="Default: <pkl_stem>_kg_trust.npy")
    parser.add_argument("--max_triples_per_entity", type=int, default=8)
    parser.add_argument("--max_entities", type=int, default=10)
    parser.add_argument("--max_triples", type=int, default=16)
    parser.add_argument("--embed_model", default=None)
    parser.add_argument("--embed_max_len", type=int, default=256)
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stem = os.path.splitext(args.pkl)[0]
    out_npy = args.out_npy or (stem + "_kg_knowledge.npy")
    out_jsonl = args.out_jsonl or (stem + "_kg.jsonl")
    out_trust = args.out_trust_npy or (stem + "_kg_trust.npy")

    with open(args.pkl, "rb") as f:
        df = pickle.load(f)

    if args.kg_backend == "local":
        kg = KGIndex(args.entity_file, args.triple_file, max_triples_per_entity=args.max_triples_per_entity)
        if not kg.entities:
            raise RuntimeError(f"No entities loaded from {args.entity_file} / {args.triple_file}")
    elif args.kg_backend == "cndbpedia":
        cache_path = args.cndbpedia_cache
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(args.pkl), "cndbpedia_cache.sqlite")
        kg = CNDBpediaIndex(
            base_url=args.cndbpedia_base_url,
            timeout=args.cndbpedia_timeout,
            cache_path=cache_path,
            apikey=args.cndbpedia_apikey,
            max_triples_per_entity=args.max_triples_per_entity,
        )
    elif args.kg_backend == "ownthink":
        cache_path = args.ownthink_cache
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(args.pkl), "ownthink_cache.sqlite")
        kg = OwnThinkIndex(
            base_url=args.ownthink_base_url,
            timeout=args.ownthink_timeout,
            cache_path=cache_path,
            max_triples_per_entity=args.max_triples_per_entity,
        )
    else:
        raise ValueError(f"Unknown kg_backend: {args.kg_backend}")

    process_split(
        df=df,
        text_field=args.text_field,
        kg=kg,
        out_jsonl=out_jsonl,
        out_npy=out_npy,
        out_trust_npy=out_trust,
        dataset=args.dataset,
        embed_model=args.embed_model,
        embed_max_len=args.embed_max_len,
        embed_batch_size=args.embed_batch_size,
        device=args.device,
        limit=args.limit,
        max_entities=args.max_entities,
        max_triples=args.max_triples,
    )
    if args.also_legacy_npy:
        legacy = stem + "_knowledge.npy"
        np.save(legacy, np.load(out_npy))
        print(f"[kg] also wrote legacy {legacy}")


if __name__ == "__main__":
    main()
