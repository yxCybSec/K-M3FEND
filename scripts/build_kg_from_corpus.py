"""
Expand local KG (entities.txt + triples.tsv) from KM3FEND pkl corpora.

Inspired by KG-MFEND: build multi-domain structured knowledge without LLM.
- Entities: jieba noun phrases + seed entities, filtered by document frequency.
- Triples: seed triples + (entity, 所属领域, domain) + (e1, 文本共现, e2) from co-occurrence.

Usage (from project root):
  python scripts/build_kg_from_corpus.py --data_dir data/ch
  python scripts/build_kg_from_corpus.py --data_dir data/ch --splits train val test
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set, Tuple

try:
    import jieba
    import jieba.posseg as pseg

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")
_VALID_FLAGS = frozenset({"nr", "ns", "nt", "nz", "n", "vn", "eng"})


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _write_entities(path: str, entities: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ordered = sorted(set(entities), key=lambda x: (-len(x), x))
    with open(path, "w", encoding="utf-8") as f:
        for e in ordered:
            f.write(e + "\n")


def _write_triples(path: str, triples: Iterable[Tuple[str, str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    seen: Set[Tuple[str, str, str]] = set()
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            key = (h, r, t)
            if key in seen or not h or not r or not t:
                continue
            seen.add(key)
            f.write(f"{h}\t{r}\t{t}\n")


def _extract_terms(text: str, min_len: int, max_len: int) -> List[str]:
    terms: List[str] = []
    if not text:
        return terms
    if _HAS_JIEBA:
        for w, flag in pseg.cut(text):
            w = w.strip()
            if not w or len(w) < min_len or len(w) > max_len:
                continue
            if flag in _VALID_FLAGS or (flag.startswith("n") and len(w) >= 2):
                if _CJK_RE.fullmatch(w) or (flag == "eng" and len(w) >= 3):
                    terms.append(w)
    for m in _CJK_RE.findall(text):
        if min_len <= len(m) <= max_len:
            terms.append(m)
    return terms


def _load_pkls(data_dir: str, splits: List[str]) -> List[Tuple[str, object]]:
    frames = []
    for sp in splits:
        p = os.path.join(data_dir, f"{sp}.pkl")
        if not os.path.exists(p):
            print(f"[build_kg] skip missing {p}")
            continue
        with open(p, "rb") as f:
            df = pickle.load(f)
        frames.append((sp, df))
    return frames


def build_kg(
    data_dir: str,
    kg_dir: str,
    splits: List[str],
    min_entity_freq: int,
    min_entity_len: int,
    max_entity_len: int,
    max_entities: int,
    max_cooccur_pairs_per_doc: int,
    max_cooccur_triples: int,
) -> None:
    seed_entity_file = os.path.join(kg_dir, "entities.txt")
    seed_triple_file = os.path.join(kg_dir, "triples.tsv")
    seed_entities = set(_read_lines(seed_entity_file))
    seed_triples: List[Tuple[str, str, str]] = []
    for line in _read_lines(seed_triple_file):
        parts = line.split("\t")
        if len(parts) == 3:
            seed_triples.append((parts[0], parts[1], parts[2]))

    entity_freq: Counter = Counter()
    domain_entity_freq: Dict[str, Counter] = defaultdict(Counter)
    cooccur: Counter = Counter()

    for _split, df in _load_pkls(data_dir, splits):
        if "content" not in df.columns:
            raise KeyError(f"{_split}.pkl must contain 'content' column")
        has_cat = "category" in df.columns
        for idx in df.index:
            text = str(df.loc[idx, "content"])
            terms = _extract_terms(text, min_entity_len, max_entity_len)
            uniq = list(dict.fromkeys(terms))
            for t in uniq:
                entity_freq[t] += 1
            if has_cat:
                cat = str(df.loc[idx, "category"])
                for t in uniq[:12]:
                    domain_entity_freq[cat][t] += 1
            if len(uniq) >= 2:
                picked = uniq[:8]
                pairs = 0
                for i in range(len(picked)):
                    for j in range(i + 1, len(picked)):
                        if pairs >= max_cooccur_pairs_per_doc:
                            break
                        a, b = picked[i], picked[j]
                        if a == b:
                            continue
                        cooccur[(a, b)] += 1
                        pairs += 1

    corpus_entities = {e for e, c in entity_freq.items() if c >= min_entity_freq}
    entities = seed_entities | corpus_entities
    if max_entities > 0 and len(entities) > max_entities:
        top = [e for e, _ in entity_freq.most_common(max_entities)]
        entities = seed_entities | set(top)

    triples: List[Tuple[str, str, str]] = list(seed_triples)
    for cat, ctr in domain_entity_freq.items():
        for ent, cnt in ctr.most_common(80):
            if ent in entities and cnt >= min_entity_freq:
                triples.append((ent, "所属领域", cat))

    added_co = 0
    for (a, b), cnt in cooccur.most_common(max_cooccur_triples):
        if cnt < min_entity_freq or a not in entities or b not in entities:
            continue
        triples.append((a, "文本共现", b))
        added_co += 1

    out_entity = os.path.join(kg_dir, "entities.txt")
    out_triple = os.path.join(kg_dir, "triples.tsv")
    _write_entities(out_entity, entities)
    _write_triples(out_triple, triples)

    print(f"[build_kg] entities: {len(entities)} (seed {len(seed_entities)}, corpus {len(corpus_entities)})")
    print(f"[build_kg] triples: {len(set(triples))} (cooccur added ~{added_co})")
    print(f"[build_kg] wrote {out_entity}")
    print(f"[build_kg] wrote {out_triple}")


def main():
    parser = argparse.ArgumentParser(description="Build/expand KG from news pkl corpora")
    parser.add_argument("--data_dir", default="./data/ch")
    parser.add_argument("--kg_dir", default="./data/kg")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--min_entity_freq", type=int, default=3)
    parser.add_argument("--min_entity_len", type=int, default=2)
    parser.add_argument("--max_entity_len", type=int, default=10)
    parser.add_argument("--max_entities", type=int, default=12000, help="0 = no cap")
    parser.add_argument("--max_cooccur_pairs_per_doc", type=int, default=12)
    parser.add_argument("--max_cooccur_triples", type=int, default=80000)
    args = parser.parse_args()
    if not _HAS_JIEBA:
        print("[build_kg] warning: jieba not installed; entity recall will be lower. pip install jieba")
    build_kg(
        data_dir=args.data_dir,
        kg_dir=args.kg_dir,
        splits=args.splits,
        min_entity_freq=args.min_entity_freq,
        min_entity_len=args.min_entity_len,
        max_entity_len=args.max_entity_len,
        max_entities=args.max_entities,
        max_cooccur_pairs_per_doc=args.max_cooccur_pairs_per_doc,
        max_cooccur_triples=args.max_cooccur_triples,
    )


if __name__ == "__main__":
    main()
