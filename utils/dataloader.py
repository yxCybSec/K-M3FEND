import torch
import random
import pandas as pd
import tqdm
import numpy as np
import pickle
import re
import os
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

def _init_fn(worker_id):
    np.random.seed(2021)

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

def df_filter(df_data, category_dict):
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data

def word2input(texts, max_len, dataset):
    if dataset == 'ch':
        tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    elif dataset == 'en':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def process(x):
    x['content_emotion'] = x['content_emotion'].astype(float)
    return x


def _load_aligned_npy(path, df, expected_cols=None, name="array"):
    if not path or not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
        if expected_cols is not None and arr.ndim == 2 and arr.shape[1] != expected_cols:
            print(f'[dataloader] WARNING: {name} cols mismatch ({arr.shape[1]} vs {expected_cols}), skipping {path}', flush=True)
            return None
        idx = df.index.to_numpy()
        if idx.size > 0:
            max_idx = int(idx.max())
            if max_idx >= arr.shape[0]:
                print(f'[dataloader] WARNING: {name} rows mismatch ({arr.shape[0]} vs {max_idx+1}), skipping {path}', flush=True)
                return None
        if arr.ndim == 1:
            return torch.tensor(arr[idx].astype('float32'))
        return torch.tensor(arr[idx].astype('float32'))
    except Exception as e:
        print(f'[dataloader] WARNING: failed to load {name} from {path}: {e}', flush=True)
        return None


class bert_data():
    def __init__(self, max_len, batch_size, category_dict, dataset, num_workers=2, balanced_domain_sampler=False):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset = dataset
        self.balanced_domain_sampler = balanced_domain_sampler
        self.knowledge_mode = 'kg'
    
    def load_data(self, path, shuffle):
        print(f'[dataloader] 读取 {path} ...', flush=True)
        self.data = df_filter(read_pkl(path), self.category_dict)
        n = len(self.data)
        print(f'[dataloader] 样本数 {n}，正在分词（CPU 上逐条 encode，请稍候）...', flush=True)
        content = self.data['content'].to_numpy()
        comments = self.data['comments'].to_numpy()
        content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
        comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        content_token_ids, content_masks = word2input(content, self.max_len, self.dataset)
        print(f'[dataloader] content 分词完成，正在处理 comments ...', flush=True)
        comments_token_ids, comments_masks = word2input(comments, self.max_len, self.dataset)

        knowledge = None
        kg_trust = None
        knowledge_candidates = []
        if self.knowledge_mode == 'none':
            print(f'[dataloader] knowledge_mode=none, skipping knowledge loading', flush=True)
        elif isinstance(path, str) and path.endswith('.pkl'):
            # Prefer structured KG vectors over legacy LLM knowledge.npy
            knowledge_candidates.append(path.replace('.pkl', '_kg_knowledge.npy'))
            knowledge_candidates.append(path.replace('.pkl', '_knowledge.npy'))
        if isinstance(path, str):
            knowledge_candidates.append(os.path.join(os.path.dirname(path), 'kg_knowledge.npy'))
            knowledge_candidates.append(os.path.join(os.path.dirname(path), 'knowledge.npy'))
        for cand in knowledge_candidates:
            if not cand or not os.path.exists(cand):
                continue
            knowledge = _load_aligned_npy(cand, self.data, expected_cols=768, name='knowledge')
            print(f'[dataloader] loaded knowledge from {cand}', flush=True)
            break

        if isinstance(path, str) and path.endswith('.pkl'):
            trust_path = path.replace('.pkl', '_kg_trust.npy')
            kg_trust = _load_aligned_npy(trust_path, self.data, expected_cols=1, name='kg_trust')
            if kg_trust is not None:
                if kg_trust.dim() == 1:
                    kg_trust = kg_trust.unsqueeze(1)
                print(f'[dataloader] loaded kg_trust from {trust_path}', flush=True)

        extra_tensors = []
        if knowledge is not None:
            extra_tensors.append(knowledge)
        if kg_trust is not None:
            extra_tensors.append(kg_trust)

        dataset = TensorDataset(
            content_token_ids,
            content_masks,
            comments_token_ids,
            comments_masks,
            content_emotion,
            comments_emotion,
            emotion_gap,
            style_feature,
            label,
            category,
            *extra_tensors,
        )
        sampler = None
        use_shuffle = shuffle
        if shuffle and self.balanced_domain_sampler:
            cat_np = category.numpy()
            counts = np.bincount(cat_np, minlength=len(self.category_dict)).astype(np.float32)
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
            sample_weights = inv[cat_np]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(sample_weights),
                replacement=True,
            )
            use_shuffle = False

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=use_shuffle,
            sampler=sampler,
            worker_init_fn=_init_fn
        )
        print(f'[dataloader] 完成 {path}（shuffle={shuffle}）', flush=True)
        return dataloader
