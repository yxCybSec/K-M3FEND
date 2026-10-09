# K-M3FEND: Knowledge-Guided Adaptive Expert Routing for Multi-View Multi-Domain Fake News Detection

This is the official implementation of **K-M3FEND**, a knowledge-guided multi-view multi-domain fake news detection framework that integrates external knowledge, domain memory retrieval, and mixture-of-experts routing.

## Introduction

K-M3FEND addresses three challenges in multi-domain fake news detection: (1) domain memory retrieval is sensitive to textual noise, (2) external knowledge is often used as auxiliary information rather than guiding domain modeling, and (3) input-driven expert routing is unreliable for noisy or ambiguous inputs. The framework incorporates four main components:

- **External Knowledge Extraction**: Entities are linked to a domain-specific knowledge graph constructed from the training corpus. Relational triples are retrieved, serialized, and encoded by a frozen pretrained encoder to produce 768-dimensional knowledge representations. A multi-factor trust score estimates knowledge reliability.

- **Multi-View Expert Extraction**: 16 sub-experts partitioned into three views — seven semantic experts (CNN-based), seven emotion experts (MLP-based), and two style experts (MLP-based) — capture complementary characteristics of news content.

- **Knowledge-Enhanced Domain Memory Bank**: External knowledge embeddings are fused with semantic features before being incorporated into the memory query, reducing the influence of textual noise on domain retrieval. A two-stage hierarchical attention mechanism (domain-level + slot-level) retrieves domain-aware representations.

- **Domain Mixture-of-Experts with Knowledge-Guided Gating**: A domain-specific expert and a shared expert produce logits that are fused via a routing score derived from memory retrieval confidence. A Knowledge-Guided Gate (K-Gate) dynamically adjusts the contributions of semantic and stylistic features based on the knowledge trust score.

The repository includes K-M3FEND and twelve baseline models: TextCNN, BiGRU, RoBERTa, StyleLSTM, DualEmotion, EANN, EDDFN, MMoE, MoSE, MDFEND, M³FEND, and DHEM-FND.

## Repository Structure

```
K-M3FEND/
├── main.py                 # Entry point
├── grid_search.py          # Training & evaluation runner
├── requirements.txt
├── data/
│   ├── ch/                 # Chinese Weibo21 dataset (3/6/9 domains)
│   └── en/                 # English dataset (3 domains)
├── models/
│   ├── km3fend.py          # K-M3FEND model
│   ├── kg_fusion.py        # Knowledge fusion layer & K-Gate
│   ├── layers.py           # Shared network layers
│   ├── bert.py             # RoBERTa baseline
│   ├── bigru.py            # BiGRU baseline
│   ├── textcnn.py          # TextCNN baseline
│   ├── eann.py             # EANN baseline
│   ├── eddfn.py            # EDDFN baseline
│   ├── mmoe.py             # MMoE baseline
│   ├── mose.py             # MoSE baseline
│   ├── mdfend.py           # MDFEND baseline
│   ├── dualemotion.py      # DualEmotion baseline
│   └── stylelstm.py        # StyleLSTM baseline
├── utils/
│   ├── dataloader.py       # Data loading & KG knowledge injection
│   └── utils.py            # Metrics & helpers
└── scripts/
    ├── prepare_kg_knowledge.py    # KG knowledge preparation pipeline
    ├── build_kg_from_corpus.py    # Build KG from corpus
    └── build_knowledge_from_kg.py # Embed KG triples to knowledge vectors
```

## Requirements

- Python 3.6+
- PyTorch >= 1.0
- Transformers >= 4.0
- Pandas, NumPy, scikit-learn, tqdm, jieba

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

- **Chinese Weibo21 dataset**: 9,592 samples across 9 domains (Science, Military, Education, Disaster, Politics, Health, Finance, Entertainment, Society). Supports `domain_num` of 3, 6, or 9.
- **English dataset**: 3 domains (GossipCop, PolitiFact, COVID).

Each dataset split (`train`/`val`/`test`) includes:
- `{split}.pkl` — raw data (content, comments, emotions, style, label, category)
- `{split}_kg.jsonl` — KG entities and triples per sample
- `{split}_kg_knowledge.npy` — embedded KG knowledge vectors `[N, 768]`
- `{split}_kg_trust.npy` — KG trust scores `[N, 1]`

## Usage

### Training

```bash
# K-M3FEND on Chinese 9-domain
python main.py --model_name km3fend --dataset ch --domain_num 9 --gpu 0

# K-M3FEND on Chinese 6-domain
python main.py --model_name km3fend --dataset ch --domain_num 6 --gpu 0

# K-M3FEND on Chinese 3-domain
python main.py --model_name km3fend --dataset ch --domain_num 3 --gpu 0

# K-M3FEND on English 3-domain
python main.py --model_name km3fend --dataset en --domain_num 3 --gpu 0
```

### KG Knowledge Preparation

If `{split}_kg_knowledge.npy` files are missing, run the preparation script:

```bash
python scripts/prepare_kg_knowledge.py --dataset ch
python scripts/prepare_kg_knowledge.py --dataset en
```

Or use `--kg_pack` flag to auto-prepare before training:

```bash
python main.py --model_name km3fend --dataset ch --domain_num 9 --kg_pack
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model_name` | `km3fend` | Model to run (km3fend, mdfend, bert, eann, etc.) |
| `--dataset` | `ch` | Dataset: `ch` or `en` |
| `--domain_num` | `3` | Number of domains (ch: 3/6/9, en: 3) |
| `--epoch` | `50` | Training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--batchsize` | `64` | Batch size |
| `--early_stop` | `5` | Early stopping patience |
| `--knowledge_mode` | `kg` | Knowledge mode: `kg`, `llm`, or `none` |
| `--con_weight` | `0.03` | Contrastive loss weight (λ_con) |
| `--con_temperature` | `0.123` | Contrastive temperature (τ_c) |
| `--k_gate_strength` | `0.4` | K-Gate intensity (β) |
| `--moe_confidence_scale` | `3.5` | MoE routing confidence scale (λ) |
| `--label_smoothing` | `0.1` | BCE label smoothing |
| `--kg_pack` | `False` | Enable KG fusion + label smoothing + contrastive loss |
| `--seed` | `2021` | Random seed |
| `--n_runs` | `1` | Number of repeated runs |

### Baselines

```bash
python main.py --model_name mdfend   --dataset ch --domain_num 9
python main.py --model_name eann     --dataset ch --domain_num 9
python main.py --model_name eddfn    --dataset ch --domain_num 9
python main.py --model_name mmoe     --dataset ch --domain_num 9
python main.py --model_name mose     --dataset ch --domain_num 9
python main.py --model_name bert     --dataset ch --domain_num 9
python main.py --model_name bigru    --dataset ch --domain_num 9
python main.py --model_name textcnn  --dataset ch --domain_num 9
python main.py --model_name dualemotion --dataset ch --domain_num 9
python main.py --model_name stylelstm --dataset ch --domain_num 9
```

## Citation

If you find this work useful, please cite:

```
@article{zhu2022memory,
  title={Memory-Guided Multi-View Multi-Domain Fake News Detection},
  author={Zhu, Yongchun and Sheng, Qiang and Cao, Juan and Nan, Qiong and Shu, Kai and Wu, Minghui and Wang, Jindong and Zhuang, Fuzhen},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}

@inproceedings{nan2021mdfend,
  title={MDFEND: Multi-domain fake news detection},
  author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3343--3347},
  year={2021}
}
```
