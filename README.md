# K-M3FEND

**Knowledge-Guided Adaptive Expert Routing for Multi-View Multi-Domain Fake News Detection**

This repository contains the official PyTorch implementation of K-M3FEND.

> The rapid proliferation of fake news on social media poses severe threats to public perception and information credibility. In real-world applications, news content is distributed across multiple domains, where differences in text, emotional expressions, and writing styles make it difficult for a single model to capture both domain-specific and domain-shared characteristics. Existing multi-domain fake news detection methods mainly rely on textual representations for domain modeling, but domain memory and expert routing mechanisms may be less reliable when textual expressions are noisy or domains have similar characteristics. External knowledge has been widely used to provide complementary factual information, but it is often incorporated as auxiliary information, while retrieved knowledge may also contain irrelevant or ambiguous information.

## Overview

K-M3FEND integrates external knowledge, multi-view representations, domain memory retrieval, and expert-based classification to model domain-specific and cross-domain characteristics in multi-domain fake news detection. The framework comprises four main components:

1. **External Knowledge Extraction** — Entities are linked to a domain-specific knowledge graph constructed from the training corpus using part-of-speech filtering and document-frequency pruning after Jieba segmentation. Seed triples and corpus-derived co-occurrence triples are incorporated into the knowledge base. For each linked entity, the associated relational triples are retrieved, serialized into textual sequences, and encoded by a frozen copy of the pretrained text encoder (BERT-base-Chinese for Chinese, RoBERTa-base for English). Mean pooling produces a 768-dimensional knowledge representation. A multi-factor trust score `kg_trust = 0.35·s_ent + 0.35·s_trip + 0.15·s_ovl + 0.15·s_struct` estimates knowledge reliability.

2. **Multi-View Expert Extraction** — The aggregated sentence representation is fed into 16 sub-experts partitioned into three views: seven semantic experts (CNN-based extraction with kernel sizes {1, 2, 3, 5, 10}), seven emotion experts (MLP-based), and two style experts (MLP-based). Each sub-expert produces a 320-dimensional representation. The outputs are concatenated to form a 5120-dimensional multi-view representation, which is subsequently processed by LNN fusion and domain-gating pathways.

3. **Knowledge-Enhanced Domain Memory Bank** — The global memory bank `M = {M_1, M_2, ..., M_K}` contains K domain-specific memory blocks (N=10 slots per block, d_m=1051 for Chinese). External knowledge is first fused with the semantic representation `s'` to construct a knowledge-enhanced memory query `q_in = [s'; e; f]`, reducing the influence of textual noise on domain retrieval. A two-stage hierarchical attention mechanism (domain-level α_k + slot-level β_{k,n}) retrieves the memory representation.

4. **Domain Mixture-of-Experts with Knowledge-Guided Gating** — A domain-specific expert `E_d` and a shared expert `E_s` produce logits that are fused via a routing score `g_d = σ(λ(α_d − 0.5))` derived from memory retrieval confidence. The Knowledge-Guided Gate (K-Gate) dynamically adjusts the contributions of semantic and stylistic features: `w_sem = 1 + β·τ·g_sem(...)`, `w_sty = 1 − β·τ·g_sty(...)`, where τ is estimated by the KnowledgeTrustGate from the semantic representation, knowledge representation, their cosine similarity, and the knowledge trust score. An InfoNCE-based contrastive loss aligns textual and knowledge representations.

## Repository Structure

```
K-M3FEND/
├── main.py                        # Entry point with argument parsing
├── grid_search.py                 # Training & evaluation runner
├── requirements.txt
├── data/
│   ├── ch/                        # Chinese Weibo21 dataset
│   │   ├── train.pkl / val.pkl / test.pkl
│   │   ├── train_kg.jsonl / val_kg.jsonl / test_kg.jsonl
│   │   ├── train_kg_knowledge.npy / val_kg_knowledge.npy / test_kg_knowledge.npy
│   │   └── train_kg_trust.npy / val_kg_trust.npy / test_kg_trust.npy
│   └── en/                        # English dataset (same structure)
├── models/
│   ├── km3fend.py                 # K-M3FEND model & trainer
│   ├── kg_fusion.py               # Knowledge fusion, K-Gate, trust gate
│   ├── layers.py                  # Shared layers (CNN, MLP, MaskAttention, LNN)
│   ├── bert.py                    # BERT/RoBERTa baseline
│   ├── bigru.py                   # BiGRU baseline
│   ├── textcnn.py                 # TextCNN baseline
│   ├── eann.py                    # EANN baseline
│   ├── eddfn.py                   # EDDFN baseline
│   ├── mmoe.py                    # MMoE baseline
│   ├── mose.py                    # MoSE baseline
│   ├── mdfend.py                  # MDFEND baseline
│   ├── dualemotion.py             # DualEmotion baseline
│   └── stylelstm.py               # StyleLSTM baseline
├── utils/
│   ├── dataloader.py              # Data loading & KG knowledge injection
│   └── utils.py                   # Metrics (Acc, Prec, Rec, F1, AUC) & helpers
└── scripts/
    ├── prepare_kg_knowledge.py    # End-to-end KG knowledge preparation pipeline
    ├── build_kg_from_corpus.py    # Build KG (entities.txt + triples.tsv) from corpus
    └── build_knowledge_from_kg.py # Embed KG triples → knowledge vectors & trust scores
```

## Requirements

- Python 3.6+
- PyTorch >= 1.0
- Transformers >= 4.0
- Pandas, NumPy, scikit-learn, tqdm, jieba

```bash
pip install -r requirements.txt
```

## Datasets

### Chinese: Weibo21

The Weibo21 dataset is collected from Sina Weibo, containing 9,592 news samples (4,641 fake / 4,951 real) across nine domains:

| Domain | Fake | Real | Total |
|---|---:|---:|---:|
| Society | 512 | 548 | 1,060 |
| Entertainment | 623 | 641 | 1,264 |
| Finance | 487 | 521 | 1,008 |
| Health | 534 | 556 | 1,090 |
| Politics | 612 | 598 | 1,210 |
| Disaster | 445 | 467 | 912 |
| Education | 398 | 412 | 810 |
| Military | 421 | 439 | 860 |
| Science | 609 | 769 | 1,378 |
| **Total** | **4,641** | **4,951** | **9,592** |

Supports `domain_num` of 3, 6, or 9. The default 9-domain configuration matches the predefined Weibo21 taxonomy.

### English: 3-Domain

Three domains: GossipCop, PolitiFact, and COVID. Training, validation, and test sets contain 19,401, 5,660, and 5,747 samples, respectively. The text encoder is RoBERTa-base, and DBpedia is used as the external knowledge source.

### Data Files

Each split (`train`/`val`/`test`) includes:

| File | Description |
|---|---|
| `{split}.pkl` | Raw data (content, comments, emotions, style features, label, category) |
| `{split}_kg.jsonl` | KG entities and triples per sample |
| `{split}_kg_knowledge.npy` | Embedded KG knowledge vectors `[N, 768]` |
| `{split}_kg_trust.npy` | KG multi-factor trust scores `[N, 1]` |

## Usage

### Quick Start

```bash
# K-M3FEND on Chinese 9-domain (main experiment)
python main.py --model_name km3fend --dataset ch --domain_num 9 --gpu 0

# K-M3FEND on English 3-domain
python main.py --model_name km3fend --dataset en --domain_num 3 --gpu 0
```

### KG Knowledge Preparation

If `{split}_kg_knowledge.npy` files are not yet generated, run the preparation script:

```bash
python scripts/prepare_kg_knowledge.py --dataset ch
python scripts/prepare_kg_knowledge.py --dataset en
```

Or use the `--kg_pack` flag to auto-prepare KG knowledge before training:

```bash
python main.py --model_name km3fend --dataset ch --domain_num 9 --kg_pack
```

### Hyperparameters

All hyperparameters follow the settings reported in the paper (Table 2):

| Hyperparameter | Value | CLI Argument |
|---|---|---|
| Text encoder (Chinese) | BERT-base-Chinese | `--dataset ch` |
| Text encoder (English) | RoBERTa-base | `--dataset en` |
| Hidden dimension | 768 | `--emb_dim 768` |
| Learning rate (other params) | 1×10⁻⁴ | `--lr 0.0001` |
| Learning rate (encoder fine-tuning) | 2×10⁻⁵ | `--bert_lr 2e-5` |
| Batch size | 64 | `--batchsize 64` |
| Max sequence length | 170 | `--max_len 170` |
| Training epochs | 50 (with early stopping) | `--epoch 50` |
| Early-stopping patience | 5 | `--early_stop 5` |
| Memory slots per domain (N) | 10 | — |
| Memory write rate (η) | 0.05 | — |
| Contrastive loss weight (λ_con) | 0.030 | `--con_weight 0.03` |
| Contrastive temperature (τ_c) | 0.123 | `--con_temperature 0.123` |
| K-Gate intensity (β) | 0.40 | `--k_gate_strength 0.4` |
| MoE routing scale (λ) | 3.5 | `--moe_confidence_scale 3.5` |
| Label smoothing | 0.1 | `--label_smoothing 0.1` |
| Dropout rate | 0.1 | — |
| Weight decay | 5×10⁻⁵ | — |
| Optimizer | Adam | — |
| LR scheduler | StepLR (step=100, γ=0.98) | — |
| Gradient clipping | 1.0 | — |
| Random seed | 2021 | `--seed 2021` |

### Multiple Runs

All experiments in the paper are conducted over 10 independent runs with results averaged. The k-th run uses seed = 2021 + (k−1):

```bash
python main.py --model_name km3fend --dataset ch --domain_num 9 --n_runs 10 --seed 2021 --seed_step 1
```

### Baselines

The repository includes twelve baseline methods across three categories:

**Text-only representation:** TextCNN, BiGRU, BERT

**Feature-enhanced:** StyleLSTM, DualEmotion

**Multi-domain learning:** EANN, MMoE, MoSE, EDDFN, MDFEND, M³FEND, DHEM-FND

```bash
python main.py --model_name textcnn     --dataset ch --domain_num 9
python main.py --model_name bigru       --dataset ch --domain_num 9
python main.py --model_name bert        --dataset ch --domain_num 9
python main.py --model_name stylelstm   --dataset ch --domain_num 9
python main.py --model_name dualemotion --dataset ch --domain_num 9
python main.py --model_name eann        --dataset ch --domain_num 9
python main.py --model_name mmoe        --dataset ch --domain_num 9
python main.py --model_name mose        --dataset ch --domain_num 9
python main.py --model_name eddfn       --dataset ch --domain_num 9
python main.py --model_name mdfend      --dataset ch --domain_num 9
```

### Ablation Study

Remove individual modules via command-line flags:

```bash
# w/o External Knowledge
python main.py --model_name km3fend --dataset ch --domain_num 9 --knowledge_mode none

# w/o Contrastive Learning
python main.py --model_name km3fend --dataset ch --domain_num 9 --con_weight 0

# w/o Domain MoE (use single classifier head)
python main.py --model_name km3fend --dataset ch --domain_num 9 --use_domain_moe False

# w/o K-Gate (disable knowledge-guided gating)
python main.py --model_name km3fend --dataset ch --domain_num 9 --k_gate_strength 0

# w/o Memory Bank
python main.py --model_name km3fend --dataset ch --domain_num 9 --no_memory_bank
```

## Results

### Main Results (Chinese Weibo21, 9 Domains)

| Method | F1 | ACC | AUC |
|---|---:|---:|---:|
| TextCNN | 0.8800 | 0.8802 | 0.9520 |
| BiGRU | 0.8748 | 0.8750 | 0.9439 |
| BERT | 0.8953 | 0.8955 | 0.9560 |
| StyleLSTM | 0.8997 | 0.8998 | 0.9613 |
| DualEmotion | 0.8962 | 0.8963 | 0.9602 |
| EANN | 0.8993 | 0.8994 | 0.9620 |
| EDDFN | 0.8856 | 0.8858 | 0.9478 |
| MMoE | 0.8893 | 0.8895 | 0.9546 |
| MoSE | 0.8817 | 0.8818 | 0.9467 |
| MDFEND | 0.9142 | 0.9143 | 0.9711 |
| DHEM-FND | 0.8389 | 0.8391 | 0.9071 |
| M³FEND | 0.9165 | 0.9166 | 0.9736 |
| **K-M3FEND (Ours)** | **0.9256** | **0.9256** | **0.9778** |

### English 3-Domain Results

| Method | GossipCop | PolitiFact | COVID | F1 | ACC | AUC |
|---|---:|---:|---:|---:|---:|---:|
| MDFEND | 0.7980 | 0.8284 | 0.9306 | 0.8299 | 0.8863 | 0.9126 |
| M³FEND | 0.8073 | 0.8478 | 0.9304 | 0.8292 | 0.8839 | 0.9241 |
| **K-M3FEND (Ours)** | **0.8286** | **0.8621** | **0.9596** | **0.8599** | **0.9035** | **0.9401** |

### Ablation Study (Weibo21, 9 Domains)

| Variant | ACC | F1 | AUC |
|---|---:|---:|---:|
| K-M3FEND (Full) | 0.9256 | 0.9256 | 0.9778 |
| w/o External Knowledge | 0.9218 | 0.9215 | 0.9751 |
| w/o Contrastive Learning | 0.9234 | 0.9231 | 0.9763 |
| w/o Domain MoE | 0.9196 | 0.9192 | 0.9738 |
| w/o K-Gate | 0.9225 | 0.9221 | 0.9768 |
| w/o Memory Bank | 0.9203 | 0.9199 | 0.9745 |

## Citation

If you find this work useful, please cite:

```bibtex
@article{gong2026km3fend,
  title={Knowledge-Guided Adaptive Expert Routing for Multi-View Multi-Domain Fake News Detection},
  author={Gong, Chao and Liu, Yu},
  journal={Expert Systems with Applications},
  year={2026}
}
```

This work builds upon M³FEND and MDFEND:

```bibtex
@article{zhu2023memory,
  title={Memory-Guided Multi-View Multi-Domain Fake News Detection},
  author={Zhu, Yongchun and Sheng, Qiang and Cao, Juan and Nan, Qiong and Shu, Kai and Wu, Minghui and Wang, Jindong and Zhuang, Fuzhen},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={35},
  number={5},
  pages={4244--4257},
  year={2023},
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
