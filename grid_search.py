import torch
import tqdm
import pickle
import logging
import os
import time
import json
from copy import deepcopy

from utils.utils import Averager
from utils.dataloader import bert_data
from models.textcnn import Trainer as TextCNNTrainer
from models.bigru import Trainer as BiGRUTrainer
from models.bert import Trainer as BertTrainer
from models.eann import Trainer as EANNTrainer
from models.eddfn import Trainer as EDDFNTrainer
from models.mmoe import Trainer as MMoETrainer
from models.mose import Trainer as MoSETrainer
from models.mdfend import Trainer as MDFENDTrainer
from models.km3fend import Trainer as KM3FENDTrainer
from models.dualemotion import Trainer as DualEmotionTrainer
from models.stylelstm import Trainer as StyleLstmTrainer


def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump


def _append_experiment_markdown(md_path: str, config: dict, metrics: dict) -> None:
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    model_name = str(config.get("model_name"))
    dataset = str(config.get("dataset"))
    domain_num = config.get("domain_num")
    lr = config.get("lr")
    batchsize = config.get("batchsize")
    epoch = config.get("epoch")

    con_weight = metrics.get("con_weight", config.get("con_weight"))
    con_temperature = metrics.get("con_temperature", config.get("con_temperature"))
    k_gate_strength = metrics.get("k_gate_strength", config.get("k_gate_strength"))
    knowledge_used = metrics.get("knowledge_used", None)
    knowledge_nonzero_rate = metrics.get("knowledge_nonzero_rate", None)

    # overall metrics
    overall_f1 = metrics.get("metric", None)  # metrics() 中把 macro-F1 存在 'metric'
    overall_acc = metrics.get("acc", None)
    overall_auc = metrics.get("auc", None)

    # per-domain metrics: auto-discover dict entries like {"政治": {...}, "auc":..., "metric":...}
    per_domain = []
    for name, m in (metrics or {}).items():
        if not isinstance(m, dict):
            continue
        per_domain.append(
            {
                "domain": str(name),
                "f1": m.get("fscore", None),
                "acc": m.get("acc", None),
                "auc": m.get("auc", None),
                "precision": m.get("precision", None),
                "recall": m.get("recall", None),
            }
        )
    per_domain.sort(key=lambda x: x["domain"])

    # Use a stable, extensible format: overall summary row + a per-domain subtable.
    overall_header = (
        "| time | model | dataset | domain_num | lr | batch | epoch | con_w | tau | k_gate | knowledge_used | k_nonzero_rate | overall_f1 | overall_acc | overall_auc |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|\n"
    )
    overall_row = (
        f"| {ts} | {model_name} | {dataset} | {domain_num} | {lr} | {batchsize} | {epoch} | "
        f"{con_weight} | {con_temperature} | {k_gate_strength} | {knowledge_used} | {knowledge_nonzero_rate} | "
        f"{overall_f1} | {overall_acc} | {overall_auc} |\n"
    )

    domain_table = ""
    if per_domain:
        domain_table += "\n"
        domain_table += "| domain | f1 | acc | auc | precision | recall |\n"
        domain_table += "|---|---:|---:|---:|---:|---:|\n"
        for d in per_domain:
            domain_table += (
                f"| {d['domain']} | {d['f1']} | {d['acc']} | {d['auc']} | {d['precision']} | {d['recall']} |\n"
            )
        domain_table += "\n"

    need_header = True
    if os.path.exists(md_path):
        try:
            need_header = os.path.getsize(md_path) == 0
        except OSError:
            need_header = True
    with open(md_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write("## 实验结果自动记录\n\n")
            f.write(overall_header)
        f.write(overall_row)
        # For variable domain counts, append a subtable per run.
        if domain_table:
            f.write(domain_table)

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.weight_decay = config['weight_decay']
        self.lr = config['lr']
        self.epoch = config['epoch']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.early_stop = config['early_stop']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.save_log_dir = config['save_log_dir']
        self.save_param_dir = config['save_param_dir']
        self.param_log_dir = config['param_log_dir']

        self.semantic_num = config['semantic_num']
        self.emotion_num = config['emotion_num']
        self.style_num = config['style_num']
        self.lnn_dim = config['lnn_dim']
        self.domain_num = config['domain_num']
        self.category_dict = config['category_dict']
        self.dataset = config['dataset']
        self.con_weight = config.get('con_weight', 0.0)
        self.con_temperature = config.get('con_temperature', 0.07)
        self.k_gate_strength = config.get('k_gate_strength', 0.4)
        self.knowledge_mode = config.get('knowledge_mode', 'kg')
        self.use_kg_cross_attn = bool(config.get('use_kg_cross_attn', True))
        self.use_soft_domain = bool(config.get('use_soft_domain', False))
        self.soft_domain_beta = float(config.get('soft_domain_beta', 0.2))
        self.soft_domain_loss_weight = float(config.get('soft_domain_loss_weight', 0.1))
        self.label_smoothing = float(config.get('label_smoothing', 0.0))
        self.con_trust_threshold = float(config.get('con_trust_threshold', 0.3))
        self.bert_finetune = bool(config.get('bert_finetune', False))
        self.bert_lr = float(config.get('bert_lr', 2e-5))
        self.auto_threshold = bool(config.get('auto_threshold', False))
        self.balanced_domain_sampler = bool(config.get('balanced_domain_sampler', False))
        self.domain_reweight_loss = bool(config.get('domain_reweight_loss', False))
        self.domain_weight_alpha = float(config.get('domain_weight_alpha', 0.7))
        self.domain_weight_max_ratio = float(config.get('domain_weight_max_ratio', 2.0))
        self.domain_reweight_warmup_epochs = int(config.get('domain_reweight_warmup_epochs', 3))
        self.use_domain_adapter = bool(config.get('use_domain_adapter', False))
        self.domain_adapter_scale = float(config.get('domain_adapter_scale', 0.2))
        self.use_domain_moe = bool(config.get('use_domain_moe', False))
        self.moe_router_temp = float(config.get('moe_router_temp', 1.0))
        self.moe_shared_weight = float(config.get('moe_shared_weight', 0.2))
        self.moe_confidence_scale = float(config.get('moe_confidence_scale', 8.0))
        self.moe_no_shared = bool(config.get('moe_no_shared', False))
        self.moe_soft_routing = bool(config.get('moe_soft_routing', False))
        self.moe_expert_min_weight = float(config.get('moe_expert_min_weight', 0.0))
        self.moe_topk_experts = int(config.get('moe_topk_experts', 0))
        self.focal_gamma = float(config.get('focal_gamma', 0.0))
        self.use_bilinear_fusion = bool(config.get('use_bilinear_fusion', False))
        self.use_multi_factor_trust = bool(config.get('use_multi_factor_trust', False))
        self.use_triple_encoder = bool(config.get('use_triple_encoder', False))
        self.max_triples = int(config.get('max_triples', 8))
        self.use_memory_bank = bool(config.get('use_memory_bank', True))
        self.save_case_results = str(config.get('save_case_results', ''))
        self.mode = str(config.get('mode', 'train'))
        self.load_ckpt = str(config.get('load_ckpt', ''))
        self.bert_unfreeze_layers = int(config.get('bert_unfreeze_layers', 0))
        if self.balanced_domain_sampler and self.domain_reweight_loss:
            # Avoid double compensation: prefer loss reweighting only.
            self.balanced_domain_sampler = False


        self.train_path = self.root_path + 'train.pkl'
        self.val_path = self.root_path + 'val.pkl'
        self.test_path = self.root_path + 'test.pkl'
        
    
    def get_dataloader(self):
        loader = bert_data(max_len = self.max_len, batch_size = self.batchsize,
                        category_dict = self.category_dict, num_workers=self.num_workers, dataset = self.dataset, balanced_domain_sampler=self.balanced_domain_sampler)
        loader.knowledge_mode = self.knowledge_mode
        train_loader = loader.load_data(self.train_path, True)
        val_loader = loader.load_data(self.val_path, False)
        test_loader = loader.load_data(self.test_path, False)
        return train_loader, val_loader, test_loader

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def _create_trainer(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        if self.model_name == 'textcnn':
            trainer = TextCNNTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, 
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch, dataset = self.dataset,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'eann':
            trainer = EANNTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims,  dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch, domain_num = self.domain_num,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'eddfn':
            trainer = EDDFNTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch, domain_num = self.domain_num,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'bigru':
            trainer = BiGRUTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, num_layers = 1, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'mmoe':
            trainer = MMoETrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, num_layers = 1, early_stop = self.early_stop, epoches = self.epoch, 
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'mdfend':
            trainer = MDFENDTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'mose':
            trainer = MoSETrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, num_layers = 1, early_stop = self.early_stop, epoches = self.epoch, 
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'bert':
            trainer = BertTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'km3fend':
            trainer = KM3FENDTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=self.mlp_dims,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                dropout=self.dropout,
                weight_decay=self.weight_decay,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name),
                semantic_num=self.semantic_num,
                emotion_num=self.emotion_num,
                style_num=self.style_num,
                lnn_dim=self.lnn_dim,
                dataset=self.dataset,
                con_weight=self.con_weight,
                con_temperature=self.con_temperature,
                k_gate_strength=self.k_gate_strength,
                knowledge_mode=self.knowledge_mode,
                use_kg_cross_attn=self.use_kg_cross_attn,
                use_soft_domain=self.use_soft_domain,
                soft_domain_beta=self.soft_domain_beta,
                soft_domain_loss_weight=self.soft_domain_loss_weight,
                label_smoothing=self.label_smoothing,
                con_trust_threshold=self.con_trust_threshold,
                bert_finetune=self.bert_finetune,
                bert_lr=self.bert_lr,
                bert_unfreeze_layers=self.bert_unfreeze_layers,
                auto_threshold=self.auto_threshold,
                domain_reweight_loss=self.domain_reweight_loss,
                domain_weight_alpha=self.domain_weight_alpha,
                domain_weight_max_ratio=self.domain_weight_max_ratio,
                domain_reweight_warmup_epochs=self.domain_reweight_warmup_epochs,
                use_domain_adapter=self.use_domain_adapter,
                domain_adapter_scale=self.domain_adapter_scale,
                use_domain_moe=self.use_domain_moe,
                moe_router_temp=self.moe_router_temp,
                moe_shared_weight=self.moe_shared_weight,
                moe_confidence_scale=self.moe_confidence_scale,
                moe_no_shared=self.moe_no_shared,
                moe_soft_routing=self.moe_soft_routing,
                moe_expert_min_weight=self.moe_expert_min_weight,
                moe_topk_experts=self.moe_topk_experts,
                focal_gamma=self.focal_gamma,
                use_bilinear_fusion=self.use_bilinear_fusion,
                use_multi_factor_trust=self.use_multi_factor_trust,
                use_triple_encoder=self.use_triple_encoder,
                max_triples=self.max_triples,
                use_memory_bank=self.use_memory_bank,
                save_case_results=self.save_case_results,
            )
        elif self.model_name == 'dualemotion':
            trainer = DualEmotionTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        elif self.model_name == 'stylelstm':
            trainer = StyleLstmTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, dataset = self.dataset,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))
        else:
            raise ValueError("unknown model_name: %s" % self.model_name)
        return trainer

    def run_training(self, logger=None):
        param_log_dir = self.param_log_dir
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.model_name +'_'+ 'oneloss_param.txt')
        if logger is None:
            logger = self.getFileLogger(param_log_file)
        print('[Run] 正在构建 DataLoader（读取 pkl + 对全文逐条 BERT 分词，数据大时可能要几分钟且无进度条）...', flush=True)
        trainer = self._create_trainer()
        print('[Run] DataLoader 就绪，开始训练（首次会从 HuggingFace 拉取/加载中文 BERT，也可能较慢）...', flush=True)
        metrics, model_path = trainer.train(logger)
        return metrics, model_path, logger

    def run_test_only(self, logger=None):
        param_log_dir = self.param_log_dir
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.model_name +'_'+ 'test_only.txt')
        if logger is None:
            logger = self.getFileLogger(param_log_file)
        print('[Run-Test] 正在构建 DataLoader ...', flush=True)
        trainer = self._create_trainer()
        trainer._build_model()
        ckpt_path = self.load_ckpt
        if ckpt_path and os.path.exists(ckpt_path):
            print(f'[Run-Test] Loading checkpoint from {ckpt_path} ...', flush=True)
            state_dict = torch.load(ckpt_path, map_location=next(trainer.model.parameters()).device)
            trainer.model.load_state_dict(state_dict)
        else:
            print('[Run-Test] WARNING: --load_ckpt not provided or file not found, using initialized weights', flush=True)
        trainer.model.eval()
        print('[Run-Test] Running test ...', flush=True)
        if trainer.auto_threshold:
            test_labels, test_preds, test_category = trainer.test(trainer.test_loader, return_raw=True, save_case=True)
            results = trainer._metrics_with_threshold(test_labels, test_preds, test_category, trainer.best_threshold)
        else:
            results = trainer.test(trainer.test_loader, save_case=True)
        print(f'[Run-Test] Results: {results}', flush=True)
        logger.info(f'Test-only results: {results}')
        return results, ckpt_path, logger

    def main(self):
        if self.mode == 'test':
            metrics, model_path, logger = self.run_test_only()
        else:
            metrics, model_path, logger = self.run_training()
        # Save JSON with dataset/domain/timestamp to avoid overwriting across experiments
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        km_tag = 'kg' if self.configinfo.get('knowledge_mode') != 'none' else 'nokg'
        json_path = f'./logs/json/{self.model_name}_{self.dataset}_d{self.domain_num}_{km_tag}_{ts}.json'
        json_dir = os.path.dirname(json_path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        json_result = [metrics]
        # Also append to a stable aggregate JSON for easy comparison
        _append_experiment_markdown(
            md_path='./docs/experiment_results.md',
            config=self.configinfo,
            metrics=metrics,
        )
        print("best model path:", model_path)
        print("best metric:", metrics)
        logger.info("best model path:" + str(model_path))
        logger.info("best metric:" + str(metrics))
        logger.info('--------------------------------------\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)
        print(f'[Results] JSON saved -> {json_path}')
        print(f'[Results] Markdown table appended -> docs/experiment_results.md')
