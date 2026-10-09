import os
import torch
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from .kg_fusion import KnowledgeFusionLayer, EnhancedKnowledgeFusionLayer, SoftDomainHead
import logging
import math
from sklearn.cluster import KMeans
import numpy as np
from torch.nn.parameter import Parameter

def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim = 1))

def norm(x):
    length = cal_length(x).view(-1, 1)
    x = x / length
    return x

def convert_to_onehot(label, batch_size, num):
    # 在无 GPU 场景下不强制使用 CUDA，保持与 label 相同的设备或默认 CPU
    if isinstance(label, torch.Tensor):
        device = label.device
    else:
        device = None
    return torch.zeros(batch_size, num, device=device).scatter_(1, label, 1)

class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num = 10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        self.domain_memory = dict()

    def _apply(self, fn):
        # domain_memory 是普通 dict，默认不会随 .cuda()/.to() 迁移，需一并转换
        super()._apply(fn)
        for k in list(self.domain_memory.keys()):
            v = self.domain_memory[k]
            if isinstance(v, torch.Tensor):
                self.domain_memory[k] = fn(v)
        return self

    def forward(self, feature, category):
        feature = norm(feature)
        # 不在这里强制放到 CUDA 上，保持与 feature 相同的设备
        domain_label = torch.tensor([index for index in category]).view(-1, 1).to(feature.device)
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        # Keep shape stable: [B, domain_num, 1] -> [B, domain_num]
        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze(2)
        
        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

        return domain_att

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim = 0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem

class KM3FENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, LNN_dim, domain_num, dataset, k_gate_strength: float = 0.4, bert_finetune: bool = False, use_domain_adapter: bool = False, domain_adapter_scale: float = 0.2, use_domain_moe: bool = False, moe_router_temp: float = 1.0, moe_shared_weight: float = 0.2, moe_confidence_scale: float = 8.0, moe_no_shared: bool = False, moe_soft_routing: bool = False, moe_expert_min_weight: float = 0.0, moe_topk_experts: int = 0, knowledge_mode: str = 'kg', use_kg_cross_attn: bool = True, use_soft_domain: bool = False, soft_domain_beta: float = 0.2, use_bilinear_fusion: bool = False, use_multi_factor_trust: bool = False, use_triple_encoder: bool = False, max_triples: int = 8, use_memory_bank: bool = True):
        super(KM3FENDModel, self).__init__()
        self.domain_num = domain_num
        self.dataset = dataset
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.LNN_dim = LNN_dim
        print('semantic_num_expert:', self.semantic_num_expert, 'emotion_num_expert:', self.emotion_num_expert, 'style_num_expert:', self.style_num_expert, 'lnn_dim:', self.LNN_dim)
        self.fea_size =256
        self.emb_dim = emb_dim
        self.use_domain_adapter = bool(use_domain_adapter)
        self.domain_adapter_scale = float(domain_adapter_scale)
        self.use_domain_moe = bool(use_domain_moe)
        self.moe_router_temp = max(1e-3, float(moe_router_temp))
        self.moe_shared_weight = min(1.0, max(0.0, float(moe_shared_weight)))
        self.moe_confidence_scale = max(1e-3, float(moe_confidence_scale))
        self.moe_no_shared = bool(moe_no_shared)
        self.moe_soft_routing = bool(moe_soft_routing)
        self.moe_expert_min_weight = min(1.0, max(0.0, float(moe_expert_min_weight)))
        self.moe_topk_experts = max(0, int(moe_topk_experts))
        if self.use_domain_moe:
            routing = 'soft' if self.moe_soft_routing else 'hard'
            shared = 'off' if self.moe_no_shared else 'on'
            min_w = self.moe_expert_min_weight if self.moe_soft_routing else 0.0
            topk_s = self.moe_topk_experts if (self.moe_soft_routing and self.moe_topk_experts > 0) else 'all'
            print(
                f'[MoE] routing={routing}, shared_expert={shared}, router_temp={self.moe_router_temp}, '
                f'confidence_scale={self.moe_confidence_scale}, expert_min_weight={min_w}, '
                f'topk_experts={topk_s}'
            )
        if self.dataset == 'ch':
            self.bert = BertModel.from_pretrained(
                '/home/ubuntu/.cache/huggingface/hub/models--hfl--chinese-bert-wwm-ext/snapshots/2a995a880017c60e4683869e817130d8af548486'
            ).requires_grad_(bool(bert_finetune))
        elif self.dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(bool(bert_finetune))
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        content_expert = []
        for i in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        emotion_expert = []
        for i in range(self.emotion_num_expert):
            if self.dataset == 'ch':
                emotion_expert.append(MLP(47 * 5, [256, 320,], dropout, output_layer=False))
            elif self.dataset == 'en':
                emotion_expert.append(MLP(38 * 5, [256, 320,], dropout, output_layer=False))
        self.emotion_expert = nn.ModuleList(emotion_expert)

        style_expert = []
        for i in range(self.style_num_expert):
            if self.dataset == 'ch':
                style_expert.append(MLP(48, [256, 320,], dropout, output_layer=False))
            elif self.dataset == 'en':
                style_expert.append(MLP(32, [256, 320,], dropout, output_layer=False))
        self.style_expert = nn.ModuleList(style_expert)


        self.gate = nn.Sequential(nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.LNN_dim),
                                      nn.Softmax(dim = 1))

        self.attention = MaskAttention(emb_dim)

        # Knowledge-guided modules (K-M3FEND)
        # Map external knowledge embedding K (768) into the same feature space as semantic representation S (emb_dim).
        self.knowledge_linear = nn.Linear(768, self.emb_dim)
        self.knowledge_mode = str(knowledge_mode)
        self.use_kg_cross_attn = bool(use_kg_cross_attn)
        self.use_soft_domain = bool(use_soft_domain)
        self.soft_domain_beta = float(soft_domain_beta)
        self.use_bilinear_fusion = bool(use_bilinear_fusion)
        self.use_multi_factor_trust = bool(use_multi_factor_trust)
        self.use_triple_encoder = bool(use_triple_encoder)
        self.max_triples = int(max_triples)
        self.use_memory_bank = bool(use_memory_bank)
        
        self.knowledge_fusion = EnhancedKnowledgeFusionLayer(
            self.emb_dim,
            dropout=dropout,
            use_bilinear=self.use_bilinear_fusion,
            use_multi_factor_trust=self.use_multi_factor_trust,
            use_triple_encoder=self.use_triple_encoder,
            max_triples=self.max_triples,
        )
        self.soft_domain_head = SoftDomainHead(self.emb_dim, domain_num) if self.use_soft_domain else None
        # K-Gate (legacy llm): S,K only. KG mode uses tau-aware gate.
        self.k_gate = nn.Sequential(
            nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], 2),
            nn.Sigmoid()
        )
        self.k_gate_kg = nn.Sequential(
            nn.Linear(self.emb_dim * 2 + 1, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], 2),
            nn.Sigmoid()
        )
        self.k_gate_strength = float(k_gate_strength)

        # 不强制使用 CUDA，保持与模型相同设备（由 Trainer 决定是否 .cuda()）
        # 注意：必须先 unsqueeze 再包 Parameter；nn.Parameter(x).unsqueeze(0) 会变成普通 Tensor，不会随 .cuda() 迁移。
        self.weight = torch.nn.Parameter(
            torch.Tensor(1, self.LNN_dim, self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert)
        )
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.dataset == 'ch':
            self.domain_memory = MemoryNetwork(input_dim = self.emb_dim + 47 * 5 + 48, emb_dim = self.emb_dim + 47 * 5 + 48, domain_num = self.domain_num, memory_num = self.memory_num)
        elif self.dataset == 'en':
            self.domain_memory = MemoryNetwork(input_dim = self.emb_dim + 38 * 5 + 32, emb_dim = self.emb_dim + 38 * 5 + 32, domain_num = self.domain_num, memory_num = self.memory_num)

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim)
        self.all_feature = {}

        self.classifier = MLP(320, mlp_dims, dropout)
        self.domain_router = nn.Sequential(
            nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.domain_num)
        )
        self.domain_experts = nn.ModuleList([
            MLP(320, mlp_dims, dropout) for _ in range(self.domain_num)
        ])
        self.shared_expert = MLP(320, mlp_dims, dropout)
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(320, 320),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(320, 320)
            )
            for _ in range(self.domain_num)
        ])
        self.domain_adapter_norm = nn.LayerNorm(320)

    def _compute_route_weights(self, domain_att):
        """Softmax routing weights, with optional Top-K filtering and/or min-weight thresholding."""
        route_weights = F.softmax(domain_att / self.moe_router_temp, dim=1)
        if self.moe_topk_experts > 0:
            route_weights = self._topk_route_weights(route_weights)
        if self.moe_expert_min_weight > 0:
            route_weights = self._threshold_route_weights(route_weights)
        return route_weights

    def _topk_route_weights(self, route_weights):
        """
        Keep only the K experts with highest routing weights (per sample),
        set others to zero, then renormalize. K is clamped to [1, domain_num].
        """
        k = min(max(1, self.moe_topk_experts), route_weights.size(1))
        topk_vals, topk_idx = torch.topk(route_weights, k=k, dim=1)
        mask = torch.zeros_like(route_weights)
        mask.scatter_(1, topk_idx, 1.0)
        filtered = route_weights * mask
        filtered = filtered / filtered.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return filtered

    def _threshold_route_weights(self, route_weights):
        """
        Keep experts whose weight >= moe_expert_min_weight, renormalize, then fuse.
        If no expert passes the threshold for a sample, fall back to top-1 expert.
        """
        min_w = self.moe_expert_min_weight
        mask = (route_weights >= min_w).float()
        no_active = mask.sum(dim=1) <= 0
        if no_active.any():
            top1 = route_weights.argmax(dim=1, keepdim=True)
            fallback = torch.zeros_like(route_weights).scatter_(1, top1, 1.0)
            mask = torch.where(no_active.unsqueeze(1), fallback, mask)
        filtered = route_weights * mask
        filtered = filtered / filtered.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return filtered

    def _forward_domain_moe(self, shared_feature, memory_att, category_idx, expert_logits, route_weights=None):
        """
        Domain MoE head with configurable routing:
        - hard: pick expert by ground-truth category (default)
        - soft: weighted sum over selected domain experts via Memory attention
        Shared expert can be disabled for ablation (moe_no_shared).
        """
        domain_att = memory_att.squeeze(1)  # [B, domain_num]
        att_true = torch.gather(domain_att, dim=1, index=category_idx.view(-1, 1)).squeeze(1)
        route_gate = torch.sigmoid((att_true - 0.5) * self.moe_confidence_scale)

        if self.moe_soft_routing:
            if route_weights is None:
                route_weights = self._compute_route_weights(domain_att)
            domain_expert_logit = (expert_logits * route_weights).sum(dim=1)
        else:
            route_weights = None
            domain_expert_logit = torch.gather(
                expert_logits, dim=1, index=category_idx.view(-1, 1)
            ).squeeze(1)

        if self.moe_no_shared:
            final_logit = domain_expert_logit
        else:
            shared_logit = self.shared_expert(shared_feature).squeeze(1)
            final_logit = route_gate * domain_expert_logit + (1.0 - route_gate) * shared_logit

        return final_logit, route_gate, att_true, route_weights

    def _apply_knowledge(
        self,
        semantic_feature,
        knowledge,
        kg_trust=None,
        content_feature=None,
        content_masks=None,
    ):
        """Fuse semantic S with external K; returns memory_semantic, K_proj, tau, sem_scale, style_scale, cos_sk, inconsistency."""
        if knowledge is None or self.knowledge_mode == 'none':
            return semantic_feature, None, None, 1.0, 1.0, None, None

        k_mask = (knowledge.abs().sum(dim=1, keepdim=True) > 0).float()
        knowledge_feature = self.knowledge_linear(knowledge.float())
        knowledge_feature = F.normalize(knowledge_feature, p=2, dim=1)

        if self.knowledge_mode == 'llm':
            memory_semantic = k_mask * knowledge_feature + (1.0 - k_mask) * semantic_feature
            cos = F.cosine_similarity(semantic_feature, knowledge_feature, dim=1).clamp(-1.0, 1.0)
            inconsistency = (1.0 - cos) / 2.0
            kg = self.k_gate(torch.cat([semantic_feature, knowledge_feature], dim=-1))
            sem_scale = 1.0 + self.k_gate_strength * inconsistency.unsqueeze(1) * kg[:, 0:1]
            style_scale = 1.0 - self.k_gate_strength * inconsistency.unsqueeze(1) * kg[:, 1:2]
            style_scale = torch.clamp(style_scale, min=0.0)
            return memory_semantic, knowledge_feature, None, sem_scale, style_scale, cos, inconsistency

        use_cross = self.use_kg_cross_attn and content_feature is not None and content_masks is not None
        if use_cross:
            memory_semantic, tau = self.knowledge_fusion(
                semantic_feature,
                knowledge_feature,
                kg_trust=kg_trust,
                content_feature=content_feature,
                content_masks=content_masks,
            )
        else:
            memory_semantic, tau = self.knowledge_fusion(
                semantic_feature,
                knowledge_feature,
                kg_trust=kg_trust,
            )
        # Keep k_mask as [B, 1] for correct broadcasting with [B, emb_dim].
        memory_semantic = k_mask * memory_semantic + (1.0 - k_mask) * semantic_feature
        memory_semantic = F.normalize(memory_semantic, p=2, dim=1)

        cos = F.cosine_similarity(semantic_feature, knowledge_feature, dim=1).clamp(-1.0, 1.0)
        inconsistency = (1.0 - cos) / 2.0

        gate_in = torch.cat([semantic_feature, knowledge_feature, tau.unsqueeze(1)], dim=-1)
        kg = self.k_gate_kg(gate_in)
        sem_scale = 1.0 + self.k_gate_strength * tau.unsqueeze(1) * kg[:, 0:1]
        style_scale = 1.0 - self.k_gate_strength * tau.unsqueeze(1) * kg[:, 1:2]
        style_scale = torch.clamp(style_scale, min=0.0)
        return memory_semantic, knowledge_feature, tau, sem_scale, style_scale, cos, inconsistency

    def _blend_memory_att(self, memory_att, semantic_feature):
        if not self.use_soft_domain or self.soft_domain_head is None:
            return memory_att
        soft_g = self.soft_domain_head(semantic_feature)
        blended = (1.0 - self.soft_domain_beta) * memory_att.squeeze(1) + self.soft_domain_beta * soft_g
        blended = blended / blended.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return blended.unsqueeze(1)
        
    def forward(self, **kwargs):
        return_features = bool(kwargs.get('return_features', False))

        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        category = kwargs['category']

        knowledge = kwargs.get('knowledge', None)
        kg_trust = kwargs.get('kg_trust', None)

        content_feature = self.bert(content, attention_mask=content_masks)[0]

        # Semantic representation S from the text encoder.
        semantic_feature, _ = self.attention(content_feature, content_masks)
        semantic_feature = F.normalize(semantic_feature, p=2, dim=1)

        memory_semantic, knowledge_feature, tau, sem_scale, style_scale, cos_sk_val, inconsistency_val = self._apply_knowledge(
            semantic_feature,
            knowledge,
            kg_trust=kg_trust,
            content_feature=content_feature,
            content_masks=content_masks,
        )

        if self.use_memory_bank:
            memory_att = self.domain_memory(
                torch.cat([memory_semantic, emotion_feature, style_feature], dim=-1),
                category
            )
            memory_att = self._blend_memory_att(memory_att, semantic_feature)
        else:
            memory_att = torch.ones(content_feature.size(0), 1, self.domain_num, device=content_feature.device) / self.domain_num
        domain_emb_all = self.domain_embedder(torch.LongTensor(range(self.domain_num)).to(content_feature.device))
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in category]).view(-1, 1).to(content_feature.device)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        gate_input = torch.cat([domain_embedding, general_domain_embedding], dim=-1)
        
        gate_value = self.gate(gate_input).view(content_feature.size(0), 1, self.LNN_dim)

        shared_feature = []
        for i in range(self.semantic_num_expert):
            fea = self.content_expert[i](content_feature)
            if not isinstance(sem_scale, float):
                fea = fea * sem_scale
            shared_feature.append(fea.unsqueeze(1))

        for i in range(self.emotion_num_expert):
            shared_feature.append(self.emotion_expert[i](emotion_feature).unsqueeze(1))

        for i in range(self.style_num_expert):
            fea = self.style_expert[i](style_feature)
            if not isinstance(style_scale, float):
                fea = fea * style_scale
            shared_feature.append(fea.unsqueeze(1))

        shared_feature = torch.cat(shared_feature, dim=1)

        embed_x_abs = torch.abs(shared_feature)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        embed_x_log = torch.log1p(embed_x_afn)

        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out)
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        # Keep batch dimension even when B=1: [B, 1, 320] -> [B, 320]
        shared_feature = torch.bmm(gate_value, shared_feature).squeeze(1)
        
        if self.use_domain_adapter:
            adapter_features = []
            for i in range(self.domain_num):
                adapter_features.append(self.domain_adapters[i](shared_feature).unsqueeze(1))
            adapter_features = torch.cat(adapter_features, dim=1)
            adapter_delta = torch.bmm(memory_att, adapter_features).squeeze(1)
            shared_feature = self.domain_adapter_norm(shared_feature + self.domain_adapter_scale * adapter_delta)

        route_gate = None
        route_confidence = None
        route_weights = None
        if self.use_domain_moe:
            category_idx = category.long() if category.dtype != torch.long else category
            route_weights = None
            if self.moe_soft_routing:
                domain_att = memory_att.squeeze(1)
                route_weights = self._compute_route_weights(domain_att)
                active_experts = (route_weights > 0).any(dim=0)
                expert_logits = shared_feature.new_zeros(shared_feature.size(0), self.domain_num)
                for i in range(self.domain_num):
                    if bool(active_experts[i].item()):
                        expert_logits[:, i] = self.domain_experts[i](shared_feature).squeeze(1)
            else:
                expert_logits = []
                for i in range(self.domain_num):
                    expert_logits.append(self.domain_experts[i](shared_feature).squeeze(1).unsqueeze(1))
                expert_logits = torch.cat(expert_logits, dim=1)

            final_logit, route_gate, route_confidence, route_weights = self._forward_domain_moe(
                shared_feature, memory_att, category_idx, expert_logits, route_weights=route_weights
            )
            pred = torch.sigmoid(final_logit)
        else:
            deep_logits = self.classifier(shared_feature)
            pred = torch.sigmoid(deep_logits.squeeze(1))

        if return_features:
            result = {
                'pred': pred,
                'semantic': semantic_feature,
                'knowledge': knowledge_feature,
                'tau': tau,
                'route_gate': route_gate,
                'route_confidence': route_confidence,
                'route_weights': route_weights,
            }
            _B = content_feature.size(0)
            _dev = content_feature.device
            result['cos_sk'] = cos_sk_val if cos_sk_val is not None else torch.zeros(_B, device=_dev)
            result['sem_scale'] = sem_scale
            result['style_scale'] = style_scale
            result['memory_att'] = memory_att.squeeze(1) if memory_att is not None else torch.ones(_B, self.domain_num, device=_dev) / self.domain_num
            k_mask = (knowledge.abs().sum(dim=1) > 0).float() if knowledge is not None else torch.zeros(_B, device=_dev)
            result['knowledge_mask'] = k_mask
            result['pred_prob'] = pred
            return result
        return pred

    def save_feature(self, **kwargs):
        if not self.use_memory_bank:
            return None
        
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        knowledge = kwargs.get('knowledge', None)
        kg_trust = kwargs.get('kg_trust', None)

        content_feature = self.bert(content, attention_mask = content_masks)[0]
        semantic_feature, _ = self.attention(content_feature, content_masks)
        semantic_feature = F.normalize(semantic_feature, p=2, dim=1)

        memory_semantic, _, _, _, _, _, _ = self._apply_knowledge(
            semantic_feature,
            knowledge,
            kg_trust=kg_trust,
            content_feature=content_feature,
            content_masks=content_masks,
        )

        all_feature = torch.cat([memory_semantic, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)

        for index in range(all_feature.size(0)):
            domain = int(category[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).cpu().detach().numpy())

    def init_memory(self):
        device = next(self.parameters()).device
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            # KMeans requires n_samples >= n_clusters. For low-resource / debug runs, pad centers by resampling.
            n_samples = all_feature.shape[0]
            if n_samples <= 0:
                continue
            if n_samples < self.memory_num:
                # Use available samples as initial centers then pad to memory_num by repeating.
                reps = self.memory_num - n_samples
                pad_idx = np.random.choice(n_samples, size=reps, replace=True)
                centers = np.concatenate([all_feature, all_feature[pad_idx]], axis=0)
            else:
                kmeans = KMeans(n_clusters=self.memory_num, init='k-means++').fit(all_feature)
                centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).to(device)
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        if not self.use_memory_bank:
            return
        
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        knowledge = kwargs.get('knowledge', None)
        kg_trust = kwargs.get('kg_trust', None)

        content_feature = self.bert(content, attention_mask = content_masks)[0]
        semantic_feature, _ = self.attention(content_feature, content_masks)
        semantic_feature = F.normalize(semantic_feature, p=2, dim=1)

        memory_semantic, _, _, _, _, _, _ = self._apply_knowledge(
            semantic_feature,
            knowledge,
            kg_trust=kg_trust,
            content_feature=content_feature,
            content_masks=content_masks,
        )

        all_feature = torch.cat([memory_semantic, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)

class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir, 
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 dataset,
                 early_stop = 5,
                 epoches = 100,
                 con_weight: float = 0.0,
                 con_temperature: float = 0.07,
                 k_gate_strength: float = 0.4,
                 knowledge_mode: str = 'kg',
                 use_kg_cross_attn: bool = True,
                 use_soft_domain: bool = False,
                 soft_domain_beta: float = 0.2,
                 soft_domain_loss_weight: float = 0.1,
                 label_smoothing: float = 0.0,
                 con_trust_threshold: float = 0.3,
                 bert_finetune: bool = False,
                 bert_lr: float = 2e-5,
                 bert_unfreeze_layers: int = 0,
                 auto_threshold: bool = False,
                 domain_reweight_loss: bool = False,
                 domain_weight_alpha: float = 0.7,
                 domain_weight_max_ratio: float = 2.0,
                 domain_reweight_warmup_epochs: int = 3,
                 use_domain_adapter: bool = False,
                 domain_adapter_scale: float = 0.2,
                 use_domain_moe: bool = False,
                 moe_router_temp: float = 1.0,
                 moe_shared_weight: float = 0.2,
                 moe_confidence_scale: float = 8.0,
                 moe_no_shared: bool = False,
                 moe_soft_routing: bool = False,
                 moe_expert_min_weight: float = 0.0,
                 moe_topk_experts: int = 0,
                 focal_gamma: float = 0.0,
                 use_bilinear_fusion: bool = False,
                 use_multi_factor_trust: bool = False,
                 use_triple_encoder: bool = False,
                 max_triples: int = 8,
                 use_memory_bank: bool = True,
                 save_case_results: str = ''
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset
        self.con_weight = con_weight
        self.con_temperature = con_temperature
        self.k_gate_strength = k_gate_strength
        self.knowledge_mode = knowledge_mode
        self.use_kg_cross_attn = use_kg_cross_attn
        self.use_soft_domain = use_soft_domain
        self.soft_domain_beta = soft_domain_beta
        self.soft_domain_loss_weight = soft_domain_loss_weight
        self.label_smoothing = label_smoothing
        self.con_trust_threshold = con_trust_threshold
        self.bert_finetune = bool(bert_finetune)
        self.bert_lr = float(bert_lr)
        self.bert_unfreeze_layers = int(bert_unfreeze_layers)
        self.auto_threshold = bool(auto_threshold)
        self.domain_reweight_loss = bool(domain_reweight_loss)
        self.domain_weight_alpha = float(domain_weight_alpha)
        self.domain_weight_max_ratio = float(domain_weight_max_ratio)
        self.domain_reweight_warmup_epochs = int(domain_reweight_warmup_epochs)
        self.use_domain_adapter = bool(use_domain_adapter)
        self.domain_adapter_scale = float(domain_adapter_scale)
        self.use_domain_moe = bool(use_domain_moe)
        self.moe_router_temp = float(moe_router_temp)
        self.moe_shared_weight = float(moe_shared_weight)
        self.moe_confidence_scale = float(moe_confidence_scale)
        self.moe_no_shared = bool(moe_no_shared)
        self.moe_soft_routing = bool(moe_soft_routing)
        self.moe_expert_min_weight = float(moe_expert_min_weight)
        self.moe_topk_experts = int(moe_topk_experts)
        self.focal_gamma = float(focal_gamma)
        self.use_bilinear_fusion = bool(use_bilinear_fusion)
        self.use_multi_factor_trust = bool(use_multi_factor_trust)
        self.use_triple_encoder = bool(use_triple_encoder)
        self.max_triples = int(max_triples)
        self.use_memory_bank = bool(use_memory_bank)
        self.save_case_results = str(save_case_results)
        self.domain_num = len(self.category_dict)
        self.best_threshold = 0.5
        self.domain_weights = None

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def _metrics_with_threshold(self, labels, preds, category, threshold: float):
        # Keep AUC computed from probability scores, only apply threshold for cls metrics.
        base = metrics(labels, preds, category, self.category_dict)
        out = dict(base)

        y_true = np.array(labels).astype(int)
        y_prob = np.array(preds).astype(float)
        y_hat = (y_prob >= float(threshold)).astype(int)

        out['metric'] = f1_score(y_true, y_hat, average='macro')
        out['recall'] = recall_score(y_true, y_hat, average='macro')
        out['precision'] = precision_score(y_true, y_hat, average='macro')
        out['acc'] = accuracy_score(y_true, y_hat)

        # Recompute per-domain classification metrics with threshold, but keep per-domain AUC from base.
        reverse_category_dict = {v: k for k, v in self.category_dict.items()}
        by_domain = {k: {"y_true": [], "y_hat": []} for k in self.category_dict.keys()}
        for i, c in enumerate(category):
            d = reverse_category_dict[int(c)]
            by_domain[d]["y_true"].append(int(y_true[i]))
            by_domain[d]["y_hat"].append(int(y_hat[i]))

        for d, res in by_domain.items():
            out[d] = {
                'precision': round(precision_score(res['y_true'], res['y_hat'], average='macro'), 4),
                'recall': round(recall_score(res['y_true'], res['y_hat'], average='macro'), 4),
                'fscore': round(f1_score(res['y_true'], res['y_hat'], average='macro'), 4),
                'auc': base[d]['auc'],
                'acc': round(accuracy_score(res['y_true'], res['y_hat']), 4),
            }
        return out

    def _search_best_threshold(self, labels, preds, category):
        best_t = 0.5
        best_m = None
        best_f1 = -1.0
        for t in np.linspace(0.1, 0.9, 17):
            m = self._metrics_with_threshold(labels, preds, category, float(t))
            f1 = float(m.get('metric', 0.0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_m = m
        return best_t, best_m

    def _knowledge_contrastive_loss(self, semantic, knowledge, tau=None):
        """
        Trust-weighted InfoNCE between S and K. Skips low-trust samples.
        """
        if knowledge is None:
            return semantic.new_tensor(0.0)
        if knowledge.numel() == 0:
            return semantic.new_tensor(0.0)

        mask = (knowledge.abs().sum(dim=1) > 0)
        if tau is not None:
            mask = mask & (tau >= float(self.con_trust_threshold))
        if mask.sum().item() < 2:
            return semantic.new_tensor(0.0)

        s = F.normalize(semantic[mask], p=2, dim=1)
        k = F.normalize(knowledge[mask], p=2, dim=1)

        logits = torch.matmul(s, k.t()) / float(self.con_temperature)
        targets = torch.arange(logits.size(0), device=logits.device)
        per_sample = F.cross_entropy(logits, targets, reduction='none')
        if tau is not None:
            w = tau[mask].clamp(min=0.0, max=1.0)
            return (per_sample * w).sum() / w.sum().clamp(min=1e-6)
        return per_sample.mean()

    def _build_model(self):
        """Create KM3FENDModel, move to CUDA, configure BERT, and initialize Memory Bank."""
        print('[Trainer] 正在构建 KM3FENDModel（含 BERT/Roberta from_pretrained，首次运行需下载权重）...', flush=True)
        self.model = KM3FENDModel(
            self.emb_dim,
            self.mlp_dims,
            self.dropout,
            self.semantic_num,
            self.emotion_num,
            self.style_num,
            self.lnn_dim,
            len(self.category_dict),
            self.dataset,
            k_gate_strength=self.k_gate_strength,
            knowledge_mode=self.knowledge_mode,
            use_kg_cross_attn=self.use_kg_cross_attn,
            use_soft_domain=self.use_soft_domain,
            soft_domain_beta=self.soft_domain_beta,
            bert_finetune=self.bert_finetune,
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
            use_bilinear_fusion=self.use_bilinear_fusion,
            use_multi_factor_trust=self.use_multi_factor_trust,
            use_triple_encoder=self.use_triple_encoder,
            max_triples=self.max_triples,
            use_memory_bank=self.use_memory_bank,
        )
        if self.use_cuda:
            print('[Trainer] 正在将模型移至 GPU ...', flush=True)
            self.model = self.model.cuda()

        # Configure BERT fine-tuning scope: full or last-N layers.
        for p in self.model.bert.parameters():
            p.requires_grad = False
        if self.bert_finetune:
            if self.bert_unfreeze_layers <= 0:
                for p in self.model.bert.parameters():
                    p.requires_grad = True
            else:
                encoder = getattr(self.model.bert, 'encoder', None)
                layers = getattr(encoder, 'layer', None)
                if layers is not None and len(layers) > 0:
                    n = min(self.bert_unfreeze_layers, len(layers))
                    for layer in layers[-n:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                pooler = getattr(self.model.bert, 'pooler', None)
                if pooler is not None:
                    for p in pooler.parameters():
                        p.requires_grad = True

        # Initialize Memory Bank by iterating through training data once.
        if self.use_memory_bank:
            print('[Trainer] 正在构建 Memory Bank（遍历训练集聚类）...', flush=True)
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                if batch_data['label'].size(0) <= 1:
                    continue
                self.model.save_feature(**batch_data)
            self.model.init_memory()
            print('[Trainer] Memory Bank 初始化完成', flush=True)

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')

        self._build_model()

        # Build per-domain weights from train distribution (for minority-domain boosting).
        if self.domain_reweight_loss:
            train_category = self.train_loader.dataset.tensors[9].cpu().numpy().astype(int)
            counts = np.bincount(train_category, minlength=len(self.category_dict)).astype(np.float32)
            max_count = float(np.max(counts[counts > 0])) if np.any(counts > 0) else 1.0
            weights = np.ones_like(counts, dtype=np.float32)
            nz = counts > 0
            weights[nz] = np.power(max_count / counts[nz], self.domain_weight_alpha)
            weights = np.minimum(weights, np.float32(max(1.0, self.domain_weight_max_ratio)))
            # Normalize around 1.0 to keep loss scale stable.
            mean_w = float(np.mean(weights[nz])) if np.any(nz) else 1.0
            if mean_w > 0:
                weights = weights / mean_w
            self.domain_weights = torch.tensor(weights, dtype=torch.float32)
            if self.use_cuda:
                self.domain_weights = self.domain_weights.cuda()

        if self.bert_finetune:
            bert_params = [p for p in self.model.bert.parameters() if p.requires_grad]
            other_params = [p for n, p in self.model.named_parameters() if not n.startswith('bert.')]
            optimizer = torch.optim.Adam(
                [
                    {'params': bert_params, 'lr': self.bert_lr},
                    {'params': other_params, 'lr': self.lr},
                ],
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)

        # Debug/record: verify whether external knowledge is present in batches.
        knowledge_used = False
        knowledge_nonzero_rate = None
        if self.train_loader:
            for step_n, batch in enumerate(tqdm.tqdm(self.train_loader)):
                batch_data = data2gpu(batch, self.use_cuda)
                if batch_data['label'].size(0) <= 1:
                    continue
                if knowledge_nonzero_rate is None:
                    knowledge_used = 'knowledge' in batch_data and batch_data['knowledge'] is not None
                    if knowledge_used:
                        k = batch_data['knowledge']
                        knowledge_nonzero_rate = float((k.abs().sum(dim=1) > 0).float().mean().item())
                break  # only need first batch to check

        mean_tau_acc = Averager()
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                # Avoid BatchNorm crash on batch_size=1
                if label.size(0) <= 1:
                    continue
                out = self.model(**batch_data, return_features=True)
                label_pred = out['pred']
                if self.label_smoothing > 0:
                    label_target = label.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
                else:
                    label_target = label.float()
                # Domain-aware + focal classification loss.
                bce = F.binary_cross_entropy(label_pred, label_target, reduction='none')
                if self.focal_gamma > 0:
                    pt = label_target * label_pred + (1.0 - label_target) * (1.0 - label_pred)
                    bce = torch.pow((1.0 - pt).clamp(min=1e-6), self.focal_gamma) * bce
                if self.domain_weights is not None:
                    if self.epoches <= max(1, self.domain_reweight_warmup_epochs):
                        reweight_progress = 1.0
                    else:
                        reweight_progress = float(max(0, epoch + 1 - self.domain_reweight_warmup_epochs)) / float(max(1, self.epoches - self.domain_reweight_warmup_epochs))
                        reweight_progress = min(1.0, max(0.0, reweight_progress))
                    sample_w = 1.0 + reweight_progress * (self.domain_weights[category] - 1.0)
                    bce = bce * sample_w
                loss_cls = bce.mean()
                loss_con = self._knowledge_contrastive_loss(out['semantic'], out['knowledge'], tau=out.get('tau'))
                loss_dom = label_pred.new_tensor(0.0)
                if self.use_soft_domain and self.model.soft_domain_head is not None:
                    dom_logits = self.model.soft_domain_head.net(out['semantic'])
                    loss_dom = F.cross_entropy(dom_logits, category.long())
                loss = loss_cls + float(self.con_weight) * loss_con + float(self.soft_domain_loss_weight) * loss_dom
                if out.get('tau') is not None:
                    mean_tau_acc.add(float(out['tau'].mean().item()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    self.model.write(**batch_data)
                avg_loss.add(loss.item())
            if(scheduler is not None):
                # Step scheduler by epoch to avoid overly fast lr decay.
                scheduler.step()
                
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss.item())
            self.model.train()
            if self.auto_threshold:
                val_labels, val_preds, val_category = self.test(self.val_loader, return_raw=True, save_case=False)
                cur_threshold, results = self._search_best_threshold(val_labels, val_preds, val_category)
            else:
                cur_threshold = 0.5
                results = self.test(self.val_loader, save_case=False)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_km3fend.pkl'))
                self.best_mem = self.model.domain_memory.domain_memory
                best_metric = results['metric']
                self.best_threshold = float(cur_threshold)
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_km3fend.pkl')))
        self.model.domain_memory.domain_memory = self.best_mem
        dev = next(self.model.parameters()).device
        for k in list(self.model.domain_memory.domain_memory.keys()):
            t = self.model.domain_memory.domain_memory[k]
            if isinstance(t, torch.Tensor):
                self.model.domain_memory.domain_memory[k] = t.to(dev)
        if self.auto_threshold:
            test_labels, test_preds, test_category = self.test(self.test_loader, return_raw=True)
            results = self._metrics_with_threshold(test_labels, test_preds, test_category, self.best_threshold)
        else:
            results = self.test(self.test_loader)
        # Attach knowledge usage info for downstream logging/analysis.
        results = dict(results)
        results['knowledge_used'] = bool(knowledge_used)
        results['knowledge_nonzero_rate'] = None if knowledge_nonzero_rate is None else round(float(knowledge_nonzero_rate), 4)
        results['con_weight'] = float(self.con_weight)
        results['con_temperature'] = float(self.con_temperature)
        results['k_gate_strength'] = float(self.k_gate_strength)
        results['knowledge_mode'] = str(self.knowledge_mode)
        results['use_kg_cross_attn'] = bool(self.use_kg_cross_attn)
        results['use_soft_domain'] = bool(self.use_soft_domain)
        results['soft_domain_beta'] = float(self.soft_domain_beta)
        results['label_smoothing'] = float(self.label_smoothing)
        results['mean_tau'] = None if mean_tau_acc.n == 0 else round(float(mean_tau_acc.item()), 4)
        results['best_threshold'] = float(self.best_threshold)
        results['domain_reweight_loss'] = bool(self.domain_reweight_loss)
        results['domain_weight_alpha'] = float(self.domain_weight_alpha)
        results['domain_weight_max_ratio'] = float(self.domain_weight_max_ratio)
        results['domain_reweight_warmup_epochs'] = int(self.domain_reweight_warmup_epochs)
        results['use_domain_adapter'] = bool(self.use_domain_adapter)
        results['domain_adapter_scale'] = float(self.domain_adapter_scale)
        results['use_domain_moe'] = bool(self.use_domain_moe)
        results['moe_router_temp'] = float(self.moe_router_temp)
        results['moe_shared_weight'] = float(self.moe_shared_weight)
        results['moe_confidence_scale'] = float(self.moe_confidence_scale)
        results['moe_no_shared'] = bool(self.moe_no_shared)
        results['moe_soft_routing'] = bool(self.moe_soft_routing)
        results['moe_expert_min_weight'] = float(self.moe_expert_min_weight)
        results['moe_topk_experts'] = int(self.moe_topk_experts)
        results['moe_routing_mode'] = 'soft' if self.moe_soft_routing else 'hard'
        results['focal_gamma'] = float(self.focal_gamma)
        # 早停依据为验证集 macro-F1（与 Recorder 一致），供超参搜索选优，避免仅用测试集挑参
        results['val_macro_f1'] = float(recorder.max.get('metric', 0.0))
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_km3fend.pkl')

    def test(self, dataloader, return_raw: bool = False, save_case: bool = True):
        pred = []
        label = []
        category = []
        self.model.eval()
        case_records = []
        global_text_offset = 0
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']

                if self.save_case_results and save_case:
                    result = self.model(**batch_data, return_features=True)
                    batch_label_pred = result['pred_prob']
                else:
                    batch_label_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

                if self.save_case_results and save_case:
                    B = batch_label.size(0)
                    for i in range(B):
                        rec = {}
                        rec['text_idx'] = global_text_offset + i
                        rec['label'] = int(batch_label[i].item())
                        rec['category_idx'] = int(batch_category[i].item())
                        rec['pred_prob'] = float(result['pred_prob'][i].item())
                        if isinstance(result.get('cos_sk'), torch.Tensor) and result['cos_sk'].dim() >= 1:
                            rec['cos_sk'] = float(result['cos_sk'][i].item())
                        else:
                            rec['cos_sk'] = 0.0
                        ss = result.get('sem_scale')
                        if isinstance(ss, torch.Tensor) and ss.dim() >= 1 and ss.size(0) > i:
                            rec['sem_scale'] = float(ss[i].item())
                        elif isinstance(ss, (int, float)):
                            rec['sem_scale'] = float(ss)
                        else:
                            rec['sem_scale'] = 1.0
                        st = result.get('style_scale')
                        if isinstance(st, torch.Tensor) and st.dim() >= 1 and st.size(0) > i:
                            rec['style_scale'] = float(st[i].item())
                        elif isinstance(st, (int, float)):
                            rec['style_scale'] = float(st)
                        else:
                            rec['style_scale'] = 1.0
                        km = result.get('knowledge_mask')
                        if isinstance(km, torch.Tensor) and km.dim() >= 1 and km.size(0) > i:
                            rec['knowledge_mask'] = float(km[i].item())
                        elif isinstance(km, (int, float)):
                            rec['knowledge_mask'] = float(km)
                        else:
                            rec['knowledge_mask'] = 0.0
                        ma = result.get('memory_att')
                        if isinstance(ma, torch.Tensor) and ma.dim() == 2:
                            for d in range(self.domain_num):
                                rec[f'm_att_{d}'] = float(ma[i, d].item())
                        case_records.append(rec)
                    global_text_offset += B
        
        if self.save_case_results and save_case and case_records:
            import pandas as pd
            os.makedirs(os.path.dirname(self.save_case_results) or '.', exist_ok=True)
            pd.DataFrame(case_records).to_csv(self.save_case_results, index=False, encoding='utf-8-sig')
            print(f'[Case Results] Saved to {self.save_case_results} (rows={len(case_records)})', flush=True)
        
        if return_raw:
            return label, pred, category
        return metrics(label, pred, category, self.category_dict)
