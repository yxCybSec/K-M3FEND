import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class KnowledgeTrustGate(nn.Module):
    """Compute trust tau in [0,1] from S, K, offline kg_trust, and cosine similarity."""

    def __init__(self, emb_dim: int, mlp_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        semantic: torch.Tensor,
        knowledge: torch.Tensor,
        kg_trust: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cos = F.cosine_similarity(semantic, knowledge, dim=1).clamp(-1.0, 1.0).unsqueeze(1)
        if kg_trust is None:
            kg_trust = cos
        elif kg_trust.dim() == 1:
            kg_trust = kg_trust.unsqueeze(1)
        gate_in = torch.cat([semantic, knowledge, cos, kg_trust.float()], dim=-1)
        return self.mlp(gate_in)


class MultiFactorTrustGate(nn.Module):
    """
    多因子信任评分系统：
    1. 语义一致性：cosine similarity 校准版本
    2. 实体覆盖率：三元组实体在文本中的出现比例
    3. 关系相关性：关系类型与领域的匹配度
    4. 不确定性建模：用方差表示信任的置信度
    """

    def __init__(self, emb_dim: int, mlp_hidden: int = 128):
        super().__init__()
        self.cos_scale = nn.Parameter(torch.tensor(1.0))
        self.cos_shift = nn.Parameter(torch.tensor(0.0))
        
        self.trust_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 5, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(
        self,
        semantic: torch.Tensor,
        knowledge: torch.Tensor,
        kg_trust: Optional[torch.Tensor] = None,
        entity_coverage: Optional[torch.Tensor] = None,
        relation_relevance: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_raw = F.cosine_similarity(semantic, knowledge, dim=1)
        cos_calibrated = (cos_raw * self.cos_scale + self.cos_shift).clamp(-1.0, 1.0)
        
        if kg_trust is None:
            kg_trust = cos_calibrated
        elif kg_trust.dim() == 1:
            kg_trust = kg_trust
        else:
            kg_trust = kg_trust.squeeze(1)
        
        if entity_coverage is None:
            entity_coverage = cos_calibrated
        if relation_relevance is None:
            relation_relevance = cos_calibrated
        if uncertainty is None:
            uncertainty = torch.zeros_like(cos_calibrated)
        
        inputs = torch.cat([
            semantic, 
            knowledge, 
            cos_calibrated.unsqueeze(1),
            kg_trust.unsqueeze(1),
            entity_coverage.unsqueeze(1),
            relation_relevance.unsqueeze(1),
            uncertainty.unsqueeze(1),
        ], dim=-1)
        
        outputs = self.trust_mlp(inputs)
        tau = torch.sigmoid(outputs[:, 0])
        confidence = torch.sigmoid(outputs[:, 1])
        
        tau = tau * confidence + (1 - confidence) * kg_trust
        
        return tau.unsqueeze(1), confidence


class TripleAwareEncoder(nn.Module):
    """
    关系感知三元组编码器：
    - 为每条三元组添加关系类型嵌入
    - 使用注意力机制动态选择最相关的三元组
    - 保留三元组的结构信息（头实体-关系-尾实体）
    """

    def __init__(self, emb_dim: int, max_triples: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_triples = max_triples
        
        self.relation_embeddings = nn.Parameter(torch.randn(32, emb_dim))
        
        self.triple_encoder = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.attention_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

    def forward(
        self,
        head_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        tail_embeddings: torch.Tensor,
        semantic: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = semantic.size(0)
        
        if head_embeddings is None or head_embeddings.size(1) == 0:
            return torch.zeros(batch_size, self.emb_dim, device=semantic.device), torch.zeros(batch_size, device=semantic.device)
        
        num_triples = head_embeddings.size(1)
        
        relation_emb = self.relation_embeddings[relation_embeddings.long()]
        
        triple_features = torch.cat([
            head_embeddings,
            relation_emb,
            tail_embeddings,
        ], dim=-1)
        
        triple_features = self.triple_encoder(triple_features)
        
        semantic_expanded = semantic.unsqueeze(1).repeat(1, num_triples, 1)
        attention_input = torch.cat([semantic_expanded, triple_features], dim=-1)
        attention_scores = self.attention_proj(attention_input).squeeze(-1)
        
        if masks is not None:
            attention_scores = attention_scores.masked_fill(masks == 0, -1e4)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        aggregated_knowledge = torch.bmm(attention_weights.unsqueeze(1), triple_features).squeeze(1)
        
        max_weight, _ = attention_weights.max(dim=-1)
        
        return aggregated_knowledge, max_weight


class KnowledgeFusionLayer(nn.Module):
    """
    Gated cross-attention style fusion: S' = normalize(S + tau * proj(K)).
    When content_feature is provided, a lightweight attention refines K using token states.
    """

    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.trust_gate = KnowledgeTrustGate(emb_dim)

    def forward(
        self,
        semantic: torch.Tensor,
        knowledge: torch.Tensor,
        kg_trust: Optional[torch.Tensor] = None,
        content_feature: Optional[torch.Tensor] = None,
        content_masks: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_vec = self.k_proj(knowledge)
        tau = self.trust_gate(semantic, k_vec, kg_trust=kg_trust)

        if content_feature is not None and content_masks is not None:
            q = self.q_proj(semantic).unsqueeze(1)
            k = self.v_proj(content_feature)
            v = k
            scores = torch.bmm(q, k.transpose(1, 2)) / (semantic.size(-1) ** 0.5)
            mask = content_masks.unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e4)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            ctx = torch.bmm(attn, v).squeeze(1)
            k_vec = k_vec + ctx

        fused = semantic + tau * k_vec
        fused = F.normalize(fused, p=2, dim=1)
        return fused, tau.squeeze(1)


class GatedBilinearFusionLayer(nn.Module):
    """
    门控双线性融合层（优化方案1）：
    替代简单加性残差 S' = S + tau * proj(K)
    采用门控双线性融合：S' = g * S + (1-g) * (S ⊙ K_proj)
    
    优势：
    - 知识可以直接调制语义的每个维度（Hadamard积）
    - 可学习门控 g 控制语义保留与知识融合的比例
    - 保留残差连接防止退化
    """

    def __init__(self, emb_dim: int, dropout: float = 0.1, use_bilinear: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_bilinear = use_bilinear
        
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        
        if use_bilinear:
            self.bilinear_weight = nn.Parameter(torch.randn(emb_dim, emb_dim))
            nn.init.xavier_normal_(self.bilinear_weight)
        
        self.gate_proj = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )
        
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.trust_gate = KnowledgeTrustGate(emb_dim)

    def forward(
        self,
        semantic: torch.Tensor,
        knowledge: torch.Tensor,
        kg_trust: Optional[torch.Tensor] = None,
        content_feature: Optional[torch.Tensor] = None,
        content_masks: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_vec = self.k_proj(knowledge)
        
        if self.use_bilinear:
            k_vec = torch.matmul(k_vec, self.bilinear_weight)
        
        tau = self.trust_gate(semantic, k_vec, kg_trust=kg_trust)
        
        cos_sim = F.cosine_similarity(semantic, k_vec, dim=1).unsqueeze(1)
        
        gate_input = torch.cat([semantic, k_vec, cos_sim], dim=-1)
        gate = self.gate_proj(gate_input)
        
        bilinear_component = semantic * k_vec
        
        if content_feature is not None and content_masks is not None:
            q = self.q_proj(semantic).unsqueeze(1)
            k = self.v_proj(content_feature)
            v = k
            scores = torch.bmm(q, k.transpose(1, 2)) / (semantic.size(-1) ** 0.5)
            mask = content_masks.unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e4)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            ctx = torch.bmm(attn, v).squeeze(1)
            bilinear_component = bilinear_component + ctx

        fused = gate * semantic + (1 - gate) * tau * bilinear_component
        
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused, tau.squeeze(1)


class EnhancedKnowledgeFusionLayer(nn.Module):
    """
    增强型知识融合层（整合所有三个优化方案）：
    1. 门控双线性融合（优化方案1）
    2. 关系感知三元组编码（优化方案2）
    3. 多因子信任评分（优化方案3）
    
    支持两种模式：
    - 单向量知识模式（向后兼容）
    - 三元组知识模式（新特性）
    """

    def __init__(
        self,
        emb_dim: int,
        dropout: float = 0.1,
        use_bilinear: bool = True,
        use_multi_factor_trust: bool = False,
        use_triple_encoder: bool = False,
        max_triples: int = 8,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_bilinear = use_bilinear
        self.use_multi_factor_trust = use_multi_factor_trust
        self.use_triple_encoder = use_triple_encoder
        
        if use_triple_encoder:
            self.triple_encoder = TripleAwareEncoder(emb_dim, max_triples=max_triples)
        
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        
        if use_bilinear:
            self.bilinear_weight = nn.Parameter(torch.randn(emb_dim, emb_dim))
            nn.init.xavier_normal_(self.bilinear_weight)
        
        self.gate_proj = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )
        
        if use_multi_factor_trust:
            self.trust_gate = MultiFactorTrustGate(emb_dim)
        else:
            self.trust_gate = KnowledgeTrustGate(emb_dim)
        
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        semantic: torch.Tensor,
        knowledge: torch.Tensor,
        kg_trust: Optional[torch.Tensor] = None,
        content_feature: Optional[torch.Tensor] = None,
        content_masks: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
        triple_head: Optional[torch.Tensor] = None,
        triple_relation: Optional[torch.Tensor] = None,
        triple_tail: Optional[torch.Tensor] = None,
        triple_mask: Optional[torch.Tensor] = None,
        entity_coverage: Optional[torch.Tensor] = None,
        relation_relevance: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_triple_encoder and triple_head is not None:
            k_vec, triple_confidence = self.triple_encoder(
                triple_head,
                triple_relation,
                triple_tail,
                semantic,
                triple_mask,
            )
        else:
            k_vec = self.k_proj(knowledge)
            triple_confidence = None
        
        if self.use_bilinear:
            k_vec = torch.matmul(k_vec, self.bilinear_weight)
        
        if self.use_multi_factor_trust:
            tau, confidence = self.trust_gate(
                semantic,
                k_vec,
                kg_trust=kg_trust,
                entity_coverage=entity_coverage,
                relation_relevance=relation_relevance,
                uncertainty=uncertainty,
            )
        else:
            tau = self.trust_gate(semantic, k_vec, kg_trust=kg_trust)
            confidence = None
        
        cos_sim = F.cosine_similarity(semantic, k_vec, dim=1).unsqueeze(1)
        
        gate_input = torch.cat([semantic, k_vec, cos_sim], dim=-1)
        gate = self.gate_proj(gate_input)
        
        bilinear_component = semantic * k_vec
        
        if content_feature is not None and content_masks is not None:
            q = self.q_proj(semantic).unsqueeze(1)
            k = self.v_proj(content_feature)
            v = k
            scores = torch.bmm(q, k.transpose(1, 2)) / (semantic.size(-1) ** 0.5)
            mask = content_masks.unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e4)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            ctx = torch.bmm(attn, v).squeeze(1)
            bilinear_component = bilinear_component + ctx

        fused = gate * semantic + (1 - gate) * tau * bilinear_component
        
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused, tau.squeeze(1)


class SoftDomainHead(nn.Module):
    """SLFEND-style soft domain distribution from semantic features."""

    def __init__(self, emb_dim: int, domain_num: int, hidden: int = 128):
        super().__init__()
        self.domain_num = domain_num
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, domain_num),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(semantic), dim=1)