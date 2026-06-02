"""
CAMER + CEIN Model for Cross-Source Fake News Detection

CAMER: Credibility-Aware Multi-source Evidence Reranker
CEIN:  Claim-Evidence Interaction Network

Two innovations that jointly serve cross-domain fake news detection:
1. CAMER reranks and filters evidence from similar-news recall and Google search recall,
   removing noisy/irrelevant results via adaptive threshold gating.
2. CEIN performs claim-level verification by decomposing news into atomic claims and
   grounding each claim against evidence, providing fine-grained feedback to CAMER.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet50


# ===========================================================================
#  CAMER: Credibility-Aware Multi-source Evidence Reranker
# ===========================================================================
class CAMER(nn.Module):
    """
    Multi-source evidence reranker that jointly scores evidence from
    similar-news memory bank and Google search results.

    Scoring dimensions:
        r_i  = cos(h_N, h_{e_i})                       (semantic relevance)
        c_i  = W_c * Emb_source(t_i)                   (source credibility)
        t_i  = exp(-lambda * dt_i)                      (temporal relevance)
        s_i  = MLP([h_N; h_e; h_N*h_e; |h_N-h_e|])     (stance, 3-class)
        z_i  = W_proj * (h_N * h_e)                     (interaction feature)
        Score_i = MLP_rank([r; c; t; s; z])             (composite score)

    Adaptive gating:
        threshold = MLP([mean(Scores); std(Scores); fill_ratio])
        gate_i    = sigmoid((Score_i - threshold) / T)

    Aggregation:
        alpha_i = gate_i * exp(Score_i) / sum(...)
        E_agg   = sum(alpha_i * h_{e_i})
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_source_types: int = 5,
        source_embed_dim: int = 16,
        stance_classes: int = 3,
        interaction_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Source credibility
        self.source_embedding = nn.Embedding(num_source_types, source_embed_dim)
        self.source_proj = nn.Linear(source_embed_dim, 1)

        # Temporal decay
        self.temporal_lambda_raw = nn.Parameter(torch.tensor(1.0))

        # Stance detector
        self.stance_detector = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, stance_classes),
        )

        # Interaction projection
        self.interaction_proj = nn.Linear(hidden_dim, interaction_dim)

        # --- Similar-news boost: learnable bonus for sim evidence ---
        # is_sim_news is a [B, K] binary indicator; sim news gets extra score
        self.sim_boost = nn.Parameter(torch.tensor(0.5))

        # --- Evidence quality gate: discount sparse/low-quality Google evidence ---
        # Input: [google_fill_ratio, avg_text_relevance, source_diversity]
        self.quality_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Composite ranking score (added +1 for is_sim indicator)
        rank_in = 1 + 1 + 1 + stance_classes + interaction_dim + 1  # +1 for sim indicator
        self.rank_mlp = nn.Sequential(
            nn.Linear(rank_in, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Adaptive threshold
        self.threshold_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # CEIN feedback mixing
        self.feedback_beta = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h_news: torch.Tensor,             # [B, d]
        h_evidence: torch.Tensor,          # [B, K, d]
        source_types: torch.Tensor,        # [B, K] long
        time_diffs: torch.Tensor,          # [B, K] float
        evidence_mask: torch.Tensor,       # [B, K] float 1=valid
        is_sim_news: Optional[torch.Tensor] = None,  # [B, K] float 1=sim, 0=google
        feedback_contrib: Optional[torch.Tensor] = None,  # [B, K]
    ):
        B, K, d = h_evidence.shape
        h_n = h_news.unsqueeze(1).expand(B, K, d)  # [B, K, d]

        # (1) semantic relevance
        relevance = F.cosine_similarity(h_n, h_evidence, dim=-1)  # [B, K]

        # (2) source credibility
        src_emb = self.source_embedding(source_types)             # [B, K, E_s]
        credibility = self.source_proj(src_emb).squeeze(-1)       # [B, K]

        # (3) temporal relevance
        lam = F.softplus(self.temporal_lambda_raw)
        temporal = torch.exp(-lam * time_diffs)                   # [B, K]

        # (4) stance detection
        stance_in = torch.cat([
            h_n, h_evidence, h_n * h_evidence, torch.abs(h_n - h_evidence)
        ], dim=-1)                                                # [B, K, 4d]
        stance_logits = self.stance_detector(stance_in)           # [B, K, 3]
        stance_probs = F.softmax(stance_logits, dim=-1)           # [B, K, 3]

        # (5) interaction
        interaction = self.interaction_proj(h_n * h_evidence)     # [B, K, d_int]

        # (5b) sim news indicator feature
        if is_sim_news is None:
            sim_indicator = torch.zeros(B, K, device=h_evidence.device)
        else:
            sim_indicator = is_sim_news.float()

        # (6) composite score (with sim indicator)
        score_in = torch.cat([
            relevance.unsqueeze(-1),
            credibility.unsqueeze(-1),
            temporal.unsqueeze(-1),
            stance_probs,
            interaction,
            sim_indicator.unsqueeze(-1),     # extra feature for sim vs google
        ], dim=-1)                                                # [B, K, rank_in]
        scores = self.rank_mlp(score_in).squeeze(-1)              # [B, K]

        # (6b) Add learnable sim boost
        sim_bonus = F.softplus(self.sim_boost) * sim_indicator    # [B, K]
        scores = scores + sim_bonus

        # (6c) Evidence quality gate for Google evidence
        # Compute quality metrics for Google evidence (non-sim)
        google_indicator = (1.0 - sim_indicator) * evidence_mask  # [B, K]
        google_cnt = google_indicator.sum(dim=1, keepdim=True).clamp(min=1)
        google_fill_ratio = google_cnt / max(K, 1)               # [B, 1]
        google_relevance = (relevance * google_indicator).sum(dim=1, keepdim=True) / google_cnt
        # Source diversity: unique source types among Google evidence
        # Approximate: std of source_types for google items
        google_src_mean = (source_types.float() * google_indicator).sum(1, keepdim=True) / google_cnt
        google_src_var = ((source_types.float() - google_src_mean) ** 2 * google_indicator).sum(1, keepdim=True) / google_cnt
        google_diversity = torch.sqrt(google_src_var + 1e-8)

        quality_input = torch.cat([google_fill_ratio, google_relevance, google_diversity], dim=-1)
        quality_weight = self.quality_gate(quality_input)         # [B, 1] in (0,1)
        # Scale down Google evidence scores when quality is low
        scores = scores * (sim_indicator + google_indicator * quality_weight)
        # Re-mask padding
        scores = scores + (1.0 - evidence_mask) * (-1e9)

        # Apply CEIN feedback if available
        if feedback_contrib is not None:
            beta = torch.sigmoid(self.feedback_beta)
            scores = scores + beta * feedback_contrib

        # Mask padding
        scores = scores * evidence_mask + (1.0 - evidence_mask) * (-1e9)

        # (7) adaptive threshold
        valid_cnt = evidence_mask.sum(dim=1, keepdim=True).clamp(min=1)
        s_mean = (scores * evidence_mask).sum(1, keepdim=True) / valid_cnt
        s_diff = (scores - s_mean) * evidence_mask
        s_std = torch.sqrt((s_diff ** 2).sum(1, keepdim=True) / valid_cnt + 1e-8)
        fill_ratio = valid_cnt / max(K, 1)
        threshold = self.threshold_net(
            torch.cat([s_mean, s_std, fill_ratio], dim=-1)
        )  # [B, 1]

        # (8) soft gating
        gate = torch.sigmoid((scores - threshold) / self.temperature) * evidence_mask

        # (9) attention-weighted aggregation (numerically stable)
        score_max = scores.max(dim=1, keepdim=True)[0]
        gated_exp = gate * torch.exp(scores - score_max) * evidence_mask
        attn = gated_exp / (gated_exp.sum(dim=1, keepdim=True) + 1e-8)
        evidence_agg = (attn.unsqueeze(-1) * h_evidence).sum(dim=1)  # [B, d]

        return evidence_agg, scores, attn, stance_logits


# ===========================================================================
#  CEIN: Claim-Evidence Interaction Network
# ===========================================================================
class CEIN(nn.Module):
    """
    Decomposes news into atomic claims and verifies each claim against
    reranked evidence via cross-attention.

    Cross-attention:
        attended = MultiheadAttn(Q=claims, K=evidence, V=evidence)
        attended = LayerNorm(attended + claims)

    Claim verification:
        v_j = sigma(W * [h_c; attended; h_c - attended; h_c * attended])

    Evidence contribution feedback:
        Contrib_i = (1/P) sum_j Attn_{j,i}

    Aggregation:
        V_claim = sum_j softmax(MLP(h_c_j)) * v_j
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        self.claim_verifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.claim_pool = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        h_claims: torch.Tensor,       # [B, P, d]
        h_evidence: torch.Tensor,      # [B, K, d]
        claim_mask: torch.Tensor,      # [B, P] float
        evidence_mask: torch.Tensor,   # [B, K] float
    ):
        B, P, d = h_claims.shape

        # Cross-attention (key_padding_mask: True = ignore)
        key_pad = (evidence_mask == 0)
        attended, cross_attn_w = self.cross_attention(
            query=h_claims,
            key=h_evidence,
            value=h_evidence,
            key_padding_mask=key_pad,
        )  # attended [B,P,d], cross_attn_w [B,P,K]
        attended = self.cross_norm(attended + h_claims)

        # Claim verification score
        verify_in = torch.cat([
            h_claims, attended,
            h_claims - attended,
            h_claims * attended,
        ], dim=-1)  # [B, P, 4d]
        claim_scores = torch.sigmoid(
            self.claim_verifier(verify_in).squeeze(-1)
        )  # [B, P]

        # Evidence contribution (avg attention each evidence receives)
        valid_claims = claim_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        weighted_attn = cross_attn_w * claim_mask.unsqueeze(-1)          # [B, P, K]
        evidence_contrib = weighted_attn.sum(dim=1) / valid_claims       # [B, K]

        # Claim-level aggregation via attention pooling
        pool_logits = self.claim_pool(h_claims).squeeze(-1)              # [B, P]
        pool_logits = pool_logits + (claim_mask - 1.0) * 1e9
        pool_weights = F.softmax(pool_logits, dim=-1) * claim_mask       # [B, P]
        V_claim = (pool_weights * claim_scores).sum(dim=1)               # [B]
        # Clamp to ensure numerical stability in BCE loss
        V_claim = torch.clamp(V_claim, min=1e-7, max=1-1e-7)

        return V_claim, evidence_contrib, claim_scores


# ===========================================================================
#  Full Model: CAMER + CEIN + Encoder
# ===========================================================================
class CAMERCEINModel(nn.Module):
    """
    End-to-end model integrating CAMER and CEIN for cross-source fake news detection.

    Pipeline:
        1. Encode target news (text + image) -> h_N
        2. Encode all evidence (sim_news + google_search) -> h_E
        3. Encode claims (if available) -> h_C
        4. CAMER: rerank + filter evidence -> E_agg
        5. CEIN: claim-level verification -> V_claim, evidence_contrib
        6. Feedback: CEIN contrib -> CAMER 2nd pass -> E_agg_v2
        7. Three-way fusion: evidence + claim + comment -> logit_final

    Loss:
        L = BCEWithLogits(logit_final, y) + gamma * BCE(V_claim, y)
    """

    def __init__(
        self,
        bert_model_name: str = "xlm-roberta-large",
        hidden_dim: int = 1024,
        image_embed_dim: int = 1024,
        num_source_types: int = 5,
        temperature: float = 1.0,
        claim_loss_weight: float = 0.3,
        evidence_dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.claim_loss_weight = claim_loss_weight
        self.evidence_dropout = evidence_dropout

        # --- Text encoder (shared, auto-detect model type) ---
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Get actual hidden size from the model config
        encoder_dim = self.bert.config.hidden_size
        # Project encoder output to hidden_dim if needed
        self.text_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()

        # --- Image encoder ---
        backbone = resnet50(weights="DEFAULT")
        self.image_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.image_proj = nn.Linear(2048, image_embed_dim)

        # --- Text-image fusion gate ---
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim + image_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # --- Similar-news label embedding ---
        self.sim_label_emb = nn.Embedding(2, 32)
        self.sim_label_proj = nn.Linear(hidden_dim + 32 + 1, hidden_dim)

        # --- CAMER ---
        self.camer = CAMER(
            hidden_dim=hidden_dim,
            num_source_types=num_source_types,
            temperature=temperature,
        )

        # --- CEIN ---
        self.cein = CEIN(hidden_dim=hidden_dim)

        # --- Evidence head ---
        self.evidence_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # --- Claim head ---
        self.claim_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # --- Comment head ---
        self.comment_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # --- Three-way learnable mixing ---
        self.mix_evidence = nn.Parameter(torch.tensor(0.5))
        self.mix_claim = nn.Parameter(torch.tensor(0.3))
        self.mix_comment = nn.Parameter(torch.tensor(0.2))

    # ---- encoder helpers ----
    def encode_text_cls(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        cls_vec = self.bert(**enc, return_dict=True).last_hidden_state[:, 0, :]
        return self.text_proj(cls_vec)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_backbone(images).flatten(1)
        return self.image_proj(feats)

    def fuse_text_image(
        self, text_vec: torch.Tensor, img_vec: torch.Tensor, img_mask: torch.Tensor
    ) -> torch.Tensor:
        if img_mask.dim() == 1:
            img_mask = img_mask.unsqueeze(1)
        gate = torch.sigmoid(self.fuse_gate(torch.cat([text_vec, img_vec], dim=-1)))
        fused = gate * text_vec + (1.0 - gate) * img_vec
        return img_mask * fused + (1.0 - img_mask) * text_vec

    # ---- forward ----
    def forward(self, batch: Dict[str, Any]):
        """
        Returns dict:
            logits:       [B]   main detection logits (for BCEWithLogitsLoss)
            V_claim:      [B]   claim verification score (for auxiliary BCE loss)
            claim_scores: [B,P] per-claim scores or None
            stance_logits:[B,K,3] stance predictions
        """
        B = batch["ver_images"].size(0)
        device = batch["ver_images"].device
        K_sim = batch["K"]

        # ===== 1. Target news encoding =====
        ver_text = self.encode_text_cls(batch["ver_enc"])             # [B, 768]
        ver_img = self.encode_image(batch["ver_images"])              # [B, 768]
        h_news = self.fuse_text_image(
            ver_text, ver_img, batch["ver_image_mask"]
        )                                                             # [B, 768]

        # ===== 2. Similar-news evidence encoding =====
        sim_text = self.encode_text_cls(batch["sim_enc"])             # [B*K, 768]
        sim_text = sim_text.view(B, K_sim, self.hidden_dim)

        sim_imgs_flat = batch["sim_images"].view(
            B * K_sim, *batch["sim_images"].shape[2:]
        )
        sim_img = self.encode_image(sim_imgs_flat).view(
            B, K_sim, self.hidden_dim
        )
        sim_vec = self.fuse_text_image(
            sim_text.reshape(B * K_sim, -1),
            sim_img.reshape(B * K_sim, -1),
            batch["sim_image_masks"].reshape(B * K_sim),
        ).view(B, K_sim, self.hidden_dim)                             # [B, K_sim, 768]

        # Inject sim label feature
        sl_emb = self.sim_label_emb(batch["sim_labels"])              # [B, K, 32]
        sl_mask = batch["sim_label_mask"].unsqueeze(-1)               # [B, K, 1]
        sim_vec = self.sim_label_proj(
            torch.cat([sim_vec, sl_emb, sl_mask], dim=-1)
        )                                                             # [B, K_sim, 768]

        # Sim evidence mask
        sim_text_valid = (
            batch["sim_enc"]["attention_mask"]
            .view(B, K_sim, -1).sum(-1) > 0
        ).float()
        sim_mask = ((batch["sim_image_masks"] > 0).float() + sim_text_valid > 0).float()

        # Source/time defaults for sim news
        sim_source = torch.zeros(B, K_sim, dtype=torch.long, device=device)
        sim_tdiff = torch.zeros(B, K_sim, dtype=torch.float, device=device)

        # ===== 3. Google search evidence (optional) =====
        has_google = batch.get("google_enc") is not None
        if has_google:
            K_g = batch["K_google"]
            g_text = self.encode_text_cls(batch["google_enc"])        # [B*K_g, 768]
            g_text = g_text.view(B, K_g, self.hidden_dim)
            g_source = batch["google_source_types"]                   # [B, K_g]
            g_tdiff = batch["google_time_diffs"]                      # [B, K_g]
            g_mask = batch["google_mask"]                             # [B, K_g]

            # Merge sim + google
            h_evidence = torch.cat([sim_vec, g_text], dim=1)
            source_types = torch.cat([sim_source, g_source], dim=1)
            time_diffs = torch.cat([sim_tdiff, g_tdiff], dim=1)
            evidence_mask = torch.cat([sim_mask, g_mask], dim=1)
            # Build is_sim indicator: 1 for sim, 0 for google
            is_sim_news = torch.cat([
                torch.ones(B, K_sim, device=device),
                torch.zeros(B, K_g, device=device),
            ], dim=1)
        else:
            h_evidence = sim_vec
            source_types = sim_source
            time_diffs = sim_tdiff
            evidence_mask = sim_mask
            is_sim_news = torch.ones(B, K_sim, device=device)

        # ===== 3b. Evidence dropout (training augmentation) =====
        # Randomly drop evidence during training to make model robust to sparse evidence
        if self.training and self.evidence_dropout > 0:
            drop_mask = torch.bernoulli(
                torch.full_like(evidence_mask, 1.0 - self.evidence_dropout)
            )
            evidence_mask = evidence_mask * drop_mask

        # ===== 4. CAMER pass 1 =====
        evidence_agg, rank_scores, attn_w, stance_logits = self.camer(
            h_news, h_evidence, source_types, time_diffs, evidence_mask,
            is_sim_news=is_sim_news,
        )

        # ===== 5. CEIN (if claims available) =====
        has_claims = batch.get("claim_enc") is not None
        if has_claims:
            P = batch["P"]
            h_claims = self.encode_text_cls(batch["claim_enc"])       # [B*P, 768]
            h_claims = h_claims.view(B, P, self.hidden_dim)
            claim_mask = batch["claim_mask"]                          # [B, P]

            V_claim, ev_contrib, claim_scores = self.cein(
                h_claims, h_evidence, claim_mask, evidence_mask
            )

            # ===== 6. Feedback: CAMER pass 2 =====
            evidence_agg, _, attn_w, _ = self.camer(
                h_news, h_evidence, source_types, time_diffs,
                evidence_mask, is_sim_news=is_sim_news,
                feedback_contrib=ev_contrib,
            )
        else:
            V_claim = torch.full((B,), 0.5, device=device)
            claim_scores = None

        # ===== 7. Evidence logit =====
        pair = torch.cat([
            h_news, evidence_agg,
            torch.abs(h_news - evidence_agg),
            h_news * evidence_agg,
        ], dim=-1)                                                    # [B, 4d]
        evidence_logit = self.evidence_mlp(pair).squeeze(-1)          # [B]

        # ===== 8. Claim logit =====
        claim_logit = self.claim_head(V_claim.unsqueeze(-1)).squeeze(-1)  # [B]

        # ===== 9. Comment logit =====
        comment_enc = batch.get("comment_enc")
        if comment_enc is None:
            comment_logit = torch.zeros(B, device=device)
        else:
            Cmax = batch["comment_weights"].size(1)
            c_vec = self.encode_text_cls(comment_enc)                 # [B*C, 768]
            c_vec = c_vec.view(B, Cmax, self.hidden_dim)
            w = torch.relu(batch["comment_weights"]) * batch["comment_mask"]
            denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            c_agg = (c_vec * w.unsqueeze(-1)).sum(dim=1) / denom     # [B, 768]
            c_pair = torch.cat([h_news, c_agg], dim=-1)              # [B, 2*768]
            comment_logit = self.comment_mlp(c_pair).squeeze(-1)      # [B]

        # ===== 10. Three-way fusion =====
        w_e = torch.sigmoid(self.mix_evidence)
        w_c = torch.sigmoid(self.mix_claim)
        w_m = torch.sigmoid(self.mix_comment)
        w_sum = w_e + w_c + w_m + 1e-8
        logits = (w_e * evidence_logit + w_c * claim_logit + w_m * comment_logit) / w_sum

        return {
            "logits": logits,                   # [B]
            "V_claim": V_claim,                 # [B]
            "claim_scores": claim_scores,       # [B, P] or None
            "stance_logits": stance_logits,     # [B, K, 3]
            "rank_scores": rank_scores,         # [B, K]
            "attn_weights": attn_w,             # [B, K]
        }
