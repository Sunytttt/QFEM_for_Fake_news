#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal Fake News Detection (Text + Images + Comments)
- Trains on ALL labeled samples by default (no train/val/test split unless you explicitly set --val_ratio/--test_ratio).
- Supports JSON/JSONL datasets where each record contains keys like:
  ver_news_text, ver_news_label, ver_news_img_path, ver_comments, ver_weights,
  sim_news_text, sim_news_label, sim_news_img_path, sim_comments, sim_weights

If your dataset is CSV, keep using your old script. This script is for the JSON-style dataset shown in your screenshots.
"""

import os
import re
import json
import math
import random
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchvision import transforms
from torchvision.models import resnet50

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from model_rerank import CAMERCEINModel


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Distributed Training Helpers
# -------------------------
def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process."""
    return rank == 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Dataset I/O helpers
# -------------------------
def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _parse_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in {"1", "true", "real", "yes", "y", "positive"}:
        return 1
    if text in {"0", "false", "fake", "no", "n", "negative"}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return None


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # Some datasets store "[]" or "['a','b']" as string
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s[0] in "[{":
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, list) else [obj]
            except json.JSONDecodeError:
                pass
        # fallback: split
        return [t for t in re.split(r"[,\s;|]+", s) if t]
    return [x]


def load_json_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    # .json
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # common wrappers
        for key in ("data", "records", "items", "examples"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        # fallback: single record
        return [obj]
    raise ValueError("Unsupported JSON structure (expected list or dict).")


# -------------------------
# Image loading
# -------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _normalize_path(p: str, image_root: Optional[str]) -> str:
    p = p.strip()
    if not p:
        return ""
    if image_root and (not os.path.isabs(p)):
        return os.path.join(image_root, p)
    return p


def _list_images_in_dir(dir_path: str) -> List[str]:
    if not dir_path or (not os.path.isdir(dir_path)):
        return []
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(IMG_EXTS):
            files.append(os.path.join(dir_path, name))
    files.sort()
    return files


def resolve_one_image_path(img_field: Any, image_root: Optional[str]) -> Optional[str]:
    """
    img_field can be:
    - [] / None
    - "path/to/img.jpg"
    - ["img1.jpg","img2.jpg"]
    - "folder_name" (directory containing images)
    """
    candidates = _ensure_list(img_field)

    # If empty, nothing
    if not candidates:
        return None

    # Iterate candidates, pick first valid file; if candidate is dir, pick first image in dir
    for cand in candidates:
        cand_str = _safe_str(cand)
        if not cand_str:
            continue
        if cand_str.startswith("http://") or cand_str.startswith("https://"):
            continue  # no downloading here
        path = _normalize_path(cand_str, image_root)
        if not os.path.exists(path):
            continue
        if os.path.isfile(path) and path.lower().endswith(IMG_EXTS):
            return path
        if os.path.isdir(path):
            imgs = _list_images_in_dir(path)
            if imgs:
                return imgs[0]
    return None


def load_image_tensor(
    img_field: Any,
    image_root: Optional[str],
    transform: transforms.Compose,
    blank_image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      image: [3,H,W]
      mask:  scalar tensor 0/1
    """
    path = resolve_one_image_path(img_field, image_root)
    if not path:
        return blank_image, torch.tensor(0.0)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return transform(im), torch.tensor(1.0)
    except (UnidentifiedImageError, OSError):
        return blank_image, torch.tensor(0.0)


# -------------------------
# Dataset
# -------------------------
class JsonPairDataset(Dataset):
    """
    Each sample:
      - ver_news_text: str
      - ver_news_label: 0/1
      - ver_news_img_path: list/str
      - ver_comments: list[str]
      - ver_weights: list[number]
      - sim_news_text: list[str] (top-K)
      - sim_news_label: list[0/1]
      - sim_news_img_path: list[list/str]
      - sim_comments: list[list[str]]
      - sim_weights: list[list[number]]
    """
    def __init__(
        self,
        data_path: str,
        image_root: Optional[str],
        image_size: int = 224,
        max_sim: int = 10,
        max_google: int = 10,
        max_claims: int = 10,
        max_comments: int = 12,
        seed: int = 42,
    ):
        super().__init__()
        self.records = load_json_records(data_path)
        self.image_root = image_root
        self.max_sim = max_sim
        self.max_google = max_google
        self.max_claims = max_claims
        self.max_comments = max_comments
        self.rng = random.Random(seed)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.blank_image = torch.zeros(3, image_size, image_size)

        # Keep only labeled ver samples
        filtered = []
        for rec in self.records:
            y = _parse_label(rec.get("ver_news_label", rec.get("label")))
            if y is None:
                continue
            rec["_ver_label"] = int(y)
            filtered.append(rec)
        if not filtered:
            raise ValueError("No labeled records found (ver_news_label/label).")
        self.records = filtered

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]

        ver_text = _safe_str(r.get("ver_news_text", r.get("text", r.get("title", ""))))
        ver_label = int(r["_ver_label"])

        ver_img_field = r.get("ver_news_img_path", r.get("ver_news_img", r.get("ver_img_path")))
        ver_image, ver_image_mask = load_image_tensor(
            ver_img_field, self.image_root, self.transform, self.blank_image
        )

        # sims
        sim_texts = _ensure_list(r.get("sim_news_text"))
        sim_labels_raw = _ensure_list(r.get("sim_news_label"))
        sim_imgs_raw = _ensure_list(r.get("sim_news_img_path"))

        # Pad/trim to max_sim
        K = self.max_sim
        sim_texts = [ _safe_str(t) for t in sim_texts ][:K]
        sim_labels = [ _parse_label(v) for v in sim_labels_raw ][:K]
        sim_imgs_raw = sim_imgs_raw[:K]

        # Fill missing length
        while len(sim_texts) < K:
            sim_texts.append("")
        while len(sim_labels) < K:
            sim_labels.append(None)
        while len(sim_imgs_raw) < K:
            sim_imgs_raw.append([])

        # Convert sim labels to tensor with mask
        sim_label_mask = torch.tensor([1.0 if v is not None else 0.0 for v in sim_labels], dtype=torch.float)
        sim_label_tensor = torch.tensor([0 if v is None else int(v) for v in sim_labels], dtype=torch.long)

        sim_images = []
        sim_image_masks = []
        for img_field in sim_imgs_raw:
            img, m = load_image_tensor(img_field, self.image_root, self.transform, self.blank_image)
            sim_images.append(img)
            sim_image_masks.append(m)
        sim_images_tensor = torch.stack(sim_images, dim=0)             # [K,3,H,W]
        sim_image_masks_tensor = torch.stack(sim_image_masks, dim=0)   # [K]

        # Comments (ver-level only by default; you can extend to sim comments similarly)
        ver_comments = _ensure_list(r.get("ver_comments"))
        ver_weights = _ensure_list(r.get("ver_weights"))
        # Align lengths
        ver_comments = [ _safe_str(c) for c in ver_comments ]
        ver_w = []
        for w in ver_weights:
            try:
                ver_w.append(float(w))
            except Exception:
                ver_w.append(0.0)
        if len(ver_w) < len(ver_comments):
            ver_w.extend([1.0] * (len(ver_comments) - len(ver_w)))
        if len(ver_comments) < len(ver_w):
            ver_w = ver_w[:len(ver_comments)]

        # keep top max_comments by weight (if available)
        if ver_comments:
            order = list(range(len(ver_comments)))
            order.sort(key=lambda i: ver_w[i], reverse=True)
            order = order[: self.max_comments]
            ver_comments = [ver_comments[i] for i in order]
            ver_w = [ver_w[i] for i in order]

        # --- Google search results (new for CAMER) ---
        K_g = self.max_google
        google_texts_raw = _ensure_list(r.get("google_search_text"))
        google_source_raw = _ensure_list(r.get("google_source_type"))
        google_tdiff_raw = _ensure_list(r.get("google_time_diff"))

        google_texts = [_safe_str(t) for t in google_texts_raw][:K_g]
        google_source = []
        for v in google_source_raw[:K_g]:
            try:
                src_type = int(v)
                # Clamp to valid range [0, 4] for 5 source types
                src_type = max(0, min(4, src_type))
                google_source.append(src_type)
            except (ValueError, TypeError):
                google_source.append(4)  # default: "other"
        google_tdiff = []
        for v in google_tdiff_raw[:K_g]:
            try:
                google_tdiff.append(float(v))
            except (ValueError, TypeError):
                google_tdiff.append(0.0)

        # Pad to K_g
        while len(google_texts) < K_g:
            google_texts.append("")
        while len(google_source) < K_g:
            google_source.append(4)
        while len(google_tdiff) < K_g:
            google_tdiff.append(0.0)

        google_mask = torch.tensor(
            [1.0 if t.strip() else 0.0 for t in google_texts], dtype=torch.float
        )
        google_source_tensor = torch.tensor(google_source, dtype=torch.long)
        google_tdiff_tensor = torch.tensor(google_tdiff, dtype=torch.float)

        # --- Atomic claims (new for CEIN) ---
        P = self.max_claims
        claims_raw = _ensure_list(r.get("claims"))
        claims = [_safe_str(c) for c in claims_raw][:P]
        while len(claims) < P:
            claims.append("")
        claim_mask = torch.tensor(
            [1.0 if c.strip() else 0.0 for c in claims], dtype=torch.float
        )

        return {
            "ver_text": ver_text,
            "ver_label": torch.tensor(ver_label, dtype=torch.float),
            "ver_image": ver_image,
            "ver_image_mask": ver_image_mask,
            "sim_texts": sim_texts,                    # list[str] length K
            "sim_images": sim_images_tensor,           # [K,3,H,W]
            "sim_image_masks": sim_image_masks_tensor, # [K]
            "sim_labels": sim_label_tensor,            # [K]
            "sim_label_mask": sim_label_mask,          # [K]
            "ver_comments": ver_comments,              # list[str]
            "ver_comment_weights": torch.tensor(ver_w, dtype=torch.float) if ver_w else torch.zeros(0, dtype=torch.float),
            # CAMER fields
            "google_texts": google_texts,              # list[str] length K_g
            "google_source_types": google_source_tensor,  # [K_g]
            "google_time_diffs": google_tdiff_tensor,     # [K_g]
            "google_mask": google_mask,                   # [K_g]
            # CEIN fields
            "claims": claims,                          # list[str] length P
            "claim_mask": claim_mask,                  # [P]
        }


# -------------------------
# Collate
# -------------------------
def collate_fn(batch: List[Dict[str, Any]], tokenizer, max_length: int) -> Dict[str, Any]:
    # text
    ver_texts = [b["ver_text"] for b in batch]
    ver_enc = tokenizer(
        ver_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # sims: flatten B*K
    B = len(batch)
    K = len(batch[0]["sim_texts"])
    sim_texts_flat = []
    for b in batch:
        sim_texts_flat.extend(b["sim_texts"])
    sim_enc = tokenizer(
        sim_texts_flat,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # images
    ver_images = torch.stack([b["ver_image"] for b in batch], dim=0)              # [B,3,H,W]
    ver_image_mask = torch.stack([b["ver_image_mask"] for b in batch], dim=0)     # [B]
    sim_images = torch.stack([b["sim_images"] for b in batch], dim=0)             # [B,K,3,H,W]
    sim_image_masks = torch.stack([b["sim_image_masks"] for b in batch], dim=0)   # [B,K]

    # sim labels
    sim_labels = torch.stack([b["sim_labels"] for b in batch], dim=0)             # [B,K]
    sim_label_mask = torch.stack([b["sim_label_mask"] for b in batch], dim=0)     # [B,K]

    # comments: pad to Cmax in batch
    labels = torch.stack([b["ver_label"] for b in batch], dim=0)                  # [B]
    comments_list = [b["ver_comments"] for b in batch]
    weights_list = [b["ver_comment_weights"] for b in batch]
    Cmax = max((len(c) for c in comments_list), default=0)

    if Cmax == 0:
        comment_enc = None
        comment_weights = torch.zeros(B, 0, dtype=torch.float)
        comment_mask = torch.zeros(B, 0, dtype=torch.float)
    else:
        # flatten and pad with empty strings
        comments_flat = []
        comment_mask = torch.zeros(B, Cmax, dtype=torch.float)
        comment_weights = torch.zeros(B, Cmax, dtype=torch.float)
        for i in range(B):
            c = comments_list[i]
            w = weights_list[i]
            for j in range(Cmax):
                if j < len(c):
                    comments_flat.append(c[j])
                    comment_mask[i, j] = 1.0
                    comment_weights[i, j] = w[j] if j < w.numel() else 1.0
                else:
                    comments_flat.append("")
                    comment_mask[i, j] = 0.0
                    comment_weights[i, j] = 0.0

        comment_enc = tokenizer(
            comments_flat,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # --- Google search evidence (CAMER) ---
    K_g = len(batch[0]["google_texts"])
    google_texts_flat = []
    for b in batch:
        google_texts_flat.extend(b["google_texts"])
    # Check if any google evidence exists
    has_google = any(t.strip() for t in google_texts_flat)
    if has_google:
        google_enc = tokenizer(
            google_texts_flat,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        google_enc = None
    google_source_types = torch.stack([b["google_source_types"] for b in batch], dim=0)  # [B, K_g]
    google_time_diffs = torch.stack([b["google_time_diffs"] for b in batch], dim=0)      # [B, K_g]
    google_mask_tensor = torch.stack([b["google_mask"] for b in batch], dim=0)            # [B, K_g]

    # --- Claims (CEIN) ---
    P = len(batch[0]["claims"])
    claims_flat = []
    for b in batch:
        claims_flat.extend(b["claims"])
    has_claims = any(c.strip() for c in claims_flat)
    if has_claims:
        claim_enc = tokenizer(
            claims_flat,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        claim_enc = None
    claim_mask_tensor = torch.stack([b["claim_mask"] for b in batch], dim=0)  # [B, P]

    return {
        "ver_enc": ver_enc,                          # dict of [B,L]
        "sim_enc": sim_enc,                          # dict of [B*K,L]
        "ver_images": ver_images,                    # [B,3,H,W]
        "ver_image_mask": ver_image_mask,            # [B]
        "sim_images": sim_images,                    # [B,K,3,H,W]
        "sim_image_masks": sim_image_masks,          # [B,K]
        "sim_labels": sim_labels,                    # [B,K]
        "sim_label_mask": sim_label_mask,            # [B,K]
        "comment_enc": comment_enc,                  # dict of [B*Cmax,L] or None
        "comment_weights": comment_weights,          # [B,Cmax]
        "comment_mask": comment_mask,                # [B,Cmax]
        "labels": labels,                            # [B]
        "K": K,
        # CAMER
        "google_enc": google_enc,                    # dict of [B*K_g,L] or None
        "google_source_types": google_source_types,  # [B, K_g]
        "google_time_diffs": google_time_diffs,      # [B, K_g]
        "google_mask": google_mask_tensor,           # [B, K_g]
        "K_google": K_g,
        # CEIN
        "claim_enc": claim_enc,                      # dict of [B*P,L] or None
        "claim_mask": claim_mask_tensor,             # [B, P]
        "P": P,
    }


# -------------------------
# Model (use CAMERCEINModel from model_rerank.py)
# -------------------------


# -------------------------
# Batch device transfer
# -------------------------
def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> None:
    """Move all tensors in batch dict to device (in-place)."""
    for enc_key in ("ver_enc", "sim_enc", "comment_enc", "google_enc", "claim_enc"):
        if batch.get(enc_key) is not None:
            batch[enc_key] = {k: v.to(device) for k, v in batch[enc_key].items()}
    for tensor_key in (
        "ver_images", "ver_image_mask",
        "sim_images", "sim_image_masks", "sim_labels", "sim_label_mask",
        "labels", "comment_weights", "comment_mask",
        "google_source_types", "google_time_diffs", "google_mask",
        "claim_mask",
    ):
        if tensor_key in batch and isinstance(batch[tensor_key], torch.Tensor):
            batch[tensor_key] = batch[tensor_key].to(device)


# -------------------------
# Train/Eval (updated for CAMER+CEIN)
# -------------------------
def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(int).tolist()
    y_pred = (probs >= 0.5).astype(int).tolist()
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"acc": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def format_metrics(m: Dict[str, float]) -> str:
    return f"acc={m['acc']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion,
    max_grad_norm: float,
    claim_loss_weight: float = 0.3,
    gradient_accumulation_steps: int = 1,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_n = 0

    all_logits = []
    all_labels = []

    claim_criterion = nn.BCELoss()

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        _move_batch_to_device(batch, device)

        # Mixed precision training
        with autocast(enabled=(scaler is not None)):
            outputs = model(batch)  # dict
            logits = outputs["logits"]  # [B]
            labels = batch["labels"]

            # Main detection loss
            loss_det = criterion(logits, labels)

        # Auxiliary claim verification loss (outside autocast for BCELoss safety)
        # Convert to fp32 to avoid numerical issues with BCELoss in mixed precision
        V_claim = outputs["V_claim"].float()  # [B], in (0,1), convert to fp32
        # Clamp to avoid log(0) or log(1) numerical issues
        V_claim = torch.clamp(V_claim, min=1e-7, max=1-1e-7)
        loss_claim = claim_criterion(V_claim, labels.float())

        # Combine losses (outside autocast)
        loss = loss_det + claim_loss_weight * loss_claim

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every N steps
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs * gradient_accumulation_steps  # Unscale for logging
        total_n += bs

        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

    avg_loss = total_loss / max(1, total_n)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics_from_logits(logits_cat, labels_cat)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    desc: str = "eval",
    claim_loss_weight: float = 0.3,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_labels = []

    claim_criterion = nn.BCELoss()

    for batch in tqdm(loader, desc=desc, leave=False):
        _move_batch_to_device(batch, device)

        outputs = model(batch)
        logits = outputs["logits"]
        labels = batch["labels"]

        loss_det = criterion(logits, labels)
        # Convert to fp32 to avoid numerical issues with BCELoss
        V_claim = outputs["V_claim"].float()
        # Clamp to avoid log(0) or log(1) numerical issues
        V_claim = torch.clamp(V_claim, min=1e-7, max=1-1e-7)
        loss_claim = claim_criterion(V_claim, labels.float())
        loss = loss_det + claim_loss_weight * loss_claim

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

    avg_loss = total_loss / max(1, total_n)
    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.zeros(0)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(0)
    metrics = compute_metrics_from_logits(logits_cat, labels_cat) if total_n else {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return avg_loss, metrics


# -------------------------
# CLI / main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="/data/sunyuantao/data/group_data_standardized/group3.csv.json")
    p.add_argument("--image_root", default=None, help="/data/sunyuantao/data/downloaded_images")
    p.add_argument("--bert_model_name", default="xlm-roberta-large")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU (will be multiplied by gradient_accumulation_steps)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    p.add_argument("--fp16", action="store_true", help="Use mixed precision training (fp16)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128, help="Max sequence length (reduced from 256 to save memory)")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", default="outputs_mm")
    p.add_argument("--save_best", action="store_true")

    # Use ALL data for training by default
    p.add_argument("--val_ratio", type=float, default=0.1, help="Optional; set >0 to create a validation set.")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Optional; set >0 to create a test set.")

    # Dataset controls
    p.add_argument("--max_sim", type=int, default=10)
    p.add_argument("--max_google", type=int, default=10)
    p.add_argument("--max_claims", type=int, default=10)
    p.add_argument("--max_comments", type=int, default=12)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--claim_loss_weight", type=float, default=0.3,
                    help="Weight gamma for auxiliary claim verification loss")
    p.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature for CAMER soft gating")
    p.add_argument("--evidence_dropout", type=float, default=0.15,
                    help="Probability of dropping evidence during training (robustness to sparse evidence)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()

    # Only log on main process
    if is_main_process(rank):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    set_seed(args.seed + rank)  # Different seed per rank for data shuffling

    # Set device to local GPU
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    if is_main_process(rank):
        logging.info(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logging.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)

    dataset = JsonPairDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        image_size=args.image_size,
        max_sim=args.max_sim,
        max_google=args.max_google,
        max_claims=args.max_claims,
        max_comments=args.max_comments,
        seed=args.seed,
    )

    # Split only if requested; otherwise train on ALL data.
    total = len(dataset)
    if is_main_process(rank):
        logging.info(f"Total dataset size: {total}")

        # Validate dataset - check for invalid values
        logging.info("Validating dataset (checking 100 samples)...")

        label_issues = 0
        source_type_stats = {'min': float('inf'), 'max': float('-inf')}

        # Check more samples for comprehensive validation
        check_samples = min(100, len(dataset))
        for i in range(check_samples):
            sample = dataset[i]
            label = sample['ver_label'].item()
            if label not in [0.0, 1.0]:
                label_issues += 1
                if label_issues <= 5:  # Only log first 5
                    logging.warning(f"Sample {i}: Invalid label {label} (should be 0 or 1)")

            # Check google source types
            google_sources = sample['google_source_types']
            if google_sources.numel() > 0:
                max_source = google_sources.max().item()
                min_source = google_sources.min().item()
                source_type_stats['min'] = min(source_type_stats['min'], min_source)
                source_type_stats['max'] = max(source_type_stats['max'], max_source)

        logging.info(f"Validation complete: {check_samples} samples checked")
        logging.info(f"Label issues: {label_issues}")
        logging.info(f"Source type range: [{source_type_stats['min']}, {source_type_stats['max']}] (valid: [0, 4])")

        if source_type_stats['max'] > 4 or source_type_stats['min'] < 0:
            logging.error(f"ERROR: Source types out of valid range [0, 4]!")
        else:
            logging.info("All source types are within valid range ✓")
    val_n = int(total * args.val_ratio)
    test_n = int(total * args.test_ratio)
    train_n = total - val_n - test_n
    if train_n <= 0:
        raise ValueError("Invalid split: train size <= 0. Reduce val_ratio/test_ratio.")

    generator = torch.Generator().manual_seed(args.seed)
    if val_n > 0 or test_n > 0:
        train_set, val_set, test_set = random_split(dataset, [train_n, val_n, test_n], generator=generator)
    else:
        train_set, val_set, test_set = dataset, None, None

    # Calculate class weights to handle class imbalance
    # Extract labels from train_set
    train_labels = []
    for i in range(len(train_set)):
        sample = train_set[i]
        label = int(sample['ver_label'].item())
        train_labels.append(label)

    # Calculate pos_weight for BCEWithLogitsLoss
    n_total = len(train_labels)
    n_pos = sum(1 for l in train_labels if l == 1)
    n_neg = sum(1 for l in train_labels if l == 0)

    if is_main_process(rank):
        logging.info(f"Class distribution in training set:")
        logging.info(f"  Label 0 (negative): {n_neg} ({n_neg/n_total*100:.2f}%)")
        logging.info(f"  Label 1 (positive): {n_pos} ({n_pos/n_total*100:.2f}%)")
        logging.info(f"  Imbalance ratio: {max(n_pos, n_neg)/min(n_pos, n_neg):.2f}:1")

    # pos_weight = weight of positive class
    # For BCEWithLogitsLoss:
    #   - pos_weight > 1: increases loss for False Negatives (正类被错分为负)
    #   - We have many FNs, so we need pos_weight > 1
    # Formula: pos_weight = n_positive / n_negative (to penalize misclassifying positive class)
    pos_weight = n_pos / n_neg if n_neg > 0 else 1.0
    if is_main_process(rank):
        logging.info(f"  pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")
        logging.info(f"  (Increased weight on positive class to reduce False Negatives)")

    def _collate(b):
        return collate_fn(b, tokenizer=tokenizer, max_length=args.max_length)

    # Create DistributedSampler for distributed training
    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        drop_last=False,
    )

    val_loader = None
    test_loader = None
    if val_set is not None and len(val_set) > 0:
        val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
        )
    if test_set is not None and len(test_set) > 0:
        test_sampler = DistributedSampler(test_set, shuffle=False) if world_size > 1 else None
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
        )

    model = CAMERCEINModel(
        bert_model_name=args.bert_model_name,
        temperature=args.temperature,
        claim_loss_weight=args.claim_loss_weight,
        evidence_dropout=args.evidence_dropout,
    ).to(device)

    # Wrap model with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Adjust total steps for gradient accumulation
    total_steps = max(1, len(train_loader) * args.epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Create GradScaler for mixed precision training
    scaler = GradScaler() if args.fp16 else None
    if is_main_process(rank) and args.fp16:
        logging.info("Mixed precision training (fp16) enabled")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best_multimodal.pt")

    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion, args.max_grad_norm,
            claim_loss_weight=args.claim_loss_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scaler=scaler,
        )

        if val_loader is None:
            if is_main_process(rank):
                logging.info("epoch=%d train_loss=%.4f train_%s", epoch, train_loss, format_metrics(train_metrics))
            current_f1 = train_metrics["f1"]
        else:
            val_loss, val_metrics = evaluate(
                model, val_loader, device, criterion, desc="val",
                claim_loss_weight=args.claim_loss_weight,
            )
            if is_main_process(rank):
                logging.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f train_%s val_%s",
                    epoch,
                    train_loss,
                    val_loss,
                    format_metrics(train_metrics),
                    format_metrics(val_metrics),
                )
            current_f1 = val_metrics["f1"]

        # Only save on main process
        if is_main_process(rank) and args.save_best and current_f1 > best_f1:
            best_f1 = current_f1
            # Save the underlying model (unwrap DDP if needed)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), best_path)

    # Optional test
    if args.save_best and os.path.exists(best_path):
        # Load the best model (handle DDP wrapper)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(best_path, map_location=device))

    if test_loader is not None:
        test_loss, test_metrics = evaluate(
            model, test_loader, device, criterion, desc="test",
            claim_loss_weight=args.claim_loss_weight,
        )
        if is_main_process(rank):
            logging.info("test_loss=%.4f test_%s", test_loss, format_metrics(test_metrics))

    # Always save last (only on main process)
    if is_main_process(rank):
        last_path = os.path.join(args.output_dir, "last_multimodal.pt")
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), last_path)
        logging.info("saved last model to %s", last_path)

    # Clean up distributed training
    cleanup_distributed()


# --------
def predict_on_new_data(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    tokenizer,
    max_length: int = 256,
    output_dir: str = "predictions",
) -> Dict[str, float]:
    """
    使用训练好的模型对新数据进行推理，并计算指标。
    """
    model.eval()
    all_logits = []
    all_labels = []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
        drop_last=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    predictions = []

    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        _move_batch_to_device(batch, device)

        # Make predictions
        outputs = model(batch)
        logits = outputs["logits"]  # [B]

        # Store predictions and labels
        all_logits.append(logits.detach())
        all_labels.append(batch["labels"].detach())

        # Save predictions
        for i in range(len(batch["labels"])):
            pred = torch.sigmoid(logits[i]).item()
            predictions.append({
                "true_label": batch["labels"][i].item(),
                "predicted_prob": pred,
            })

    # Concatenate logits and labels
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    # Calculate metrics
    metrics = compute_metrics_from_logits(logits_cat, labels_cat)

    # Save predictions to output directory (can be used for further analysis)
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    return metrics

def load_model(model_path: str, device: torch.device, bert_model_name: str = "xlm-roberta-large") -> nn.Module:
    model = CAMERCEINModel(bert_model_name=bert_model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def cross_domain_test(args, model_path: str, test_paths: List[str]):
    """训练完成后，在其他域数据集上测试跨域泛化能力。"""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = load_model(model_path, device, bert_model_name=args.bert_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    criterion = nn.BCEWithLogitsLoss()

    for path in test_paths:
        name = os.path.basename(path)
        logging.info("=" * 60)
        logging.info("Cross-domain test on: %s", name)
        ds = JsonPairDataset(
            data_path=path, image_root=args.image_root,
            image_size=args.image_size, max_sim=args.max_sim,
            max_google=args.max_google, max_claims=args.max_claims,
            max_comments=args.max_comments, seed=args.seed,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
            drop_last=False,
        )
        loss, metrics = evaluate(
            model, loader, device, criterion, desc=f"cross-{name}",
            claim_loss_weight=args.claim_loss_weight,
        )
        logging.info("[%s] loss=%.4f %s", name, loss, format_metrics(metrics))

        # 保存预测结果
        out_dir = os.path.join(args.output_dir, f"predictions_{name}")
        pred_metrics = predict_on_new_data(
            model, ds, args.batch_size, device, tokenizer,
            args.max_length, output_dir=out_dir,
        )
        logging.info("[%s] saved predictions to %s", name, out_dir)


if __name__ == "__main__":
    args = parse_args()

    # Get rank before main() to check if we should run cross-domain testing
    rank = int(os.environ.get('RANK', 0))

    main()

    # 跨域测试: 在 group1 和 group2 上测试 (only on main process)
    if rank == 0:
        data_dir = os.path.dirname(args.data_path)
        train_name = os.path.basename(args.data_path)
        cross_paths = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".json") and f != train_name:
                cross_paths.append(os.path.join(data_dir, f))

        if cross_paths:
            best_path = os.path.join(args.output_dir, "best_multimodal.pt")
            last_path = os.path.join(args.output_dir, "last_multimodal.pt")
            model_path = best_path if os.path.exists(best_path) else last_path
            logging.info("Starting cross-domain evaluation with model: %s", model_path)
            cross_domain_test(args, model_path, cross_paths)
