import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms
import torch.optim as optim
import json
import argparse
import csv
import random
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
try:
    from utils.utils import *  # noqa: F401,F403
except ImportError:
    pass
import logging 

########################################################################################################################
#用于判断 encodings 是否为全零编码
def is_zero_encodings(encoding):
    # 遍历所有编码
    for key, tensor in encoding.items():
        if torch.all(tensor.eq(0)):
            return 1
    return 0
########################################################################################################################
class FakeNewsDetectionModel(nn.Module):
    def __init__(self, args, device):
        super(FakeNewsDetectionModel, self).__init__()
        self.args = args
        self.device = device
        # BERT模型用于文本编码
        self.bert = BertModel.from_pretrained(args.bert_model_name)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)

        # VGG19用于图片特征提取
        self.vgg = vgg19(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])  # 移除最后一层
        
        # 自然语言推理部分
        self.nli_fc = nn.Linear(768, 3)  # 输出三个类别：蕴含、中立、矛盾
        # 立场检测部分
        self.stance_fc = nn.Linear(768, 3)  # 立场关系为3分类（支持、反对、中立）
        
        # 可学习的权重参数
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        
        # 最终分类
        self.final_fc = nn.Linear(2, 1)  # 综合NLI和立场检测的结果
        
    def forward(self, sim_news_text, val_news_text, sim_news_image, val_news_image, 
                sim_image_mask, val_image_mask, sim_comments, val_comments, sim_likes, 
                val_likes, sim_label):
        # BERT编码
        sim_news_text['input_ids'] = sim_news_text['input_ids'].squeeze(1)
        sim_news_text['token_type_ids'] = sim_news_text['token_type_ids'].squeeze(1)
        sim_news_text['attention_mask'] = sim_news_text['attention_mask'].squeeze(1)
        sim_news_text_features = self.bert(**sim_news_text, return_dict=True).last_hidden_state[:, 0, :]  # [CLS] token
        
        val_news_text['input_ids'] = val_news_text['input_ids'].squeeze(1)
        val_news_text['token_type_ids'] = val_news_text['token_type_ids'].squeeze(1)
        val_news_text['attention_mask'] = val_news_text['attention_mask'].squeeze(1)
        val_news_text_features = self.bert(**val_news_text, return_dict=True).last_hidden_state[:, 0, :]

        # VGG19提取图像特征
        sim_news_image_features = self.vgg(sim_news_image)
        val_news_image_features = self.vgg(val_news_image)
        
        # 利用mask屏蔽没有图片的情况
        sim_news_image_features = sim_news_image_features * sim_image_mask.unsqueeze(-1)
        val_news_image_features = val_news_image_features * val_image_mask.unsqueeze(-1)
        
        # 自然语言推理部分
        sim_news_features = torch.cat((sim_news_text_features, sim_news_image_features), dim=1)
        val_news_features = torch.cat((val_news_text_features, val_news_image_features), dim=1)
        # 使用多头注意力机制进行自然语言推理
        sim_news_features = sim_news_features.unsqueeze(0)  # 增加批次维度
        val_news_features = val_news_features.unsqueeze(0)  # 增加批次维度

        #sim_news_features = sim_news_text_features.unsqueeze(0)
        #val_news_features = val_news_text_features.unsqueeze(0)

        #attn_output, _ = self.multihead_attention(sim_news_features, val_news_features, val_news_features)

        # 平均池化得到最终的特征表示
        #attn_output = torch.mean(attn_output, dim=0)

        #nli_output = self.nli_fc(attn_output)  # 输出三个类别
        
        sim_news_features = sim_news_text_features
        val_news_features = val_news_text_features
        attention_scores = torch.matmul(sim_news_features, val_news_features.T)  # 计算注意力得分
        attention_weights = nn.Softmax(dim=-1)(attention_scores)  # 计算注意力权重
        attended_val_news_features = torch.matmul(attention_weights, val_news_features)  # 加权求和

        nli_output = self.nli_fc(attended_val_news_features)  # 输出三个类别

        # 根据NLI输出和相似新闻标签来确定真假
        nli_softmax = nn.Softmax(dim=1)(nli_output)
        entailment_prob = nli_softmax[:, 0]
        neutral_prob = nli_softmax[:, 1]
        contradiction_prob = nli_softmax[:, 2]

        nli_classification_output = torch.where(
            (sim_label == 1) & (entailment_prob > contradiction_prob) & (entailment_prob > neutral_prob),
            torch.tensor(1.0,device=self.device),
            torch.where(
                (sim_label == 0) & (contradiction_prob > entailment_prob) & (contradiction_prob > neutral_prob),
                torch.tensor(1.0,device=self.device),
                torch.where(
                    neutral_prob > entailment_prob,
                    sim_label.float(),
                    torch.tensor(0.0,device=self.device)
                )
            )
        )
        # 立场检测部分 - 计算评论加权特征
        comments_features = []

        for comment, like in zip(sim_comments, sim_likes):
            if not is_zero_encodings(comment):
                comment_feature = self.bert(**comment).last_hidden_state[:, 0, :]  # 提取[CLS] token的特征
                weighted_feature = comment_feature * like.unsqueeze(-1)  # 利用点赞数进行加权
                comments_features.append(weighted_feature)
        for comment, like in zip(val_comments, val_likes):
            if not is_zero_encodings(comment):
                comment_feature = self.bert(**comment).last_hidden_state[:, 0, :]  # 提取[CLS] token的特征
                weighted_feature = comment_feature * like.unsqueeze(-1)  # 利用点赞数进行加权
                comments_features.append(weighted_feature)

        if len(comments_features) == 0:
            batch_size = val_news_text_features.size(0)
            comments_weighted = torch.zeros(batch_size, 768, device=self.device)
        else:
            comments_weighted = torch.sum(torch.stack(comments_features), dim=0)  # 所有加权特征求和
      
        # 立场检测部分 - 拼接评论特征和新闻文本特征
        attention_scores = torch.matmul(comments_weighted, val_news_text_features.T)  # 计算注意力得分
        attention_weights = nn.Softmax(dim=-1)(attention_scores)  # 计算注意力权重
        attended_comments_features = torch.matmul(attention_weights, val_news_text_features)  # 加权求和
        stance_output_combined = self.stance_fc(attended_comments_features)

        # 假设第一个维度是“支持”的概率，进行二分类
        stance_classification_output = torch.sigmoid(stance_output_combined[:, 0])
        
        # 综合两个部分的输出
        nli_weighted = nli_classification_output * self.a
        stance_weighted = stance_classification_output * self.b

        combined_output = torch.cat((nli_weighted.unsqueeze(1), stance_weighted.unsqueeze(1)), dim=1)
        final_output = torch.sigmoid(self.final_fc(combined_output))
        
        return final_output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_str(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _parse_label(value):
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


def _build_text(row):
    text = _safe_str(row.get("text"))
    if text:
        return text
    parts = []
    for key in ("title", "content", "summary", "keywords", "hashtag", "query"):
        value = _safe_str(row.get(key))
        if value:
            parts.append(value)
    return " ".join(parts)


def _split_sim_tokens(raw_value):
    raw_value = _safe_str(raw_value)
    if not raw_value:
        return []
    if raw_value[0] in "[{":
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    tokens = re.split(r"[,\s;|]+", raw_value)
    return [token.strip() for token in tokens if token.strip()]


class NewsDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=256,
        image_root=None,
        image_size=224,
        sim_strategy="first",
        drop_unlabeled=True,
        seed=42,
        transform=None,
    ):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_root = image_root
        self.sim_strategy = sim_strategy
        self.rng = random.Random(seed)
        self.rows = self._load_rows(csv_path, drop_unlabeled)
        self.id_to_index = {
            _safe_str(row.get("news_id")): idx
            for idx, row in enumerate(self.rows)
            if _safe_str(row.get("news_id"))
        }
        self.sim_indices = [
            self._resolve_sim_indices(row.get("sim_index_top10")) for row in self.rows
        ]
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.blank_image = torch.zeros(3, image_size, image_size)

    def _load_rows(self, csv_path, drop_unlabeled):
        rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = _parse_label(row.get("label"))
                if label is None and drop_unlabeled:
                    continue
                row["_label"] = -1 if label is None else int(label)
                rows.append(row)
        if not rows:
            raise ValueError("No rows loaded from csv. Check path or label filtering.")
        return rows

    def _resolve_sim_indices(self, raw_value):
        tokens = _split_sim_tokens(raw_value)
        indices = []
        seen = set()
        for token in tokens:
            if token in self.id_to_index:
                idx = self.id_to_index[token]
            else:
                try:
                    num = int(float(token))
                except ValueError:
                    continue
                if 0 <= num < len(self.rows):
                    idx = num
                elif 0 <= num - 1 < len(self.rows):
                    idx = num - 1
                else:
                    continue
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices

    def _pick_sim_index(self, indices):
        if not indices:
            return None
        if self.sim_strategy == "random":
            return self.rng.choice(indices)
        return indices[0]

    def _resolve_image_path(self, row):
        """
        兼容三种情况：
        1) row['pic_url'] 是图片文件路径
        2) row['pic_url'] 是图片所在文件夹路径 / 文件夹名
        3) row['pic_url'] 为空，则用 row['news_id'] 对应的文件夹
        """
        # 先尝试 pic_url
        raw = _safe_str(row.get("pic_url"))
        candidates = []
        if raw:
            candidates = re.split(r"[,\s;|]+", raw)

        # 如果 pic_url 为空，用 news_id/文件夹名兜底
        if not candidates:
            folder = _safe_str(row.get("news_id"))  # 如果你的列名不是 news_id，请改成你的列名
            if folder:
                candidates = [folder]

        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            # 跳过 http(s)
            if cand.startswith("http://") or cand.startswith("https://"):
                continue

            path = cand
            if self.image_root and not os.path.isabs(path):
                path = os.path.join(self.image_root, path)

            if not os.path.exists(path):
                continue

            # 情况A：直接是文件
            if os.path.isfile(path) and path.lower().endswith(self.support_exts):
                return path

            # 情况B：是目录，则从目录里挑一张
            if os.path.isdir(path):
                files = self._list_images_in_dir(path)
                chosen = self._pick_one_image(files)
                if chosen:
                    return chosen

        return None
    def _load_image(self, row):
        path = self._resolve_image_path(row)
        if not path:
            return self.blank_image, torch.tensor(0.0)
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                return self.transform(image), torch.tensor(1.0)
        except (UnidentifiedImageError, OSError):
            return self.blank_image, torch.tensor(0.0)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        val_text = _build_text(row)
        val_text_enc = self.tokenizer(
            val_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        val_image, val_image_mask = self._load_image(row)

        sim_indices = self.sim_indices[idx]
        sim_idx = self._pick_sim_index(sim_indices)
        if sim_idx is None:
            sim_idx = idx
        sim_row = self.rows[sim_idx]
        sim_text = _build_text(sim_row)
        sim_text_enc = self.tokenizer(
            sim_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        sim_image, sim_image_mask = self._load_image(sim_row)
        
        sim_label_value = sim_row["_label"]
        if sim_label_value < 0:
            sim_label_value = row["_label"] if row["_label"] >= 0 else 0

        return {
            "sim_news_text": sim_text_enc,
            "val_news_text": val_text_enc,
            "sim_news_image": sim_image,
            "val_news_image": val_image,
            "sim_image_mask": sim_image_mask,
            "val_image_mask": val_image_mask,
            "sim_comments": [],
            "val_comments": [],
            "sim_likes": [],
            "val_likes": [],
            "sim_label": torch.tensor(sim_label_value, dtype=torch.long),
            "label": torch.tensor(row["_label"], dtype=torch.float),
        }


def _move_encoding_to_device(encoding, device):
    return {key: value.to(device) for key, value in encoding.items()}


def _forward_batch(model, batch, device):
    sim_news_text = _move_encoding_to_device(batch["sim_news_text"], device)
    val_news_text = _move_encoding_to_device(batch["val_news_text"], device)
    sim_news_image = batch["sim_news_image"].to(device)
    val_news_image = batch["val_news_image"].to(device)
    sim_image_mask = batch["sim_image_mask"].to(device)
    val_image_mask = batch["val_image_mask"].to(device)
    sim_label = batch["sim_label"].to(device)
    return model(
        sim_news_text,
        val_news_text,
        sim_news_image,
        val_news_image,
        sim_image_mask,
        val_image_mask,
        batch["sim_comments"],
        batch["val_comments"],
        batch["sim_likes"],
        batch["val_likes"],
        sim_label,
    )


def _compute_metrics(preds, labels):
    if not labels:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def train_one_epoch(model, data_loader, optimizer, scheduler, device, criterion, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(data_loader, desc="train", leave=False):
        optimizer.zero_grad()
        outputs = _forward_batch(model, batch, device).squeeze(1)
        labels = batch["label"].to(device)
        mask = labels >= 0
        if mask.sum().item() == 0:
            continue
        loss = criterion(outputs[mask], labels[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_count = int(mask.sum().item())
        total_loss += loss.item() * batch_count
        total_count += batch_count

        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        preds = (outputs_np[mask_np] >= 0.5).astype(int).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels_np[mask_np].astype(int).tolist())

    avg_loss = total_loss / max(1, total_count)
    metrics = _compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, data_loader, device, criterion, desc="eval"):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(data_loader, desc=desc, leave=False):
        outputs = _forward_batch(model, batch, device).squeeze(1)
        labels = batch["label"].to(device)
        mask = labels >= 0
        if mask.sum().item() == 0:
            continue
        loss = criterion(outputs[mask], labels[mask])
        batch_count = int(mask.sum().item())
        total_loss += loss.item() * batch_count
        total_count += batch_count

        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        preds = (outputs_np[mask_np] >= 0.5).astype(int).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels_np[mask_np].astype(int).tolist())

    avg_loss = total_loss / max(1, total_count)
    metrics = _compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


def _format_metrics(metrics):
    return (
        f"acc={metrics['acc']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="/data/sunyuantao/data/group_data/group1.csv")
    parser.add_argument("--image_root", default=None, help="/data/sunyuantao/data/downloaded_images/")
    parser.add_argument("--bert_model_name", default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--sim_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--drop_unlabeled", action="store_true")
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    dataset = NewsDataset(
        args.csv_path,
        tokenizer,
        max_length=args.max_length,
        image_root=args.image_root,
        image_size=args.image_size,
        sim_strategy=args.sim_strategy,
        drop_unlabeled=args.drop_unlabeled,
        seed=args.seed,
    )

    total_size = len(dataset)
    val_size = int(total_size * args.val_ratio)
    train_size = int(total_size * args.train_ratio)
    test_size = total_size - train_size - val_size
    if test_size <= 0:
        raise ValueError("Invalid split sizes. Adjust train_ratio/val_ratio.")

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = FakeNewsDetectionModel(args, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.BCELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            criterion,
            args.max_grad_norm,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, device, criterion, desc="val"
        )
        logging.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f train_%s val_%s",
            epoch,
            train_loss,
            val_loss,
            _format_metrics(train_metrics),
            _format_metrics(val_metrics),
        )

        if args.save_best and val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    if args.save_best and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_metrics = evaluate(
        model, test_loader, device, criterion, desc="test"
    )
    logging.info("test_loss=%.4f test_%s", test_loss, _format_metrics(test_metrics))


if __name__ == "__main__":
    main()
