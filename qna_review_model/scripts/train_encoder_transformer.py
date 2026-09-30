#!/usr/bin/env python
"""Train PALDA's shared Transformer Encoder with category-masked task heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SENTIMENTS = ["positive", "negative", "neutral"]
SPECIAL_THEMES = {"OTHER", "UNUSABLE"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--review-pooling", choices=("shared", "label-attention"), default="shared")
    parser.add_argument("--qna-pooling", choices=("shared", "cls", "mean"), default="shared")
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--review-pos-weight", type=float, default=3.0)
    parser.add_argument("--review-loss-weight", type=float, default=1.0)
    parser.add_argument("--review-weight-mode", choices=("constant", "train-balanced"), default="constant")
    parser.add_argument("--secondary-pos-weight", type=float, default=3.0)
    parser.add_argument("--secondary-loss-weight", type=float, default=0.5)
    parser.add_argument("--usable-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label-space", choices=("category", "canonical"), default="canonical")
    parser.add_argument("--skip-test", action="store_true", help="Never load or evaluate Test during development")
    parser.add_argument("--metrics-only", action="store_true", help="Keep config/predictions but no model weights; requires --skip-test")
    parser.add_argument("--diagnostic-probes", type=Path, help="Frozen synthetic diagnostics, never used for selection or training")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class PaldaDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class PaldaCollator:
    def __init__(self, tokenizer: Any, label_map: dict[str, Any], max_length: int) -> None:
        self.tokenizer = tokenizer; self.max_length = max_length
        self.theme_to_index = label_map["targetToIndex"]
        self.category_masks = label_map["categoryMasks"]
        self.special_indices = set(label_map["specialIndices"])
        self.sentiment_to_index = {value: index for index, value in enumerate(SENTIMENTS)}

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            f"[TASK={row['sourceType'].upper()}] [CAT1={row['category']['level1Name']}] "
            f"[CAT2={row['category']['level2Name']}] {row['text']}"
            for row in rows
        ]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        batch_size, theme_count = len(rows), max(self.theme_to_index.values()) + 1
        category_mask = torch.zeros((batch_size, theme_count), dtype=torch.bool)
        review_targets = torch.zeros((batch_size, theme_count, 3), dtype=torch.float32)
        primary_targets = torch.full((batch_size,), -100, dtype=torch.long)
        secondary_targets = torch.zeros((batch_size, theme_count), dtype=torch.float32)
        review_task = torch.zeros(batch_size, dtype=torch.bool)
        usable_targets = torch.zeros(batch_size, dtype=torch.float32)
        sample_weights = torch.ones(batch_size, dtype=torch.float32)
        for row_index, row in enumerate(rows):
            allowed = self.category_masks[row["category"]["level2Id"]]
            category_mask[row_index, allowed] = True
            review_task[row_index] = row["sourceType"] == "review"
            usable_targets[row_index] = float(row["usable"])
            sample_weights[row_index] = float(row["annotation"]["sampleWeight"])
            if row["sourceType"] == "review":
                for label in row["labels"]["reviewLabels"]:
                    theme_index = self.theme_to_index[label["categoryThemeId"]]
                    review_targets[row_index, theme_index, self.sentiment_to_index[label["sentiment"]]] = 1
            else:
                primary = row["labels"]["qnaPrimaryTheme"]
                primary_targets[row_index] = self.theme_to_index[primary["categoryThemeId"]]
                for label in row["labels"]["qnaSecondaryThemes"]:
                    secondary_targets[row_index, self.theme_to_index[label["categoryThemeId"]]] = 1
        non_special_mask = category_mask.clone()
        if self.special_indices:
            non_special_mask[:, list(self.special_indices)] = False
        return {
            **encoded, "rows": rows, "category_mask": category_mask,
            "non_special_mask": non_special_mask, "review_targets": review_targets,
            "primary_targets": primary_targets, "secondary_targets": secondary_targets,
            "review_task": review_task, "usable_targets": usable_targets,
            "sample_weights": sample_weights,
        }


def pool_tokens(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0]
    if mode != "mean":
        raise ValueError(f"Unknown pooling: {mode}")
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def label_attention_pool(hidden: torch.Tensor, attention_mask: torch.Tensor,
                         queries: torch.Tensor) -> torch.Tensor:
    """Per-theme token representations; attention is not evidence attribution."""
    scores = torch.einsum("blh,th->btl", hidden, queries).float()
    mask = attention_mask[:, None, :].bool()
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1) * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("btl,blh->bth", weights.to(hidden.dtype), hidden)


class PaldaEncoder(nn.Module):
    def __init__(self, model_name: str, theme_count: int, pooling: str = "cls",
                 review_pooling: str = "shared", qna_pooling: str = "shared") -> None:
        super().__init__()
        if review_pooling not in {"shared", "label-attention"}:
            raise ValueError(f"Unknown Review pooling: {review_pooling}")
        if qna_pooling not in {"shared", "cls", "mean"}:
            raise ValueError(f"Unknown Q&A pooling: {qna_pooling}")
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.review_head = nn.Linear(hidden, theme_count * 3)
        self.primary_head = nn.Linear(hidden, theme_count)
        self.secondary_head = nn.Linear(hidden, theme_count)
        self.usable_head = nn.Linear(hidden, 1)
        self.theme_count = theme_count
        self.pooling = pooling
        self.review_pooling = review_pooling
        self.qna_pooling = qna_pooling
        if review_pooling == "label-attention":
            # Zero initialization leaves existing head RNG initialization unchanged.
            self.review_queries = nn.Parameter(torch.zeros(theme_count, hidden))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(pool_tokens(output.last_hidden_state, attention_mask, self.pooling))
        if self.review_pooling == "label-attention":
            contexts = self.dropout(label_attention_pool(
                output.last_hidden_state, attention_mask, self.review_queries))
            review = torch.einsum("bth,tsh->bts", contexts,
                                  self.review_head.weight.view(self.theme_count, 3, -1))
            review = review + self.review_head.bias.view(self.theme_count, 3)
        else:
            review = self.review_head(pooled).view(-1, self.theme_count, 3)
        qna_pooled = pooled if self.qna_pooling == "shared" else self.dropout(
            pool_tokens(output.last_hidden_state, attention_mask, self.qna_pooling))
        return {
            "review": review,
            "primary": self.primary_head(qna_pooled),
            "secondary": self.secondary_head(qna_pooled),
            "usable": self.usable_head(qna_pooled).squeeze(-1),
        }


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
               weights: torch.Tensor, pos_weight: float | torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=torch.as_tensor(pos_weight, device=logits.device)
    )
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    weighted_mask = mask.float() * weights
    return (loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)


def training_positive_weights(rows: list[dict[str, Any]], label_map: dict[str, Any]) -> torch.Tensor:
    """Bounded square-root negative/positive ratio using Train and category exposure only."""
    size=len(label_map["indexToCanonicalTheme"])
    exposed=torch.zeros((size,3)); positive=torch.zeros_like(exposed)
    for row in rows:
        if row["sourceType"] != "review":
            continue
        weight=float(row["annotation"]["sampleWeight"])
        allowed=label_map["categoryMasks"][row["category"]["level2Id"]]
        exposed[allowed] += weight
        targets={(label_map["targetToIndex"][l["categoryThemeId"]], SENTIMENTS.index(l["sentiment"]))
                 for l in row["labels"]["reviewLabels"]}
        for idx,sentiment in targets:
            if idx not in allowed:
                raise ValueError("Training target is masked out")
            positive[idx,sentiment] += weight
    ratio=(exposed-positive).clamp_min(0)/positive.clamp_min(1)
    return torch.where(positive>0, ratio.sqrt().clamp(1,10), torch.ones_like(ratio))


def batch_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], args: argparse.Namespace,
               review_positive_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    review = batch["review_task"]; qna = ~review; weights = batch["sample_weights"]
    review_mask = batch["non_special_mask"].unsqueeze(-1).expand(-1, -1, 3) & review[:, None, None]
    review_loss = masked_bce(outputs["review"], batch["review_targets"], review_mask, weights,
                             args.review_pos_weight if review_positive_weights is None else review_positive_weights)
    masked_primary = outputs["primary"].masked_fill(~batch["category_mask"], -1e4)
    if qna.any():
        primary_each = nn.functional.cross_entropy(masked_primary[qna], batch["primary_targets"][qna], reduction="none")
        primary_loss = (primary_each * weights[qna]).sum() / weights[qna].sum().clamp_min(1)
    else:
        primary_loss = outputs["primary"].sum() * 0
    secondary_mask = batch["non_special_mask"] & qna[:, None]
    secondary_loss = masked_bce(outputs["secondary"], batch["secondary_targets"], secondary_mask, weights, args.secondary_pos_weight)
    if qna.any():
        usable_each = nn.functional.binary_cross_entropy_with_logits(
            outputs["usable"][qna], batch["usable_targets"][qna], reduction="none"
        )
        usable_loss = (usable_each * weights[qna]).sum() / weights[qna].sum().clamp_min(1)
    else:
        usable_loss = outputs["usable"].sum() * 0
    total = args.review_loss_weight * review_loss + primary_loss + args.secondary_loss_weight * secondary_loss + args.usable_loss_weight * usable_loss
    return total, {"review": float(review_loss.detach()), "primary": float(primary_loss.detach()),
                   "secondary": float(secondary_loss.detach()), "usable": float(usable_loss.detach())}


def set_metrics(true_sets: list[set[str]], pred_sets: list[set[str]]) -> dict[str, float]:
    classes = sorted(set().union(*true_sets, *pred_sets))
    if not classes:
        return {"microPrecision": 0, "microRecall": 0, "microF1": 0, "macroF1": 0, "exactMatch": 1}
    mlb = MultiLabelBinarizer(classes=classes); y=mlb.fit_transform(true_sets); p=mlb.transform(pred_sets)
    tp=(y&p).sum(); fp=((1-y)&p).sum(); fn=(y&(1-p)).sum()
    precision=float(tp/max(1,tp+fp)); recall=float(tp/max(1,tp+fn))
    return {"microPrecision":round(precision,4),"microRecall":round(recall,4),
            "microF1":round(2*precision*recall/max(1e-12,precision+recall),4),
            "macroF1":round(float(f1_score(y,p,average="macro",zero_division=0)),4),
            "exactMatch":round(float(np.mean(np.all(y==p,axis=1))),4)}


def runtime_label_map(label_map: dict[str, Any], label_space: str) -> dict[str, Any]:
    category_theme_to_index = label_map["categoryThemeToIndex"]
    index_to_category_theme = label_map["indexToCategoryTheme"]
    metadata = label_map["categoryThemeMetadata"]
    if label_space == "category":
        index_to_canonical = [metadata[value]["canonicalThemeId"] for value in index_to_category_theme]
        return {
            "targetToIndex": category_theme_to_index,
            "indexToCanonicalTheme": index_to_canonical,
            "categoryMasks": label_map["categoryMasks"],
            "specialIndices": [index for index, value in enumerate(index_to_canonical) if value in SPECIAL_THEMES],
        }
    canonical = sorted({value["canonicalThemeId"] for value in metadata.values()})
    canonical_to_index = {value: index for index, value in enumerate(canonical)}
    target_to_index = {
        category_theme_id: canonical_to_index[value["canonicalThemeId"]]
        for category_theme_id, value in metadata.items()
    }
    category_masks = {}
    for category_id, original_indices in label_map["categoryMasks"].items():
        category_masks[category_id] = sorted({
            target_to_index[index_to_category_theme[index]] for index in original_indices
        })
    return {
        "targetToIndex": target_to_index,
        "indexToCanonicalTheme": canonical,
        "categoryMasks": category_masks,
        "specialIndices": sorted(canonical_to_index[value] for value in SPECIAL_THEMES if value in canonical_to_index),
    }


@torch.no_grad()
def collect_predictions(model: PaldaEncoder, loader: DataLoader, device: torch.device,
                        label_map: dict[str, Any]) -> list[dict[str, Any]]:
    model.eval(); results=[]
    index_to_canonical=label_map["indexToCanonicalTheme"]
    for batch in loader:
        rows=batch.pop("rows")
        for key,value in list(batch.items()):
            if torch.is_tensor(value):batch[key]=value.to(device)
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            outputs=model(batch["input_ids"],batch["attention_mask"])
        review_prob=torch.sigmoid(outputs["review"]).cpu().numpy();secondary_prob=torch.sigmoid(outputs["secondary"]).cpu().numpy()
        primary_logits=outputs["primary"].masked_fill(~batch["category_mask"],-1e4);primary_pred=primary_logits.argmax(-1).cpu().numpy()
        primary_prob=torch.softmax(primary_logits.float(), dim=-1).cpu().numpy()
        usable_prob=torch.sigmoid(outputs["usable"]).cpu().numpy();allowed=batch["non_special_mask"].cpu().numpy()
        for i,row in enumerate(rows):
            result={"itemId":row["itemId"],"sourceType":row["sourceType"],"usableProbability":float(usable_prob[i])}
            if row["sourceType"]=="review":
                candidates=[]
                for theme_index in np.flatnonzero(allowed[i]):
                    canonical=index_to_canonical[theme_index]
                    for sentiment_index,sentiment in enumerate(SENTIMENTS):
                        candidates.append((f"{canonical}::{sentiment}",float(review_prob[i,theme_index,sentiment_index])))
                result["reviewCandidates"]=candidates
                result["reviewTruth"]=[f"{x['themeId']}::{x['sentiment']}" for x in row["labels"]["reviewLabels"]]
            else:
                result["primaryPrediction"]=index_to_canonical[int(primary_pred[i])]
                ordered = np.sort(primary_prob[i])
                result["primaryProbability"] = float(ordered[-1])
                result["primaryMargin"] = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])
                result["primaryCandidates"] = [(index_to_canonical[j],float(primary_prob[i,j]))
                    for j in torch.nonzero(batch['category_mask'][i],as_tuple=False).flatten().cpu().tolist()]
                result["primaryTruth"]=row["labels"]["qnaPrimaryTheme"]["themeId"]
                candidates=[]
                for theme_index in np.flatnonzero(allowed[i]):
                    candidates.append((index_to_canonical[theme_index],float(secondary_prob[i,theme_index])))
                result["secondaryCandidates"]=candidates
                result["secondaryTruth"]=[x["themeId"] for x in row["labels"]["qnaSecondaryThemes"]]
                result["usableTruth"]=bool(row["usable"])
            results.append(result)
    return results


def choose_threshold(predictions: list[dict[str, Any]], field: str, truth_field: str) -> tuple[float, dict[str, float]]:
    best=(0.0,0.5,{})
    for threshold in np.arange(0.2,0.91,0.05):
        rows=[row for row in predictions if field in row]
        true=[set(row[truth_field]) for row in rows]
        pred=[{label for label,score in row[field] if score>=threshold} for row in rows]
        metrics=set_metrics(true,pred)
        if metrics["microF1"]>best[0]:best=(metrics["microF1"],float(threshold),metrics)
    return round(best[1],2),best[2]


def evaluate_predictions(predictions: list[dict[str, Any]], review_threshold: float,
                         secondary_threshold: float) -> dict[str, Any]:
    review=[row for row in predictions if row["sourceType"]=="review"]
    qna=[row for row in predictions if row["sourceType"]=="qna"]
    review_metrics=set_metrics([set(x["reviewTruth"]) for x in review],[{label for label,score in x["reviewCandidates"] if score>=review_threshold} for x in review])
    secondary_metrics=set_metrics([set(x["secondaryTruth"]) for x in qna],[{label for label,score in x["secondaryCandidates"] if score>=secondary_threshold} for x in qna])
    primary_true=[x["primaryTruth"] for x in qna];primary_pred=[x["primaryPrediction"] for x in qna]
    usable_true=[x["usableTruth"] for x in qna];usable_pred=[x["usableProbability"]>=0.5 for x in qna]
    primary_metrics = (
        {"accuracy":round(float(accuracy_score(primary_true,primary_pred)),4),"macroF1":round(float(f1_score(primary_true,primary_pred,average="macro",zero_division=0)),4)}
        if qna else {"accuracy":0.0,"macroF1":0.0}
    )
    usable_metrics = (
        {"accuracy":round(float(accuracy_score(usable_true,usable_pred)),4),"macroF1":round(float(f1_score(usable_true,usable_pred,average="macro",zero_division=0)),4),"classCounts":dict(Counter(map(str,usable_true)))}
        if qna else {"accuracy":0.0,"macroF1":0.0,"classCounts":{}}
    )
    return {"reviewThemeSentiment":review_metrics,
            "qnaPrimary":primary_metrics,
            "qnaSecondary":secondary_metrics,
            "usable":usable_metrics}


def save_checkpoint(model: PaldaEncoder, tokenizer: Any, output: Path, config: dict[str, Any], *, save_weights: bool = True) -> None:
    output.mkdir(parents=True,exist_ok=True)
    if save_weights:
        torch.save(model.state_dict(),output/"model_state.pt")
    model.backbone.config.save_pretrained(output/"backbone_config")
    tokenizer.save_pretrained(output/"tokenizer")
    (output/"training_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main() -> None:
    args=arguments()
    if args.metrics_only and not args.skip_test:
        raise ValueError("--metrics-only requires --skip-test")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output_dir}")
    set_seed(args.seed);started=time.time();device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA is not available in this execution environment")
    stored_label_map=json.loads((args.data_dir/"label_map.json").read_text(encoding="utf-8"));label_map=runtime_label_map(stored_label_map,args.label_space);theme_count=len(label_map["indexToCanonicalTheme"])
    train=read_jsonl(args.data_dir/"train.jsonl",args.max_train);validation=read_jsonl(args.data_dir/"validation.jsonl",args.max_eval)
    probes = read_jsonl(args.diagnostic_probes) if args.diagnostic_probes else []
    if probes:
        training_texts = {''.join(r['text'].split()) for r in train}
        if any(r.get('trainingEligible') is not False or r['sourceType'] != 'review'
               or ''.join(r['text'].split()) in training_texts for r in probes):
            raise ValueError("Diagnostics must be non-training Review probes distinct from Train")
    review_positive_weights = training_positive_weights(train,label_map).to(device) if args.review_weight_mode == "train-balanced" else None
    test=[] if args.skip_test else read_jsonl(args.data_dir/"test.jsonl",args.max_eval)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True);collator=PaldaCollator(tokenizer,label_map,args.max_length)
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(PaldaDataset(train),batch_size=args.batch_size,shuffle=True,collate_fn=collator,num_workers=0,generator=generator)
    validation_loader=DataLoader(PaldaDataset(validation),batch_size=args.batch_size*2,shuffle=False,collate_fn=collator,num_workers=0)
    probe_loader=DataLoader(PaldaDataset(probes),batch_size=args.batch_size*2,shuffle=False,collate_fn=collator,num_workers=0)
    test_loader=DataLoader(PaldaDataset(test),batch_size=args.batch_size*2,shuffle=False,collate_fn=collator,num_workers=0)
    model=PaldaEncoder(args.model,theme_count,pooling=args.pooling,review_pooling=args.review_pooling,
                       qna_pooling=args.qna_pooling).to(device)
    backbone_params=[];head_params=[]
    for name,param in model.named_parameters():
        (backbone_params if name.startswith("backbone.") else head_params).append(param)
    optimizer=torch.optim.AdamW([{"params":backbone_params,"lr":args.learning_rate},{"params":head_params,"lr":args.head_learning_rate}],weight_decay=0.01)
    update_steps=math.ceil(len(train_loader)/args.gradient_accumulation);total_steps=update_steps*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*args.warmup_ratio),total_steps)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda");best_score=-1.0;history=[]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    provenance={'sourceHashes':{name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
                    for name in ('train.jsonl','validation.jsonl','label_map.json','theme_taxonomy.json')},
                'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'pythonVersion':sys.version,'torchVersion':str(torch.__version__),
                'numpyVersion':np.__version__,'seed':args.seed,
                'deterministicAlgorithms':torch.are_deterministic_algorithms_enabled(),
                'warning':'Seeds are fixed; bitwise identity across hardware/software is not guaranteed. Test not hashed here.'}
    (args.output_dir/'run-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if review_positive_weights is not None:
        (args.output_dir / "review-positive-weights.json").write_text(json.dumps({"themes":label_map["indexToCanonicalTheme"],
            "sentiments":SENTIMENTS,"weights":review_positive_weights.cpu().tolist(),"source":"Train only; sqrt ratio capped at 10"},indent=2)+"\n")
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);running=Counter();epoch_start=time.time()
        for step,batch in enumerate(train_loader,1):
            batch.pop("rows")
            for key,value in list(batch.items()):
                if torch.is_tensor(value):batch[key]=value.to(device,non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                outputs=model(batch["input_ids"],batch["attention_mask"]);loss,parts=batch_loss(outputs,batch,args,review_positive_weights);scaled_loss=loss/args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step%args.gradient_accumulation==0 or step==len(train_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
            running["loss"]+=float(loss.detach());
            for key,value in parts.items():running[key]+=value
            if step%100==0 or step==len(train_loader):
                print(json.dumps({"epoch":epoch,"step":step,"steps":len(train_loader),"loss":round(running["loss"]/step,4),"lr":scheduler.get_last_lr()[0]},ensure_ascii=False),flush=True)
        validation_predictions=collect_predictions(model,validation_loader,device,label_map)
        review_threshold,review_val=choose_threshold(validation_predictions,"reviewCandidates","reviewTruth")
        secondary_threshold,secondary_val=choose_threshold(validation_predictions,"secondaryCandidates","secondaryTruth")
        validation_metrics=evaluate_predictions(validation_predictions,review_threshold,secondary_threshold)
        score=validation_metrics["reviewThemeSentiment"]["microF1"]+validation_metrics["qnaPrimary"]["accuracy"]+validation_metrics["qnaSecondary"]["microF1"]
        epoch_row={"epoch":epoch,"seconds":round(time.time()-epoch_start,2),"trainLoss":round(running["loss"]/max(1,len(train_loader)),4),"trainLossParts":{k:round(running[k]/max(1,len(train_loader)),6) for k in ('review','primary','secondary','usable')},"thresholds":{"review":review_threshold,"secondary":secondary_threshold},"validation":validation_metrics,"selectionScore":round(score,4)};history.append(epoch_row)
        print(json.dumps(epoch_row,ensure_ascii=False),flush=True)
        if score>best_score:
            best_score=score;save_checkpoint(model,tokenizer,args.output_dir/"best",{"model":args.model,"labelSpace":args.label_space,"themeCount":theme_count,"indexToCanonicalTheme":label_map["indexToCanonicalTheme"],"epoch":epoch,"reviewThreshold":review_threshold,"secondaryThreshold":secondary_threshold,"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}, save_weights=not args.metrics_only)
            with (args.output_dir/"best-validation-predictions.jsonl").open("w",encoding="utf-8") as handle:
                for row in validation_predictions:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
            if probes:
                diagnostic_predictions = collect_predictions(model,probe_loader,device,label_map)
                with (args.output_dir/"best-diagnostic-predictions.jsonl").open("w",encoding="utf-8") as handle:
                    for probe,prediction in zip(probes,diagnostic_predictions,strict=True):
                        prediction['expectedLabels']=probe['expectedLabels']
                        prediction['probeKind']=probe['probeKind']
                        handle.write(json.dumps(prediction,ensure_ascii=False)+'\n')
    config=json.loads((args.output_dir/"best"/"training_config.json").read_text())
    test_metrics = None
    if not args.skip_test:
        model.load_state_dict(torch.load(args.output_dir/"best"/"model_state.pt",map_location=device,weights_only=True))
        test_predictions=collect_predictions(model,test_loader,device,label_map);test_metrics=evaluate_predictions(test_predictions,config["reviewThreshold"],config["secondaryThreshold"])
        with (args.output_dir/"test-predictions.jsonl").open("w",encoding="utf-8") as handle:
            for row in test_predictions:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    report={"schemaVersion":"palda-transformer-encoder-experiment-v1","model":args.model,"labelSpace":args.label_space,"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,"records":{"train":len(train),"validation":len(validation),"test":len(test)},"themeCount":theme_count,"bestEpoch":config["epoch"],"thresholds":{"review":config["reviewThreshold"],"secondary":config["secondaryThreshold"]},"history":history,"test":test_metrics,"elapsedSeconds":round(time.time()-started,2),"warning":"Validation/Test remain provisional Silver until human review."}
    report["testSkipped"] = args.skip_test
    report["weightsSaved"] = not args.metrics_only
    report["warning"] = "Development run: Test not loaded or evaluated." if args.skip_test else "Interpret metrics according to the source dataset annotation provenance."
    report["bestValidation"] = history[config["epoch"] - 1]["validation"]
    if probes:
        diagnostics=read_jsonl(args.output_dir/'best-diagnostic-predictions.jsonl')
        passed=sum({l for l,s in r['reviewCandidates'] if s>=config['reviewThreshold']}==set(r['expectedLabels']) for r in diagnostics)
        report['diagnosticProbes']={'passed':passed,'count':len(diagnostics),
            'usedForSelection':False,'warning':'Assistant-authored synthetic probes, not natural-distribution accuracy or Gold.'}
    (args.output_dir/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
