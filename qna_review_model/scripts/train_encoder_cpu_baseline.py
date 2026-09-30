#!/usr/bin/env python
"""Fast CPU sanity baseline for PALDA Encoder v1 tasks."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-features", type=int, default=120_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_text(row: dict[str, Any]) -> str:
    category = row["category"]
    task = "REVIEW" if row["sourceType"] == "review" else "QNA"
    return f"[TASK={task}] [CAT1={category['level1Name']}] [CAT2={category['level2Name']}] {row['text']}"


def review_targets(row: dict[str, Any]) -> list[str]:
    return [f"{item['themeId']}::{item['sentiment']}" for item in row["labels"]["reviewLabels"]]


def secondary_targets(row: dict[str, Any]) -> list[str]:
    return [item["themeId"] for item in row["labels"]["qnaSecondaryThemes"]]


def binary_metrics(y_true: csr_matrix | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "microPrecision": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "microRecall": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "microF1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macroF1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "sampleF1": round(float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4),
        "exactMatch": round(float(np.mean(np.all(np.asarray(y_true.todense() if hasattr(y_true, "todense") else y_true) == y_pred, axis=1))), 4),
    }


def fit_ovr(x: csr_matrix, y: csr_matrix, seed: int) -> OneVsRestClassifier:
    estimator = SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-3,
        class_weight="balanced", random_state=seed, average=True,
    )
    model = OneVsRestClassifier(estimator, n_jobs=-1)
    model.fit(x, y)
    return model


def tune_threshold(y_true: csr_matrix, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.arange(0.2, 0.91, 0.05)
    scored = [
        (float(f1_score(y_true, probabilities >= threshold, average="micro", zero_division=0)), float(threshold))
        for threshold in candidates
    ]
    score, threshold = max(scored, key=lambda value: (value[0], value[1]))
    return round(threshold, 2), round(score, 4)


def cat1_metrics(rows: list[dict[str, Any]], task_mask: list[bool], y_true: Any, y_pred: Any, task: str) -> dict[str, float]:
    task_rows = [row for row, keep in zip(rows, task_mask, strict=True) if keep]
    result = {}
    for cat1 in sorted({row["category"]["level1Name"] for row in task_rows}):
        indices = [i for i, row in enumerate(task_rows) if row["category"]["level1Name"] == cat1]
        if not indices:
            continue
        if task == "review":
            result[cat1] = round(float(f1_score(y_true[indices], y_pred[indices], average="micro", zero_division=0)), 4)
        else:
            result[cat1] = round(float(accuracy_score(np.asarray(y_true)[indices], np.asarray(y_pred)[indices])), 4)
    return result


def predictions(rows: list[dict[str, Any]], review_model: Any, review_mlb: MultiLabelBinarizer,
                qna_model: Any, secondary_model: Any, secondary_mlb: MultiLabelBinarizer,
                usable_model: Any, matrix: csr_matrix, review_threshold: float,
                secondary_threshold: float) -> list[dict[str, Any]]:
    review_mask = np.array([row["sourceType"] == "review" for row in rows])
    qna_mask = ~review_mask
    review_pred = (review_model.predict_proba(matrix[review_mask]) >= review_threshold) if review_mask.any() else np.empty((0, len(review_mlb.classes_)))
    qna_pred = qna_model.predict(matrix[qna_mask]) if qna_mask.any() else np.array([])
    secondary_pred = (secondary_model.predict_proba(matrix[qna_mask]) >= secondary_threshold) if qna_mask.any() else np.empty((0, len(secondary_mlb.classes_)))
    usable_pred = usable_model.predict(matrix[qna_mask]) if qna_mask.any() else np.array([])
    result=[];ri=qi=0
    for row in rows:
        if row["sourceType"] == "review":
            result.append({"itemId":row["itemId"],"sourceType":"review","predictedReviewLabels":list(review_mlb.inverse_transform(review_pred[ri:ri+1]))[0]});ri+=1
        else:
            result.append({"itemId":row["itemId"],"sourceType":"qna","predictedUsable":bool(usable_pred[qi]),"predictedPrimaryTheme":str(qna_pred[qi]),"predictedSecondaryThemes":list(secondary_mlb.inverse_transform(secondary_pred[qi:qi+1]))[0]});qi+=1
    return result


def main() -> None:
    args = arguments(); started=time.time()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    train=load(args.data_dir/"train.jsonl"); validation=load(args.data_dir/"validation.jsonl"); test=load(args.data_dir/"test.jsonl")
    vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=args.min_df,max_features=args.max_features,sublinear_tf=True,dtype=np.float32)
    x_train=vectorizer.fit_transform(model_text(row) for row in train)
    x_validation=vectorizer.transform(model_text(row) for row in validation)
    x_test=vectorizer.transform(model_text(row) for row in test)
    review_train=np.array([r["sourceType"]=="review" for r in train]);qna_train=~review_train
    all_rows=[*train,*validation,*test]
    review_classes=sorted({label for row in all_rows if row["sourceType"]=="review" for label in review_targets(row)})
    review_mlb=MultiLabelBinarizer(classes=review_classes,sparse_output=True);review_mlb.fit([[]])
    y_review_train=review_mlb.transform([review_targets(row) for row in train if row["sourceType"]=="review"])
    review_model=fit_ovr(x_train[review_train],y_review_train,args.seed)
    primary_train=np.array([row["labels"]["qnaPrimaryTheme"]["themeId"] for row in train if row["sourceType"]=="qna"])
    qna_model=SGDClassifier(loss="log_loss",alpha=1e-5,max_iter=1000,tol=1e-3,class_weight="balanced",random_state=args.seed,average=True)
    qna_model.fit(x_train[qna_train],primary_train)
    secondary_classes=sorted({label for row in all_rows if row["sourceType"]=="qna" for label in secondary_targets(row)})
    secondary_mlb=MultiLabelBinarizer(classes=secondary_classes,sparse_output=True);secondary_mlb.fit([[]])
    y_secondary_train=secondary_mlb.transform([secondary_targets(row) for row in train if row["sourceType"]=="qna"])
    secondary_model=fit_ovr(x_train[qna_train],y_secondary_train,args.seed)
    usable_train=np.array([row["usable"] for row in train if row["sourceType"]=="qna"])
    usable_model=(
        SGDClassifier(loss="log_loss",alpha=1e-5,max_iter=1000,tol=1e-3,class_weight="balanced",random_state=args.seed,average=True)
        if len(np.unique(usable_train)) > 1 else DummyClassifier(strategy="most_frequent")
    )
    usable_model.fit(x_train[qna_train],usable_train)
    validation_review_mask=np.array([r["sourceType"]=="review" for r in validation]);validation_qna_mask=~validation_review_mask
    validation_review_y=review_mlb.transform([review_targets(row) for row in validation if row["sourceType"]=="review"])
    validation_secondary_y=secondary_mlb.transform([secondary_targets(row) for row in validation if row["sourceType"]=="qna"])
    review_threshold,review_tuned_f1=tune_threshold(validation_review_y,review_model.predict_proba(x_validation[validation_review_mask]))
    secondary_threshold,secondary_tuned_f1=tune_threshold(validation_secondary_y,secondary_model.predict_proba(x_validation[validation_qna_mask]))
    metrics={}
    for split,rows,matrix in (("validation",validation,x_validation),("test",test,x_test)):
        review_mask=np.array([r["sourceType"]=="review" for r in rows]);qna_mask=~review_mask
        y_review=review_mlb.transform([review_targets(row) for row in rows if row["sourceType"]=="review"]);p_review=review_model.predict_proba(matrix[review_mask])>=review_threshold
        y_primary=np.array([row["labels"]["qnaPrimaryTheme"]["themeId"] for row in rows if row["sourceType"]=="qna"]);p_primary=qna_model.predict(matrix[qna_mask])
        y_secondary=secondary_mlb.transform([secondary_targets(row) for row in rows if row["sourceType"]=="qna"]);p_secondary=secondary_model.predict_proba(matrix[qna_mask])>=secondary_threshold
        y_usable=np.array([row["usable"] for row in rows if row["sourceType"]=="qna"]);p_usable=usable_model.predict(matrix[qna_mask])
        metrics[split]={
            "reviewThemeSentiment":binary_metrics(y_review,p_review),
            "qnaPrimary":{"accuracy":round(float(accuracy_score(y_primary,p_primary)),4),"macroF1":round(float(f1_score(y_primary,p_primary,average="macro",zero_division=0)),4)},
            "qnaSecondary":binary_metrics(y_secondary,p_secondary),
            "usable":{"accuracy":round(float(accuracy_score(y_usable,p_usable)),4),"macroF1":round(float(f1_score(y_usable,p_usable,average="macro",zero_division=0)),4),"classCounts":dict(Counter(map(str,y_usable)))},
            "cat1ReviewMicroF1":cat1_metrics(rows,review_mask.tolist(),y_review,p_review,"review"),
            "cat1QnaPrimaryAccuracy":cat1_metrics(rows,qna_mask.tolist(),y_primary,p_primary,"qna"),
        }
        with (args.output_dir/f"{split}-predictions.jsonl").open("w",encoding="utf-8") as handle:
            for value in predictions(rows,review_model,review_mlb,qna_model,secondary_model,secondary_mlb,usable_model,matrix,review_threshold,secondary_threshold):handle.write(json.dumps(value,ensure_ascii=False)+"\n")
    artifact={"vectorizer":vectorizer,"reviewModel":review_model,"reviewClasses":review_classes,"reviewThreshold":review_threshold,"qnaPrimaryModel":qna_model,"secondaryModel":secondary_model,"secondaryClasses":secondary_classes,"secondaryThreshold":secondary_threshold,"usableModel":usable_model}
    joblib.dump(artifact,args.output_dir/"baseline.joblib",compress=3)
    report={"schemaVersion":"encoder-cpu-baseline-report-v1","model":"char-tfidf+sgd-logistic","purpose":"CPU pipeline sanity baseline; not the planned Transformer Encoder","dataDir":str(args.data_dir),"records":{"train":len(train),"validation":len(validation),"test":len(test)},"features":len(vectorizer.get_feature_names_out()),"reviewClasses":len(review_classes),"qnaPrimaryClasses":len(qna_model.classes_),"qnaSecondaryClasses":len(secondary_classes),"thresholds":{"review":review_threshold,"reviewValidationMicroF1":review_tuned_f1,"qnaSecondary":secondary_threshold,"qnaSecondaryValidationMicroF1":secondary_tuned_f1},"elapsedSeconds":round(time.time()-started,2),"metrics":metrics,"warning":"Validation/Test are provisional Silver labels; metrics are not final quality estimates."}
    (args.output_dir/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
