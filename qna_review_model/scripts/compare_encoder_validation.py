"""Compare saved runs on exactly the same Validation IDs and targets."""
import argparse
import hashlib
import json
from pathlib import Path


def load_run(path):
    config = json.loads((path / "best/training_config.json").read_text())
    rows = [json.loads(line) for line in (path / "best-validation-predictions.jsonl").open() if line.strip()]
    if len({r['itemId'] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return config, {r['itemId']: r for r in rows}


def f1_sets(truth, pred, labels):
    tp = sum(len(t & p) for t, p in zip(truth, pred, strict=True))
    fp = sum(len(p - t) for t, p in zip(truth, pred, strict=True))
    fn = sum(len(t - p) for t, p in zip(truth, pred, strict=True))
    macro = []
    for label in labels:
        a = sum(label in t and label in p for t, p in zip(truth, pred))
        b = sum(label not in t and label in p for t, p in zip(truth, pred))
        c = sum(label in t and label not in p for t, p in zip(truth, pred))
        macro.append(2*a/(2*a+b+c) if 2*a+b+c else 0)
    return {"microF1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            "fixedLabelMacroF1": sum(macro)/len(macro) if macro else 0,
            "precision": tp/(tp+fp) if tp+fp else 0, "recall": tp/(tp+fn) if tp+fn else 0}


def compare(paths):
    runs = [load_run(path) for path in paths]
    reference = runs[0][1]
    def normalized_truth(rows):
        return [{k: sorted(set(r[k])) if k in ('reviewTruth','secondaryTruth') else r[k]
                 for k in ('itemId','sourceType','reviewTruth','primaryTruth','secondaryTruth','usableTruth') if k in r}
                for _,r in sorted(rows.items())]
    fingerprint = normalized_truth(reference)
    for _, rows in runs:
        other = normalized_truth(rows)
        if other != fingerprint:
            raise ValueError("Different Validation IDs or labels: cannot compare")
    # All models share the same macro denominator, including missed/unobserved labels.
    canonical = runs[0][0]['indexToCanonicalTheme']
    if any(c['indexToCanonicalTheme'] != canonical for c, _ in runs):
        raise ValueError("Different canonical heads")
    normal = [x for x in canonical if x not in ('OTHER', 'UNUSABLE')]
    report = []
    for path, (config, rows) in zip(paths, runs):
        review = [r for r in rows.values() if r['sourceType'] == 'review']
        qna = [r for r in rows.values() if r['sourceType'] == 'qna']
        rp = [{l for l, s in r['reviewCandidates'] if s >= config['reviewThreshold']} for r in review]
        sp = [{l for l, s in r['secondaryCandidates'] if s >= config['secondaryThreshold']} for r in qna]
        report.append({"run": str(path), "epoch": config['epoch'],
            "reviewThreshold": config['reviewThreshold'], "secondaryThreshold": config['secondaryThreshold'],
            "review": f1_sets([set(r['reviewTruth']) for r in review], rp, [f'{l}::{s}' for l in normal for s in ('positive','negative','neutral')]),
            "qnaSecondary": f1_sets([set(r['secondaryTruth']) for r in qna], sp, normal),
            "qnaPrimaryAccuracy": sum(r['primaryTruth']==r['primaryPrediction'] for r in qna)/len(qna) if qna else 0})
    return {"fingerprintVersion": "v2-unordered-multilabel-targets",
            "validationLabelFingerprint": hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest(),
            "records": len(reference), "runs": report,
            "warning": "Development Validation comparisons; not Test or an unbiased final performance claim. Macro denominator fixed across all runs."}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
