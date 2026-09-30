"""Grouped out-of-fold Q&A confidence calibration; no Test or classifier retraining.

Checkpoint selection already used Validation. This is development evidence, not a
guarantee of correctness. Top-1 labels do not change. Temperature is fit by NLL.
Reference: https://proceedings.mlr.press/v70/guo17a.html
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from scripts.measure_encoder_coverage import measure


def scale(probabilities, temperature):
    p = np.asarray(probabilities, dtype=float)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('Temperature must be finite and positive')
    if not len(p) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError('Invalid probability vector')
    logits = np.log(np.maximum(p / p.sum(), 1e-300)) / temperature
    values = np.exp(logits - logits.max())
    return values / values.sum()


def quality(probabilities, truths):
    if not probabilities or len(probabilities) != len(truths):
        raise ValueError('Nonempty equal-length probabilities and targets required')
    confidence = np.array([p.max() for p in probabilities])
    correct = np.array([p.argmax() == t for p, t in zip(probabilities, truths, strict=True)])
    nll = np.mean([-np.log(max(float(p[t]), 1e-300)) for p, t in zip(probabilities, truths)])
    brier = np.mean([float(np.square(p).sum() - 2 * p[t] + 1) for p, t in zip(probabilities, truths)])
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    details = []
    for i in range(10):
        selected = bins == i
        if not selected.any():
            continue
        acc, conf = float(correct[selected].mean()), float(confidence[selected].mean())
        ece += float(selected.mean()) * abs(acc - conf)
        details.append({'bin': i, 'count': int(selected.sum()), 'accuracy': acc, 'confidence': conf})
    return {'accuracy': float(correct.mean()), 'negativeLogLikelihood': float(nll),
            'multiclassBrierSum': float(brier), 'ece10EqualWidth': ece, 'bins': details}


def fit_temperature(probabilities, truths):
    def loss(log_t):
        return np.mean([-np.log(max(float(scale(p, np.exp(log_t))[t]), 1e-300))
                        for p, t in zip(probabilities, truths, strict=True)])
    result = minimize_scalar(loss, bounds=(np.log(.1), np.log(10)), method='bounded')
    if not result.success:
        raise ValueError('Temperature optimization failed')
    return float(np.exp(result.x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'validation', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = [json.loads(s) for s in args.validation.open() if s.strip()]
    data = [r for r in data if r['sourceType'] == 'qna']
    rows = [json.loads(s) for s in args.predictions.open() if s.strip()]
    rows = [r for r in rows if r['sourceType'] == 'qna']
    lookup = {r['itemId']: r for r in data}
    if len(lookup) != len(data) or len({r['itemId'] for r in rows}) != len(rows) or set(lookup) != {r['itemId'] for r in rows}:
        raise ValueError('Q&A identity mismatch')
    probabilities, truths = [], []
    for row in rows:
        labels, values = zip(*row['primaryCandidates'])
        if len(set(labels)) != len(labels) or row['primaryTruth'] != lookup[row['itemId']]['labels']['qnaPrimaryTheme']['themeId']:
            raise ValueError('Invalid candidates or target mismatch')
        p = scale(values, 1.0)
        if labels[int(p.argmax())] != row['primaryPrediction']:
            raise ValueError('Top-1 label inconsistent with probabilities')
        probabilities.append(p)
        truths.append(labels.index(row['primaryTruth']))
    products = sorted({r['productGroupId'] for r in data},
                      key=lambda p: hashlib.sha256(('primary-calibration-v1:' + p).encode()).hexdigest())
    fold_by_product = {pid: i % 5 for i, pid in enumerate(products)}
    folds = np.array([fold_by_product[lookup[r['itemId']]['productGroupId']] for r in rows])
    calibrated = [None] * len(rows)
    fitted = []
    for fold in range(5):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if not len(train) or not len(held):
            raise ValueError('Empty calibration fold')
        temperature = fit_temperature([probabilities[i] for i in train], [truths[i] for i in train])
        for i in held:
            calibrated[i] = scale(probabilities[i], temperature)
        fitted.append({'fold': fold, 'temperature': temperature, 'fitRecords': len(train),
                       'heldoutRecords': len(held), 'heldoutProducts': [p for p in products if fold_by_product[p] == fold]})
    updated = []
    for row, p in zip(rows, calibrated, strict=True):
        updated.append(row | {'primaryProbability': float(p.max()),
                              'primaryMargin': float(np.sort(p)[-1] - np.sort(p)[-2]) if len(p) > 1 else float(p[0]),
                              'primaryCandidates': [[label, float(v)] for (label, _), v in zip(row['primaryCandidates'], p)]})
    report = {'warning': __doc__, 'records': len(rows), 'products': len(products), 'fitted': fitted,
              'before': quality(probabilities, truths), 'outOfFold': quality(calibrated, truths),
              'beforeCoverage': measure(rows), 'outOfFoldCoverage': measure(updated),
              'sourceHashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.predictions, args.validation)}}
    if report['before']['accuracy'] != report['outOfFold']['accuracy']:
        raise ValueError('Calibration changed top-1 labels')
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'oof-qna-predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in updated))
    print(json.dumps({k: v for k, v in report.items() if k not in ('beforeCoverage', 'outOfFoldCoverage')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
