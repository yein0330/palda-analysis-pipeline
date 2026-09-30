"""Separate theme detection and sentiment attachment against existing Validation labels.

These are reference-label discrepancies, not newly adjudicated model errors.
Conditional sentiment accuracy omits missed themes and must not be used alone.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.compare_encoder_validation import compare, load_run


def by_theme(labels):
    result = {}
    for label in labels:
        theme, sentiment = label.rsplit('::', 1)
        result.setdefault(theme, set()).add(sentiment)
    return result


def decompose(rows, threshold):
    counts = Counter()
    confusions = Counter()
    for row in rows:
        if row['sourceType'] != 'review':
            continue
        truth = by_theme(row['reviewTruth'])
        pred = by_theme(label for label, score in row['reviewCandidates'] if score >= threshold)
        matched = truth.keys() & pred.keys()
        counts['records'] += 1
        counts['themeTP'] += len(matched)
        counts['themeFP'] += len(pred.keys() - truth.keys())
        counts['themeFN'] += len(truth.keys() - pred.keys())
        counts['exactSentimentSetsOnMatchedThemes'] += sum(truth[t] == pred[t] for t in matched)
        for theme in matched:
            counts['sentimentFPOnMatchedThemes'] += len(pred[theme] - truth[theme])
            counts['sentimentFNOnMatchedThemes'] += len(truth[theme] - pred[theme])
            if truth[theme] != pred[theme]:
                confusions[(theme, ','.join(sorted(truth[theme])), ','.join(sorted(pred[theme])))] += 1
        counts['labelFPFromAbsentThemes'] += sum(len(pred[t]) for t in pred.keys() - truth.keys())
        counts['labelFNFromMissedThemes'] += sum(len(truth[t]) for t in truth.keys() - pred.keys())
    tp, fp, fn = (counts[k] for k in ('themeTP', 'themeFP', 'themeFN'))
    result = dict(counts)
    result['themeOnlyMicroF1'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    result['sentimentSetAccuracyOnMatchedThemes'] = counts['exactSentimentSetsOnMatchedThemes'] / tp if tp else None
    result['sentimentMismatchCounts'] = [dict(theme=t, truth=a, predicted=b, count=n)
                                        for (t, a, b), n in confusions.most_common()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = compare(args.runs)
    results = []
    for run in args.runs:
        config, rows = load_run(run)
        results.append({'run': str(run), 'reviewThreshold': config['reviewThreshold'],
                        **decompose(rows.values(), config['reviewThreshold'])})
    report = {'warning': __doc__, 'validationLabelFingerprint': identity['validationLabelFingerprint'],
              'runs': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({**report, 'runs': [{k: v for k, v in r.items() if k != 'sentimentMismatchCounts'}
                                        for r in results]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
