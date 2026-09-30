"""Score frozen assistant constraints; never reinterpret them after seeing predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def evaluate_constraint(decision, predicted, primary=None):
    required=set(decision['required']);allowed=set(decision['allowed'])
    if not required<=allowed:raise ValueError('Required label not allowed')
    missing=sorted(required-predicted);unsupported=sorted(predicted-allowed)
    primary_ok=primary in decision['primaryAllowed'] if 'primaryAllowed' in decision else True
    return {'passed':not missing and not unsupported and primary_ok,'missing':missing,
            'unsupported':unsupported,'requiredCovered':len(required & predicted),'requiredTotal':len(required),
            'predictedTotal':len(predicted),'primaryAccepted':primary_ok,'predicted':sorted(predicted)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--judgments',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude-unreviewed',action='store_true',help='Score annotationReady themes only; report other predictions separately')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    queue={r['itemId']:r for r in map(json.loads,a.queue.open())}
    judgments=json.loads(a.judgments.read_text());decisions=judgments['decisions']
    if len({d['itemId'] for d in decisions})!=len(decisions) or {d['itemId'] for d in decisions}!=set(queue):
        raise ValueError('Judgments must cover exactly the queue IDs once')
    for d in decisions:
        if any(not q or q not in queue[d['itemId']]['text'] for q in d['quotes']):raise ValueError('Quote not in source')
        allowed_themes={t['canonicalThemeId'] for t in queue[d['itemId']]['allowedThemes'] if t['annotationReady']}
        if d['status']=='score' and any(l.split('::')[0] not in allowed_themes for l in d['allowed']):
            raise ValueError('Judge label not in active taxonomy')
    results=[]
    for run in a.runs:
        cfg=json.loads((run/'best/training_config.json').read_text())
        rows={r['itemId']:r for r in map(json.loads,(run/'best-validation-predictions.jsonl').open())}
        outcomes=[]
        for d in decisions:
            if d['status']!='score':continue
            r=rows[d['itemId']];primary=None
            if r['sourceType']=='review':
                pred={l for l,s in r['reviewCandidates'] if s>=cfg['reviewThreshold']}
                reference=set(r['reviewTruth'])
            else:
                primary=r['primaryPrediction']
                pred={primary}|{l for l,s in r['secondaryCandidates'] if s>=cfg['secondaryThreshold']}
                reference={r['primaryTruth']}|set(r['secondaryTruth'])
            excluded_model=[];excluded_reference=[]
            if a.exclude_unreviewed:
                ready={t['canonicalThemeId'] for t in queue[d['itemId']]['allowedThemes'] if t['annotationReady']}
                excluded_model=sorted(l for l in pred if l.split('::')[0] not in ready)
                excluded_reference=sorted(l for l in reference if l.split('::')[0] not in ready)
                pred-=set(excluded_model);reference-=set(excluded_reference)
            outcomes.append({'itemId':r['itemId'],'sourceType':r['sourceType'],
                'outsideJudgeScope':{'model':excluded_model,'reference':excluded_reference},
                'model':evaluate_constraint(d,pred,primary),
                'reference':evaluate_constraint(d,reference,r.get('primaryTruth'))})
        summary={}
        for task in ('review','qna'):
            subset=[o for o in outcomes if o['sourceType']==task]
            summary[task]={'records':len(subset)}
            for kind in ('model','reference'):
                values=[o[kind] for o in subset]
                summary[task][kind]={'allConstraintsPassed':sum(v['passed'] for v in values),
                    'requiredCovered':sum(v['requiredCovered'] for v in values),
                    'requiredTotal':sum(v['requiredTotal'] for v in values),
                    'unsupportedPredictions':sum(len(v['unsupported']) for v in values),
                    'predictedTotal':sum(v['predictedTotal'] for v in values)}
                summary[task][kind]['outsideJudgeScope']=sum(len(o['outsideJudgeScope'][kind]) for o in subset)
        results.append({'run':str(run),'summary':summary,'outcomes':outcomes})
    result={'warning':judgments['warning'],'excludeUnreviewedThemes':a.exclude_unreviewed,
            'judgmentsSha256':hashlib.sha256(a.judgments.read_bytes()).hexdigest(),
            'queueSha256':hashlib.sha256(a.queue.read_bytes()).hexdigest(),
            'statuses':dict(Counter(d['status'] for d in decisions)),'runs':results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({**result,'runs':[{k:v for k,v in r.items() if k!='outcomes'} for r in results]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
