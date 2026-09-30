"""Check frozen judge assertions against saved Validation predictions."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--checks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    checks=json.loads(a.checks.read_text())
    result={'warning':checks['warning'],'runs':[]}
    for run in a.runs:
        cfg=json.loads((run/'best/training_config.json').read_text())
        rows={r['itemId']:r for r in map(json.loads,(run/'best-validation-predictions.jsonl').open())}
        outcomes=[]
        for check in checks['checks']:
            r=rows[check['itemId']]
            if r['sourceType']=='review':
                predicted={l for l,s in r['reviewCandidates'] if s>=cfg['reviewThreshold']}
            else:
                predicted={'primary:'+r['primaryPrediction']}|{'secondary:'+l for l,s in r['secondaryCandidates'] if s>=cfg['secondaryThreshold']}
            missing=sorted(set(check['required'])-predicted)
            unsupported=sorted(set(check['forbidden'])&predicted)
            outcomes.append({'itemId':check['itemId'],'passed':not missing and not unsupported,
                             'missing':missing,'forbiddenPredictions':unsupported,'predicted':sorted(predicted)})
        result['runs'].append({'run':str(run),'passed':sum(x['passed'] for x in outcomes),'checks':len(outcomes),'outcomes':outcomes})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='outcomes'} for r in result['runs']],indent=2))
