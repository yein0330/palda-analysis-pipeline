#!/usr/bin/env python3
"""Explicitly bounded feedback pilot. Saved Validation sources only; no training/Test.

Use a NEW round name after editing policy. Never overwrite prior results or
quietly turn Codex/LLM opinions into human gold labels.
"""
import argparse
import asyncio
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from input_size_tradeoff import ROOT, RUNTIME, DEFAULT_OUT, read, write, file_hash, MemorySampler

OUT = ROOT / "artifacts/grounded-feedback-20260928"


def render(folder):
    cases = [read(p) for p in sorted((folder / "cases").glob("*.json"))]
    esc = html.escape
    doc = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>PALDA 직접 근거 실험</title>',
           '<style>body{max-width:1000px;margin:30px auto;font-family:sans-serif;line-height:1.6}section{border:1px solid #ccc;padding:18px;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{border-left:3px solid #aaa;padding-left:12px}</style>',
           '<h1>PALDA · 직접 근거 기반 피드백 실험</h1><p>내부 실험 / Validation / AI 평가이며 사람 정답 아님. 누락·실패를 숨기지 않습니다. 원문은 외부 공개하지 마세요.</p>']
    for c in cases:
        doc.append(f'<section><h2>{esc(c["category"])} · 리뷰 {c["reviewRows"]} / 질문 {c["qnaRows"]}</h2><p>{esc(c["status"])} · {c.get("seconds", 0):.1f}초</p>')
        if c["status"] == "completed":
            diag = c["diagnostics"]
            doc.append(f'<p>소구점 생성 상태: {esc(diag["sellingGeneration"]["status"])} · 원문 검토 실패 {c["classificationAudit"]["failed"]}건</p>')
            for key in ["positiveThemes", "negativeThemes"]:
                for g in c["result"]["reviewAnalysis"][key]:
                    doc.append(f'<h3>{esc(g["theme"])} ({esc(g["sentiment"])})</h3><p>{esc(g["summary"])}</p><blockquote>{esc(g["quote"])}</blockquote>')
            for g in c["result"]["qnaAnalysis"]["themes"]:
                doc.append(f'<h3>질문 · {esc(g["theme"])}</h3><p>{esc(g["summary"])}</p><blockquote>{esc(g["question"])}</blockquote>')
            doc.append('<h3>소구점</h3>')
            for p in c["result"]["sellingPoints"]:
                doc.append(f'<p>{esc(p["type"])}: {esc(p["text"])}</p>')
            for key in ["request", "encoderPredictions", "predictions", "classificationAudit", "diagnostics", "baselineComparison", "result"]:
                doc.append(f'<details><summary>{key}</summary><pre>{esc(json.dumps(c.get(key), ensure_ascii=False, indent=2))}</pre></details>')
        doc.append('</section>')
    (folder / "review.html").write_text('\n'.join(doc) + '</html>')


async def run(args, folder):
    sys.path.insert(0, str(RUNTIME / "src"))
    import torch
    from palda_runtime.models import ServiceEncoderModel, GroundedDecoderModel
    from palda_runtime import grounded_v2 as policy
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / "cases").mkdir(exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one free GPU")
    free, total = torch.cuda.mem_get_info()
    if total-free > 512*1024**2:
        raise RuntimeError("GPU occupied; not starting")
    enc_dir = ROOT / "artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit"
    dec_dir = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"
    published = read(RUNTIME / "artifacts/integration-release-20260910/published.json")
    for kind, directory in [("encoder", enc_dir), ("decoder", dec_dir)]:
        for name, expected in published[kind]["files"].items():
            if name != "metrics.json" and file_hash(directory/name) != expected:
                raise RuntimeError("Pinned model checksum differs: " + kind + "/" + name)
    config = {"round": args.round, "policy": policy.POLICY_VERSION,
              "runnerHash": file_hash(__file__), "policyHash": file_hash(policy.__file__),
              "runtimeHashes": {n:file_hash(RUNTIME/'src/palda_runtime'/n) for n in ['models.py','aggregation.py','schemas.py']},
              "datasetHash": file_hash(DEFAULT_OUT/'dataset.json'), "products": args.products,
              "split": "validation-exploratory", "phase": args.phase,
              "classification": "all records re-read, not the former 25 percent router",
              "classificationAndSellingModel": "base Qwen", "summaryModel": "PALDA Qwen adapter",
              "decoderBatch": 8, "maxInputTokens":8192, "contextOverflow":"exclude and mark failure; no truncation",
              "modelArtifactsVerified": True, "physicalGpu":os.environ.get('CUDA_VISIBLE_DEVICES'),
              "qualityChecks": "same-base-model support checks, not independent evaluation or human gold",
              "classificationCache":str(args.classification_cache) if args.classification_cache else None,
              "timingScope":"generation_and_checks_only_cached_classification" if args.classification_cache else "full_pipeline",
              "productionApproved":False, "humanGoldWritten":False, "timingComparableAsIsolatedAblation":False}
    if (folder/'config.json').exists() and read(folder/'config.json') != config:
        raise RuntimeError("Changed policy/config: choose a new round, do not overwrite history")
    write(folder/'config.json', config)
    (folder/'sources').mkdir(exist_ok=True)
    for source in [Path(__file__), Path(policy.__file__)]:
        target = folder/'sources'/source.name
        if not target.exists():
            target.write_text(source.read_text())
    write(folder/'status.json', {"stage":"loading", "pid":os.getpid()})
    encoder = ServiceEncoderModel(enc_dir,enc_dir/'taxonomy.json',device='cuda:0',batch_size=64)
    decoder = GroundedDecoderModel(dec_dir,device_map='cuda:0',load_in_4bit=True,batch_size=8,max_input_length=8192)
    versions = {"encoderModel":"213eb63610eeac4bb24b5d0272095b2110553a05", "decoderModel":"3b97b7aefad2769f67742beec183056f4d06804b",
                "taxonomy":"theme-v1.0.0+experimental-design-overlay", "calculation":policy.POLICY_VERSION}
    product_rows = read(DEFAULT_OUT/'dataset.json')['products'][:args.products]
    # Minimal warm-up, not a second quality sample.
    decoder._complete_many(['JSON만 출력: {"ready":true}'], max_new_tokens=16, disable_adapter=True)
    for product in product_rows:
        baseline_path = ROOT / 'artifacts/architecture-tradeoff-20260922/architecture-encoder/cases' / (product['id']+'-n100-e3-r0.json')
        baseline = read(baseline_path)
        request = baseline['request']
        if args.phase == 'holdout':
            from input_size_tradeoff import digest
            # Disjoint from the 100-row development prefix; still Validation, NOT sealed Test.
            prior_ids={i['id'] for key in ['reviewSample','qnaSample'] for i in baseline['request'][key]['items']}
            # Exclude every previously selected decoder evidence, even outside the 100-row pilot.
            for case_path in (DEFAULT_OUT/'cases').glob(product['id']+'-*.json'):
                old = read(case_path)
                for group_key in ['reviewGroups','qnaGroups']:
                    for group in old.get('decoderInput',{}).get(group_key,[]):
                        prior_ids.update(e['id'] for e in group.get('evidence',[]))
            eligible = sorted((r for r in product['items'] if r['id'] not in prior_ids),key=lambda r:(digest(f"42/{r['id']}"),r['id']))
            rows = [r for kind in ['review','qna'] for r in [e for e in eligible if e['kind']==kind][:25]]
            if len(rows) != 50: raise RuntimeError('Insufficient disjoint review/Q&A holdout')
            request = {**request, 'analysisId':request['analysisId']+'-holdout', **{
                key:{**request[key], 'items':[{'id':r['id'],'text':r['text']} for r in rows if r['kind']==kind]}
                for key,kind in [('reviewSample','review'),('qnaSample','qna')]}}
            assert not prior_ids & {r['id'] for r in rows}
        case_id=product['id']+'-'+args.phase
        path=folder/'cases'/(case_id+'.json')
        if path.exists():
            if read(path)['status']=='completed': continue
            raise RuntimeError("Existing failed case needs a new round or explicit review")
        write(folder/'status.json', {'stage':'running','current':case_id,'completed':len(list((folder/'cases').glob('*.json'))),'planned':len(product_rows),'pid':os.getpid()})
        trace={'caseId':case_id,'category':product['category'],'product':product['id'],'request':request,
               'reviewRows':len(request['reviewSample']['items']),'qnaRows':len(request['qnaSample']['items'])}
        started=time.perf_counter()
        with MemorySampler() as memory:
            if args.classification_cache:
                cache_path=args.classification_cache/'cases'/(case_id+'.json')
                cached=read(cache_path)
                if cached['request'] != request or cached['status']!='completed':
                    raise RuntimeError('Classification cache input mismatch')
                encoder_predictions=cached['encoderPredictions'];encoded=time.perf_counter()
                predictions=cached['predictions'];classification_audit=cached['classificationAudit']
                trace['classificationCache']={'path':str(cache_path),'sha256':file_hash(cache_path),
                    'sourcePolicy':cached['diagnostics']['policyVersion'],'classificationExcludedFromTiming':True}
            else:
                encoder_predictions=await encoder.classify(request)
                torch.cuda.synchronize(); encoded=time.perf_counter()
                predictions, classification_audit=policy.verify_records(decoder,encoder,request,encoder_predictions)
            torch.cuda.synchronize(); verified=time.perf_counter()
            trace.update(encoderPredictions=encoder_predictions,predictions=predictions,classificationAudit=classification_audit)
            write(folder/(case_id+'-classification.json'),trace)
            result,diagnostics,report=policy.generate_report(decoder,request,predictions,versions)
            torch.cuda.synchronize(); finished=time.perf_counter()
        trace.update(status='completed',seconds=finished-started,result=result,diagnostics=diagnostics,
                     decoderInput={k:v for k,v in report.items() if not k.startswith('_')},
                     stages={'encoder':encoded-started,'verification':verified-encoded,'generationAndChecks':finished-verified},
                     memory={'rssPeakGiB':memory.peak/2**30,'cudaMaxAllocatedGiB':torch.cuda.max_memory_allocated()/2**30},
                     baselineComparison={'sameRequest':request==baseline['request'],'baselineSeconds':baseline['seconds'] if args.phase=='development' else None,
                                         'baselineFile':str(baseline_path),'overallQualityVerdict':'pending_Codex_direct_audit'},
                     preview={'schemaVersion':'grounded-v2-experimental','report':result,'generationStatus':diagnostics['sellingGeneration']['status'],
                              'qnaIntents':diagnostics['qnaIntents'],'productionApproved':False})
        write(path,trace);render(folder)
        print(encode_progress(trace),flush=True)
    write(folder/'status.json',{'stage':'completed','completed':len(product_rows),'semanticApproval':'pending_Codex_direct_audit','finishedAt':time.time()})


def encode_progress(trace):
    return json.dumps({'case':trace['caseId'],'seconds':round(trace['seconds'],2),'classificationFailures':trace['classificationAudit']['failed'],
                       'sellingStatus':trace['diagnostics']['sellingGeneration']['status'],'points':len(trace['result']['sellingPoints'])})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',required=True)
    parser.add_argument('--products',type=int,default=3,choices=[1,2,3])
    parser.add_argument('--phase',choices=['development','holdout'],default='development')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--gpu',choices=['1','2','3'],default='1')
    parser.add_argument('--classification-cache',type=Path)
    args=parser.parse_args()
    if not args.round.replace('-','').replace('_','').isalnum(): parser.error('Simple round identifier required')
    folder=OUT/(args.round+'-'+args.phase);folder.mkdir(parents=True,exist_ok=True)
    if args.launch:
        command=[sys.executable,__file__,'--round',args.round,'--products',str(args.products),'--phase',args.phase,'--gpu',args.gpu]
        if args.classification_cache: command += ['--classification-cache',str(args.classification_cache.resolve())]
        with (folder/'run.log').open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':p.pid,'status':str(folder/'status.json')}));return
    try:
        asyncio.run(run(args,folder))
    except Exception as exc:
        write(folder/'status.json',{'stage':'failed','errorType':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':main()
