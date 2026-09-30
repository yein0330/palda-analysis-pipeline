#!/usr/bin/env python
"""Build three product-safe HTML assignments for Encoder v1 label review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def stable(seed: int, *values: object) -> str:
    return hashlib.sha256(":".join(map(str, (seed, *values))).encode()).hexdigest()


def assign_products(records: list[dict[str, Any]], reviewers: int, seed: int) -> dict[str, int]:
    products: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        products[row["productGroupId"]].append(row)
    cat1_load: dict[str, list[int]] = defaultdict(lambda: [0] * reviewers)
    total_load = [0] * reviewers
    assignment = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for product_id, rows in products.items():
        grouped[rows[0]["category"]["level1Name"]].append(product_id)
    for cat1 in sorted(grouped):
        for product_id in sorted(grouped[cat1], key=lambda value: stable(seed, cat1, value)):
            index = min(
                range(reviewers),
                key=lambda value: (cat1_load[cat1][value], total_load[value], stable(seed, product_id, value)),
            )
            assignment[product_id] = index
            count = len(products[product_id])
            cat1_load[cat1][index] += count
            total_load[index] += count
    return assignment


CSV_COLUMNS = [
    "reviewer_id", "annotation_id", "product_id", "source_split", "category_path", "record_type", "text",
    "usable", "review_labels_json", "qna_primary_theme_id", "qna_secondary_theme_ids_json",
    "model_confidence", "validation_status", "reviewer_status", "reviewer_notes",
]


def csv_row(record: dict[str, Any], reviewer_id: str) -> dict[str, Any]:
    labels = record["labels"]
    primary = labels.get("qnaPrimaryTheme")
    return {
        "reviewer_id": reviewer_id,
        "annotation_id": record["itemId"],
        "product_id": record["productGroupId"],
        "source_split": record["annotation"]["sourceSplit"],
        "category_path": record["category"]["path"],
        "record_type": record["sourceType"],
        "text": record["text"],
        "usable": str(record["usable"]).lower(),
        "review_labels_json": json.dumps(labels.get("reviewLabels", []), ensure_ascii=False, separators=(",", ":")),
        "qna_primary_theme_id": primary["themeId"] if primary else "",
        "qna_secondary_theme_ids_json": json.dumps(
            [item["themeId"] for item in labels.get("qnaSecondaryThemes", [])],
            ensure_ascii=False, separators=(",", ":"),
        ),
        "model_confidence": record["annotation"]["modelConfidence"],
        "validation_status": record["annotation"]["validationStatus"],
        "reviewer_status": record["annotation"]["reviewerStatus"],
        "reviewer_notes": "",
    }


def main() -> None:
    args = arguments()
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    taxonomy_by_id = {row["level2Id"]: row for row in taxonomy["categories"]}
    assignment = assign_products(records, args.reviewers, args.seed)
    groups = [[] for _ in range(args.reviewers)]
    for record in records:
        groups[assignment[record["productGroupId"]]].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schemaVersion": "encoder-review-assignments-v1",
        "dataset": str(args.dataset),
        "taxonomyVersion": taxonomy["taxonomyVersion"],
        "reviewers": args.reviewers,
        "records": len(records),
        "assignments": [],
    }
    for index, rows in enumerate(groups, 1):
        reviewer_id = f"annotator-{index}"
        directory = args.output_dir / reviewer_id
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_row(row, reviewer_id) for row in rows)
        ui_rows = []
        for row in rows:
            category = taxonomy_by_id[row["category"]["level2Id"]]
            allowed = [theme for theme in category["allowedThemes"] if theme["annotationReady"]]
            ui_rows.append({**row, "allowedThemes": allowed, "reviewerId": reviewer_id})
        payload = json.dumps(ui_rows, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
        storage_key = "palda-encoder-review-" + hashlib.sha256(
            f"{args.dataset.resolve()}:{reviewer_id}".encode()
        ).hexdigest()[:16]
        (directory / "review.html").write_text(
            TEMPLATE.replace("__DATA__", payload)
            .replace("__REVIEWER__", reviewer_id)
            .replace("__STORAGE_KEY__", storage_key),
            encoding="utf-8",
        )
        summary = {
            "reviewerId": reviewer_id,
            "records": len(rows),
            "products": len({row["productGroupId"] for row in rows}),
            "review": sum(row["sourceType"] == "review" for row in rows),
            "qna": sum(row["sourceType"] == "qna" for row in rows),
            "cat1": dict(Counter(row["category"]["level1Name"] for row in rows)),
            "cat2": len({row["category"]["level2Id"] for row in rows}),
        }
        (directory / "assignment.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["assignments"].append(summary)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


TEMPLATE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PALDA Encoder v1 검수 · __REVIEWER__</title><style>
:root{--bg:#f4f6fa;--paper:#fff;--line:#d9e0ea;--ink:#182230;--muted:#667085;--blue:#245dd8;--green:#087443;--red:#b42318;--amber:#a15c00}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,sans-serif}header{position:sticky;top:0;z-index:4;background:#fffffff5;border-bottom:1px solid var(--line);padding:14px 20px}.top,.filters,.actions,.pills{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.top h1{font-size:19px;margin:0 auto 0 0}.pill{background:#edf2fb;border-radius:99px;padding:5px 9px;font-weight:750}.bar{height:5px;background:#e5e8ee;margin-top:10px;border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}button,select,input,textarea{border:1px solid var(--line);background:white;border-radius:8px;padding:8px 10px}button{cursor:pointer;font-weight:700}.primary{background:var(--blue);color:white;border-color:var(--blue)}.filters{margin-top:9px}.filters input{flex:1;min-width:230px}main{max-width:980px;margin:18px auto;padding:0 16px}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px}.path{color:var(--muted);font-weight:650}.text{font-size:20px;line-height:1.7;margin:20px 0;padding:17px;background:#f8fafc;border-radius:10px;white-space:pre-wrap}.section{border-top:1px solid var(--line);padding-top:16px;margin-top:16px}.section h2{font-size:14px;margin:0 0 10px}.labels{display:flex;gap:8px;flex-wrap:wrap}.label{border:1px solid var(--line);border-radius:8px;padding:7px 9px}.positive{background:#e9f8ef;color:var(--green)}.negative{background:#fff0ee;color:var(--red)}.neutral{background:#f1f3f6}.editrow{display:grid;grid-template-columns:1fr 150px auto;gap:7px;margin:7px 0}.qgrid{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:start}.secondary{columns:2}.secondary label{display:block;padding:3px}.nav{display:flex;justify-content:space-between;margin-top:14px}.notes{width:100%;min-height:70px}.hint{color:var(--muted);font-size:12px}.empty{color:var(--muted)}@media(max-width:650px){.editrow,.qgrid{grid-template-columns:1fr}.secondary{columns:1}}
</style></head><body><header><div class="top"><h1>PALDA Encoder v1 검수 · __REVIEWER__</h1><span class="pill" id="progress"></span><button onclick="exportCsv()">CSV 저장</button></div><div class="bar"><i id="bar"></i></div><div class="filters"><select id="type"><option value="">전체 유형</option><option value="review">Review</option><option value="qna">Q&A</option></select><select id="status"><option value="">전체 상태</option><option value="pending">미검수</option><option value="approved">승인</option><option value="corrected">수정</option><option value="rejected">제외</option></select><input id="search" placeholder="카테고리·원문 검색"></div></header><main><div id="position" class="hint"></div><article class="card" id="card"></article><div class="nav"><button onclick="move(-1)">← 이전</button><button onclick="move(1)">다음 →</button></div></main><script>
const memoryStore={};const store={getItem(k){try{return window.localStorage.getItem(k)}catch(e){return memoryStore[k]||null}},setItem(k,v){try{window.localStorage.setItem(k,v)}catch(e){memoryStore[k]=v}}};
const base=__DATA__,KEY='__STORAGE_KEY__',edits=JSON.parse(store.getItem(KEY)||'{}');let index=Number(store.getItem(KEY+'-index')||0);const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function own(o,k){return Object.prototype.hasOwnProperty.call(o,k)}
function row(x){const e=edits[x.itemId]||{};return {...x,usable:own(e,'usable')?e.usable:x.usable,labels:own(e,'labels')?e.labels:x.labels,reviewerStatus:own(e,'reviewerStatus')?e.reviewerStatus:x.annotation.reviewerStatus,reviewerNotes:own(e,'reviewerNotes')?e.reviewerNotes:''}}
function filtered(){const t=document.getElementById('type').value,s=document.getElementById('status').value,q=document.getElementById('search').value.trim().toLowerCase();return base.map(row).filter(r=>(!t||r.sourceType===t)&&(!s||r.reviewerStatus===s)&&(!q||`${r.category.path} ${r.text}`.toLowerCase().includes(q)))}
function options(r,selected=''){return r.allowedThemes.filter(t=>!['OTHER','UNUSABLE'].includes(t.canonicalThemeId)).map(t=>`<option value="${esc(t.canonicalThemeId)}" ${t.canonicalThemeId===selected?'selected':''}>${esc(t.reviewLabel)} · ${esc(t.canonicalThemeId)}</option>`).join('')}
function reviewEditor(r){const ls=r.labels.reviewLabels||[];return `<div id="reviewRows">${ls.map((x,i)=>`<div class="editrow"><select class="theme">${options(r,x.themeId)}</select><select class="sentiment"><option value="positive" ${x.sentiment==='positive'?'selected':''}>긍정</option><option value="negative" ${x.sentiment==='negative'?'selected':''}>부정</option><option value="neutral" ${x.sentiment==='neutral'?'selected':''}>중립</option></select><button onclick="this.parentElement.remove()">삭제</button></div>`).join('')}</div><button onclick="addReviewRow()">+ 테마·감성 추가</button><p class="hint">같은 테마에 긍정과 부정을 각각 추가할 수 있습니다.</p>`}
function qnaEditor(r){const p=(r.labels.qnaPrimaryTheme&&r.labels.qnaPrimaryTheme.themeId)||'OTHER',sec=new Set((r.labels.qnaSecondaryThemes||[]).map(x=>x.themeId));return `<div class="qgrid"><label><input id="usable" type="checkbox" ${r.usable?'checked':''}> 분석 가능</label><div><label>대표 테마 1개</label><select id="primary"><option value="OTHER" ${p==='OTHER'?'selected':''}>기타 문의 · OTHER</option><option value="UNUSABLE" ${p==='UNUSABLE'?'selected':''}>분석 불가 · UNUSABLE</option>${options(r,p)}</select><h2>보조 테마</h2><div class="secondary">${r.allowedThemes.filter(t=>!['OTHER','UNUSABLE'].includes(t.canonicalThemeId)).map(t=>`<label><input type="checkbox" class="sec" value="${esc(t.canonicalThemeId)}" ${sec.has(t.canonicalThemeId)?'checked':''}> ${esc(t.qnaLabel)}</label>`).join('')}</div></div></div>`}
function render(){const rows=filtered();if(index>=rows.length)index=Math.max(0,rows.length-1);const reviewed=base.map(row).filter(r=>r.reviewerStatus!=='pending').length;document.getElementById('progress').textContent=`${reviewed.toLocaleString()} / ${base.length.toLocaleString()} 검수`;document.getElementById('bar').style.width=`${base.length?reviewed/base.length*100:0}%`;if(!rows.length){document.getElementById('position').textContent='0건';document.getElementById('card').innerHTML='<div class="empty">조건에 맞는 데이터가 없습니다.</div>';return}const r=rows[index];document.getElementById('position').textContent=`${index+1} / ${rows.length}`;document.getElementById('card').innerHTML=`<div class="pills"><span class="pill">${r.sourceType==='review'?'REVIEW':'Q&A'}</span><span class="pill">${esc(r.annotation.modelConfidence)}</span><span class="pill">${esc(r.annotation.validationStatus)}</span></div><div class="path">${esc(r.category.path)}</div><div class="text">${esc(r.text)}</div><section class="section"><h2>${r.sourceType==='review'?'테마별 감성':'대표·보조 문의 테마'}</h2>${r.sourceType==='review'?reviewEditor(r):qnaEditor(r)}</section><section class="section"><h2>검수 메모</h2><textarea class="notes" id="notes">${esc(r.reviewerNotes)}</textarea><div class="actions"><button class="primary" onclick="saveCurrent('approved')">그대로 승인</button><button onclick="saveCurrent('corrected')">수정 저장</button><button onclick="saveCurrent('rejected')">제외</button></div></section>`}
function addReviewRow(){const r=filtered()[index],box=document.createElement('div');box.className='editrow';box.innerHTML=`<select class="theme">${options(r)}</select><select class="sentiment"><option value="positive">긍정</option><option value="negative">부정</option><option value="neutral">중립</option></select><button onclick="this.parentElement.remove()">삭제</button>`;document.getElementById('reviewRows').appendChild(box)}
function saveCurrent(status){const r=filtered()[index];let labels={reviewLabels:[],qnaPrimaryTheme:null,qnaSecondaryThemes:[]},usable=true;if(r.sourceType==='review'){document.querySelectorAll('#reviewRows .editrow').forEach(x=>{const id=x.querySelector('.theme').value,t=r.allowedThemes.find(v=>v.canonicalThemeId===id);labels.reviewLabels.push({themeId:id,categoryThemeId:t.categoryThemeId,sentiment:x.querySelector('.sentiment').value})})}else{usable=document.getElementById('usable').checked;let p=document.getElementById('primary').value;if(!usable)p='UNUSABLE';const pt=r.allowedThemes.find(x=>x.canonicalThemeId===p);labels.qnaPrimaryTheme={themeId:p,categoryThemeId:pt.categoryThemeId};labels.qnaSecondaryThemes=[...document.querySelectorAll('.sec:checked')].filter(x=>x.value!==p).map(x=>{const t=r.allowedThemes.find(v=>v.canonicalThemeId===x.value);return {themeId:x.value,categoryThemeId:t.categoryThemeId}})}edits[r.itemId]={usable,labels,reviewerStatus:status,reviewerNotes:document.getElementById('notes').value};store.setItem(KEY,JSON.stringify(edits));move(1)}
function move(n){const rows=filtered();index=Math.max(0,Math.min(rows.length-1,index+n));store.setItem(KEY+'-index',index);render()}
function csv(v){const s=typeof v==='string'?v:JSON.stringify(v);return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s}function exportCsv(){const h=['reviewer_id','annotation_id','product_id','source_split','category_path','record_type','text','usable','review_labels_json','qna_primary_theme_id','qna_secondary_theme_ids_json','model_confidence','validation_status','reviewer_status','reviewer_notes'];const lines=[h.join(',')];base.map(row).forEach(r=>{const p=r.labels.qnaPrimaryTheme;const o={reviewer_id:r.reviewerId,annotation_id:r.itemId,product_id:r.productGroupId,source_split:r.annotation.sourceSplit,category_path:r.category.path,record_type:r.sourceType,text:r.text,usable:String(r.usable).toLowerCase(),review_labels_json:JSON.stringify(r.labels.reviewLabels||[]),qna_primary_theme_id:(p&&p.themeId)||'',qna_secondary_theme_ids_json:JSON.stringify((r.labels.qnaSecondaryThemes||[]).map(x=>x.themeId)),model_confidence:r.annotation.modelConfidence,validation_status:r.annotation.validationStatus,reviewer_status:r.reviewerStatus,reviewer_notes:r.reviewerNotes};lines.push(h.map(k=>csv(o[k]==null?'':o[k])).join(','))});const a=document.createElement('a'),b=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'});a.href=URL.createObjectURL(b);a.download='palda-encoder-v1-__REVIEWER__-reviewed.csv';a.click();URL.revokeObjectURL(a.href)}document.querySelectorAll('.filters input,.filters select').forEach(x=>x.oninput=()=>{index=0;render()});render();
</script></body></html>'''


if __name__ == "__main__":
    main()
