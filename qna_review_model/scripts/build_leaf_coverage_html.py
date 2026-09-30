#!/usr/bin/env python
"""Build a searchable standalone HTML report for leaf-category coverage."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    with args.coverage_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PALDA 5,090 카테고리 커버리지</title><style>
:root{{--bg:#f4f6f8;--card:#fff;--line:#dfe4ea;--text:#17212b;--muted:#667085;--ok:#087443;--warn:#b54708;--bad:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--line);padding:18px 24px}}h1{{margin:0 0 12px;font-size:22px}}
.summary,.filters{{display:flex;gap:8px;flex-wrap:wrap}}.pill{{padding:6px 10px;border-radius:999px;background:#eef2f6;font-weight:700}}
.filters{{margin-top:12px}}select,input{{border:1px solid var(--line);border-radius:8px;padding:9px 11px;background:#fff}}input{{min-width:320px;flex:1}}
main{{padding:20px 24px}}.count{{color:var(--muted);margin-bottom:10px}}.table{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #edf0f3;text-align:left}}th{{position:sticky;top:0;background:#f8fafc}}
.ok{{color:var(--ok);font-weight:750}}.warn{{color:var(--warn);font-weight:750}}.bad{{color:var(--bad);font-weight:750}}.path{{white-space:normal;min-width:480px}}
</style></head><body><header><h1>PALDA 리프 카테고리 5,090개 커버리지</h1><div class="summary" id="summary"></div>
<div class="filters"><select id="cat1"><option value="">대분류 전체</option></select><select id="state"><option value="">상태 전체</option><option value="missing">크롤링 누락</option><option value="no_records">원문 없음</option><option value="partial">20건 미만</option><option value="ready">20건 확보</option></select><input id="search" placeholder="CID 또는 카테고리 경로 검색"></div></header>
<main><div class="count" id="count"></div><div class="table"><table><thead><tr><th>상태</th><th>CID</th><th>카테고리 경로</th><th>상품</th><th>Review</th><th>Q&amp;A</th><th>표본 가능</th></tr></thead><tbody id="body"></tbody></table></div></main>
<script>const rows={payload};
const n=x=>Number(x||0),yes=x=>String(x).toLowerCase()==='true';
const cat=document.getElementById('cat1');[...new Set(rows.map(r=>r.cat1))].sort().forEach(x=>cat.add(new Option(x,x)));
function state(r){{if(!yes(r.crawl_attempted))return'missing';if(n(r.collected_record_count)===0)return'no_records';if(n(r.sample_capacity)<20)return'partial';return'ready'}}
function label(s){{return {{missing:'크롤링 누락',no_records:'원문 없음',partial:'20건 미만',ready:'20건 확보'}}[s]}}
function cls(s){{return s==='ready'?'ok':s==='partial'?'warn':'bad'}}
function render(){{const c=cat.value,s=document.getElementById('state').value,q=document.getElementById('search').value.trim().toLowerCase();const out=rows.filter(r=>(!c||r.cat1===c)&&(!s||state(r)===s)&&(!q||(r.category_cid+' '+r.category_path).toLowerCase().includes(q)));document.getElementById('count').textContent=`${{out.length.toLocaleString()}}개 표시`;document.getElementById('body').innerHTML=out.map(r=>`<tr><td class="${{cls(state(r))}}">${{label(state(r))}}</td><td>${{r.category_cid}}</td><td class="path">${{r.category_path}}</td><td>${{n(r.inventory_product_count).toLocaleString()}}</td><td>${{n(r.collected_review_count).toLocaleString()}}</td><td>${{n(r.collected_qna_count).toLocaleString()}}</td><td>${{n(r.sample_capacity)}}</td></tr>`).join('')}}
const counts={{ready:0,partial:0,no_records:0,missing:0}};rows.forEach(r=>counts[state(r)]++);document.getElementById('summary').innerHTML=`<span class="pill">전체 ${{rows.length.toLocaleString()}}</span><span class="pill">20건 확보 ${{counts.ready.toLocaleString()}}</span><span class="pill">20건 미만 ${{counts.partial.toLocaleString()}}</span><span class="pill">원문 없음 ${{counts.no_records.toLocaleString()}}</span><span class="pill">크롤링 누락 ${{counts.missing.toLocaleString()}}</span>`;
document.querySelectorAll('select,input').forEach(x=>x.addEventListener('input',render));render();</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "categories": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
