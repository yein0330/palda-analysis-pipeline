#!/usr/bin/env python
"""Build a searchable standalone HTML for Cat2 theme candidate review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.review.read_text(encoding="utf-8").splitlines()]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = TEMPLATE.replace("__DATA__", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "cat2": len(rows),
        "themes": sum(len(row["accepted_themes"]) for row in rows),
        "evidence_warnings": sum(
            not theme["evidence_exact"] for row in rows for theme in row["accepted_themes"]
        ),
    }, ensure_ascii=False, indent=2))


TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PALDA Cat2 Theme 검수</title><style>
:root{--bg:#f4f6fa;--paper:#fff;--line:#dfe4ec;--ink:#172033;--muted:#667085;--blue:#245dd8;--green:#087443;--red:#b42318;--amber:#a15c00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:4;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);padding:17px 24px}h1{margin:0 0 10px;font-size:22px}.summary,.filters{display:flex;gap:8px;flex-wrap:wrap}.pill{background:#edf2fb;border-radius:99px;padding:6px 10px;font-weight:750}.filters{margin-top:10px}select,input{border:1px solid var(--line);border-radius:9px;padding:9px 11px;background:white}input{min-width:320px;flex:1}
main{max-width:1200px;margin:20px auto;padding:0 18px}.count{color:var(--muted);margin:8px 0 12px}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin-bottom:13px;overflow:hidden}.head{padding:17px 20px;display:flex;gap:12px;align-items:center}.path{font-size:17px;font-weight:800;flex:1}.badge{font-size:12px;border-radius:99px;padding:5px 8px;background:#e9f7ef;color:var(--green);font-weight:750}.badge.warn{background:#fff1dc;color:var(--amber)}.body{border-top:1px solid var(--line);padding:18px 20px}.section{margin-bottom:17px}.label{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:800;margin-bottom:8px}.themes{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:9px}.theme{border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:9px;padding:11px 13px}.theme h3{margin:0 0 4px;font-size:16px}.id{color:var(--muted);font-family:ui-monospace,monospace}.kw{margin-top:7px;color:#344054}.evidence{margin-top:8px;padding:8px;background:#f8fafc;border-radius:7px}.bad{color:var(--red)}details{margin-top:10px}summary{cursor:pointer;font-weight:750}.sample{padding:8px 0;border-bottom:1px solid #eef0f3}.type{font-size:11px;color:var(--blue);font-weight:800;margin-right:7px}.inherited{color:var(--muted)}
</style></head><body><header><h1>PALDA Cat2 Theme 후보 검수</h1><div class="summary" id="summary"></div><div class="filters"><select id="cat1"><option value="">Cat1 전체</option></select><select id="warning"><option value="">전체 상태</option><option value="clean">근거 일치</option><option value="warn">근거 확인 필요</option></select><input id="search" placeholder="Cat2, Theme ID, 라벨, 키워드 검색"></div></header><main><div class="count" id="count"></div><div id="cards"></div></main>
<script>const rows=__DATA__;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const cat=document.getElementById('cat1');[...new Set(rows.map(r=>r.cat1))].sort().forEach(x=>cat.add(new Option(x,x)));
function hasWarn(r){return r.accepted_themes.some(t=>!t.evidence_exact)||r.rejected_themes.length>0}
function exmeta(x){return [x.record_type,x.sentiment_bucket,x.category_path,x.product_name].filter(Boolean).map(v=>`<span class="type">${esc(v)}</span>`).join('')}
function examples(xs){return `<details><summary>키워드 연관 예시 ${(xs||[]).length}건</summary>${(xs||[]).map(x=>`<div class="sample">${exmeta(x)}${esc(x.text)}</div>`).join('')||'<div class="sample">현재 표본에서 키워드 직접 매칭 없음</div>'}</details>`}
function render(){const c=cat.value,w=document.getElementById('warning').value,q=document.getElementById('search').value.trim().toLowerCase();const out=rows.filter(r=>(!c||r.cat1===c)&&(!w||(w==='warn')===hasWarn(r))&&(!q||JSON.stringify([r.cat2_path,r.accepted_themes]).toLowerCase().includes(q)));document.getElementById('count').textContent=`${out.length.toLocaleString()}개 Cat2 표시`;document.getElementById('cards').innerHTML=out.map(r=>`<article class="card"><div class="head"><div class="path">${esc(r.cat2_path)}</div><span class="badge">leaf ${r.diversity?.leaf_categories||0} · 상품 ${r.diversity?.products||0} · 예시 ${r.diversity?.examples||r.sample_records.length}</span><span class="badge ${hasWarn(r)?'warn':''}">${r.accepted_themes.length}개 Theme · ${hasWarn(r)?'확인 필요':'근거 일치'}</span></div><div class="body"><div class="section"><div class="label">신규 Cat2 Theme</div><div class="themes">${r.accepted_themes.map(t=>`<div class="theme"><h3>${esc(t.label)} <span class="id">${esc(t.theme_id)}</span></h3><div class="kw">${t.keywords.map(esc).join(' · ')}</div><div class="evidence ${t.evidence_exact?'':'bad'}">“${esc(t.evidence_quote)}” ${t.evidence_exact?'':'⚠ 원문 완전일치 확인'}</div>${examples(t.matched_examples||[])}</div>`).join('')}</div></div><div class="section"><div class="label">상속 Theme</div><div class="inherited">${r.inherited_themes.map(t=>`${esc(t.label)} (${esc(t.theme_id)})`).join(' · ')}</div></div>${r.rejected_themes.length?`<details><summary>제외 후보 ${r.rejected_themes.length}개</summary>${r.rejected_themes.map(t=>`<div class="sample bad">${esc(t.label)} (${esc(t.theme_id)}): ${esc(t.rejection_reasons.join(', '))}</div>`).join('')}</details>`:''}<details><summary>다양성 예시 ${(r.diverse_examples||r.sample_records).length}건 전체 보기</summary>${(r.diverse_examples||r.sample_records).map(x=>`<div class="sample">${exmeta(x)}${esc(x.text)}</div>`).join('')}</details></div></article>`).join('')}
const warnings=rows.filter(hasWarn).length,themes=rows.reduce((n,r)=>n+r.accepted_themes.length,0);document.getElementById('summary').innerHTML=`<span class="pill">Cat2 ${rows.length}</span><span class="pill">신규 Theme ${themes}</span><span class="pill">확인 필요 Cat2 ${warnings}</span><span class="pill">전체 raw 표본 ${(rows.length*30).toLocaleString()}건</span>`;document.querySelectorAll('select,input').forEach(x=>x.oninput=render);render();</script></body></html>'''


if __name__ == "__main__":
    main()
