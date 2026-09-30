#!/usr/bin/env python
"""Build a dependency-free, one-record-at-a-time HTML annotation reviewer."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--service-drafts", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-id", default="all")
    return parser.parse_args()


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    for tag in ("<br>", "<br/>", "<br />", "<BR>"):
        text = text.replace(tag, "\n")
    return text.strip()


SERVICE_GROUP = {
    "delivery_speed": "delivery", "delivery_reliability": "delivery",
    "delivery_staff": "delivery", "packaging": "delivery",
    "customer_support": "support", "returns_refunds": "support",
    "installation": "installation_as", "after_sales": "installation_as",
    "delivery": "delivery", "support": "support", "installation_as": "installation_as",
}


def simplify_service_mentions(mentions: list[dict]) -> list[dict]:
    """Collapse the detailed service taxonomy to three reviewer-friendly groups."""
    grouped: dict[str, dict] = {}
    polarity_sets: dict[str, set[str]] = {}
    for mention in mentions:
        group = SERVICE_GROUP.get(str(mention.get("service_group") or mention.get("service_type") or ""))
        evidence = clean_text(mention.get("evidence_quote"))
        if not group or not evidence:
            continue
        polarity = str(mention.get("polarity") or "neutral")
        polarity_sets.setdefault(group, set()).add(polarity)
        grouped.setdefault(group, {
            "service_group": group, "polarity": polarity, "evidence_quote": evidence,
        })
    for group, polarities in polarity_sets.items():
        if len(polarities) > 1:
            grouped[group]["polarity"] = "mixed"
    return list(grouped.values())


def main() -> None:
    args = arguments()
    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = {row["annotation_id"]: row for row in csv.DictReader(handle)}
    service_rows = {}
    for line in args.service_drafts.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        service_rows[row["annotation_id"]] = row
    records = []
    for line in args.drafts.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line); source = raw["input"]; draft = raw.get("draft") or {}
        overlay = csv_rows[source["annotation_id"]]
        service_row = service_rows.get(source["annotation_id"], {})
        service_draft = service_row.get("draft") or {}
        overlay_services = []
        for index in range(1, 4):
            service_name = (overlay.get(f"service_group_{index}") or overlay.get(f"service_type_{index}") or "").strip()
            if service_name:
                overlay_services.append({
                    "service_group": service_name,
                    "polarity": overlay.get(f"service_polarity_{index}") or "neutral",
                    "evidence_quote": clean_text(overlay.get(f"service_evidence_{index}")),
                })
        service_mentions = simplify_service_mentions(overlay_services or service_draft.get("mentions") or [])
        mentions = []
        for index in range(1, 4):
            theme_id = (overlay.get(f"theme_{index}") or "").strip()
            if theme_id:
                mentions.append({
                    "theme_id": theme_id,
                    "polarity": overlay.get(f"polarity_{index}") or "",
                    "evidence_quote": clean_text(overlay.get(f"evidence_{index}")),
                })
        records.append({
            "reviewer_id": args.reviewer_id,
            "annotation_id": source["annotation_id"], "split": source["split"],
            "product_id": source["product_id"], "product_name": clean_text(source["product_name"]),
            "category_path": source["category_path"], "record_type": source["record_type"],
            "text": clean_text(source["text"]), "score": source.get("score"),
            "allowed_themes": source["allowed_themes"],
            "qna_stage": overlay.get("qna_stage") or None, "mentions": mentions,
            "new_theme_candidates": overlay.get("new_theme_candidates") or "",
            "confidence": overlay.get("confidence") or draft.get("confidence") or "medium",
            "validation_status": raw["validation_status"],
            "validation_errors": raw.get("validation_errors") or [],
            "reviewer_status": overlay.get("reviewer_status") or "pending",
            "reviewer_notes": overlay.get("reviewer_notes") or "",
            "service_mentions": service_mentions,
            "service_confidence": overlay.get("service_confidence") or service_draft.get("confidence") or "medium",
            "service_validation_status": overlay.get("service_validation_status") or service_row.get("validation_status") or "needs_review",
            "service_validation_errors": ([overlay.get("service_validation_errors")] if overlay.get("service_validation_errors") else service_row.get("validation_errors") or []),
        })
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    storage_key = f"palda-review-v4-{args.reviewer_id}"
    download_name = f"palda-reviewed-{args.reviewer_id}.csv"
    page = (TEMPLATE.replace("__PALDA_DATA__", payload)
            .replace("__PALDA_REVIEWER_ID__", html.escape(args.reviewer_id))
            .replace("__PALDA_STORAGE_KEY__", json.dumps(storage_key))
            .replace("__PALDA_DOWNLOAD_NAME__", json.dumps(download_name)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "records": len(records),
        "pending": sum(row["reviewer_status"] == "pending" for row in records),
        "reviewed": sum(row["reviewer_status"] != "pending" for row in records),
    }, ensure_ascii=False, indent=2))


TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PALDA 원문 라벨 검수</title>
<style>
:root{--ink:#172033;--muted:#667085;--line:#e4e7ec;--paper:#fff;--bg:#f3f6fa;--blue:#2764e7;--purple:#7c3aed;--green:#138a55;--red:#c23b45;--amber:#a86606}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Pretendard,"Noto Sans KR",system-ui,sans-serif}
button,select,input,textarea{font:inherit} button{cursor:pointer}.top{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.95);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
.topin{max-width:1120px;margin:auto;padding:13px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}.brand{font-size:19px;font-weight:800;margin-right:auto}.progress{font-weight:700}.bar{width:150px;height:8px;background:#e8edf5;border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;background:var(--green)}
.filters{max-width:1120px;margin:auto;padding:0 20px 12px;display:flex;gap:8px;flex-wrap:wrap}.filters select,.filters input{border:1px solid var(--line);border-radius:9px;background:#fff;padding:8px 10px}.filters input{min-width:230px;flex:1}
.wrap{max-width:1120px;margin:22px auto 110px;padding:0 20px}.card{background:var(--paper);border:1px solid var(--line);border-radius:17px;box-shadow:0 8px 28px rgba(38,51,77,.07);overflow:hidden}.meta{padding:18px 22px;border-bottom:1px solid var(--line)}
.badges{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}.badge{font-size:12px;font-weight:700;padding:5px 9px;border-radius:99px;background:#edf2ff;color:#3158ae}.badge.warn{background:#fff0df;color:#9a5900}.badge.ok{background:#e8f7ef;color:#087343}.badge.review{background:#f2edff;color:#6941c6}
.category{font-size:13px;color:var(--muted);margin-bottom:5px}.product{font-size:17px;font-weight:750}.section{padding:22px}.label{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:800;margin-bottom:9px}.original{font-size:22px;line-height:1.72;white-space:pre-wrap;background:#f8fafc;border-radius:12px;padding:20px;word-break:keep-all;overflow-wrap:anywhere}mark{background:#fff0a8;padding:1px 2px;border-radius:3px}
.analysis{border-top:1px solid var(--line)}.service{border-top:1px solid var(--line);background:#fcfaff}.stage{display:flex;align-items:center;gap:10px;margin-bottom:14px}.stage strong{min-width:100px}.mentions{display:grid;gap:11px}.mention{border:1px solid var(--line);border-left:5px solid var(--blue);border-radius:11px;padding:14px 16px}.service .mention{border-left-color:var(--purple);background:#fff}.mentionhead{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}.theme{font-weight:850;font-size:17px}.pol{font-size:12px;padding:4px 8px;border-radius:99px;background:#eef2f7;font-weight:750}.evidence{color:#344054;line-height:1.55}.empty{padding:17px;border:1px dashed #b8c1cf;border-radius:11px;color:var(--muted)}
.errors{margin-top:15px;background:#fff1f2;color:#9f2731;border-radius:10px;padding:12px 14px;white-space:pre-wrap}.notes{margin-top:15px;color:var(--muted)}.editbox{display:none;margin-top:16px;border-top:1px dashed var(--line);padding-top:16px}.editing .editbox{display:block}.editing .readonly{display:none}.editrow{display:grid;grid-template-columns:1fr 160px 2fr auto;gap:8px;margin-bottom:8px}.editrow select,.editrow input,.editbox textarea,.editbox>select{width:100%;border:1px solid #cfd5df;border-radius:8px;padding:9px;background:#fff}.remove{border:0;background:#fff0f0;color:var(--red);border-radius:8px;padding:8px 11px}.smallbtn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 11px}
.actions{position:fixed;bottom:0;left:0;right:0;background:rgba(255,255,255,.96);border-top:1px solid var(--line);padding:13px 20px;z-index:12}.actionsin{max-width:1080px;margin:auto;display:flex;gap:9px;align-items:center}.btn{border:0;border-radius:10px;padding:12px 17px;font-weight:800}.prev,.next,.edit{background:#eef1f6;color:#344054}.approve{background:var(--green);color:#fff;margin-left:auto}.reject{background:#fff0f1;color:var(--red)}.pending{background:#fff3dc;color:var(--amber)}.export{background:var(--blue);color:#fff}.count{font-size:13px;color:var(--muted);min-width:85px;text-align:center}.done{text-align:center;padding:70px 20px}.shortcut{font-size:11px;opacity:.7;margin-left:4px}
@media(max-width:720px){.original{font-size:18px}.editrow{grid-template-columns:1fr}.actionsin{flex-wrap:wrap}.approve{margin-left:0}.btn{padding:10px 12px}.bar{display:none}.section{padding:17px}.wrap{padding:0 10px}.topin,.filters{padding-left:12px;padding-right:12px}}
</style>
</head>
<body>
<header class="top"><div class="topin"><div class="brand">PALDA 원문 검수 · __PALDA_REVIEWER_ID__</div><div class="progress" id="progress"></div><div class="bar"><i id="bar"></i></div><button class="btn export" onclick="exportCSV()">검수 결과 CSV 저장</button></div>
<div class="filters"><select id="statusFilter"><option value="all">검수상태 전체</option><option value="pending">미검수만</option><option value="approved">승인만</option><option value="needs_review">구조 오류만</option></select><select id="serviceFilter"><option value="all">서비스 전체</option><option value="relevant">서비스 언급만</option><option value="none">서비스 언급 없음</option><option value="needs_review">서비스 구조 오류</option></select><select id="typeFilter"><option value="all">Review + Q&A</option><option value="review">Review</option><option value="qna">Q&A</option></select><input id="search" placeholder="원문·상품명·카테고리 검색"></div></header>
<main class="wrap" id="app"></main>
<footer class="actions"><div class="actionsin"><button class="btn prev" onclick="move(-1)">← 이전</button><span class="count" id="position"></span><button class="btn next" onclick="move(1)">다음 →</button><button class="btn edit" onclick="toggleEdit()">수정 <span class="shortcut">E</span></button><button class="btn pending" onclick="setStatus('pending')">보류</button><button class="btn reject" onclick="setStatus('rejected')">거절 <span class="shortcut">R</span></button><button class="btn approve" onclick="approve()">승인 후 다음 <span class="shortcut">A</span></button></div></footer>
<script type="application/json" id="dataset">__PALDA_DATA__</script>
<script>
const VERSION=__PALDA_STORAGE_KEY__; const DOWNLOAD_NAME=__PALDA_DOWNLOAD_NAME__; const base=JSON.parse(document.getElementById('dataset').textContent);
let edits=JSON.parse(localStorage.getItem(VERSION)||'{}'), index=Number(localStorage.getItem(VERSION+'-index')||0), editing=false;
const get=r=>edits[r.annotation_id]?{...r,...edits[r.annotation_id]}:r;
function save(r){edits[r.annotation_id]={qna_stage:r.qna_stage,mentions:r.mentions,new_theme_candidates:r.new_theme_candidates,confidence:r.confidence,reviewer_status:r.reviewer_status,reviewer_notes:r.reviewer_notes,service_mentions:r.service_mentions,service_confidence:r.service_confidence};localStorage.setItem(VERSION,JSON.stringify(edits))}
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function filtered(){const sf=document.getElementById('statusFilter').value,vf=document.getElementById('serviceFilter').value,tf=document.getElementById('typeFilter').value,q=document.getElementById('search').value.trim().toLowerCase();return base.map(get).filter(r=>{let ok=sf==='all'||(sf==='needs_review'?(r.validation_status==='needs_review'||r.service_validation_status==='needs_review'):r.reviewer_status===sf);let vok=vf==='all'||(vf==='relevant'?(r.service_mentions||[]).length>0:vf==='none'?(r.service_mentions||[]).length===0:r.service_validation_status==='needs_review');return ok&&vok&&(tf==='all'||r.record_type===tf)&&(!q||[r.text,r.product_name,r.category_path,r.annotation_id].join(' ').toLowerCase().includes(q))})}
function highlight(text,mentions){let ranges=[];for(const m of mentions||[]){let p=text.indexOf(m.evidence_quote);if(p>=0)ranges.push([p,p+m.evidence_quote.length])}ranges.sort((a,b)=>a[0]-b[0]);let out='',at=0;for(const [s,e] of ranges){if(s<at)continue;out+=esc(text.slice(at,s))+'<mark>'+esc(text.slice(s,e))+'</mark>';at=e}return out+esc(text.slice(at))}
function statusName(s){return {pending:'미검수',approved:'승인',corrected:'수정',rejected:'거절'}[s]||s}
const SERVICE_LABELS={delivery:'배송',support:'고객지원',installation_as:'설치·AS'};
const POLARITY_LABELS={positive:'긍정',negative:'부정',neutral:'중립',concern:'문의·우려',mixed:'복합'};
function cards(ms,labeler,empty){return (ms||[]).map(m=>`<div class="mention"><div class="mentionhead"><span class="theme">${esc(labeler(m))}</span><span class="pol">${esc(m.polarity)}</span></div><div class="evidence">“${esc(m.evidence_quote)}”</div></div>`).join('')||`<div class="empty">${empty}</div>`}
function render(){const rows=filtered();if(index>=rows.length)index=Math.max(0,rows.length-1);localStorage.setItem(VERSION+'-index',index);const all=base.map(get), reviewed=all.filter(r=>r.reviewer_status!=='pending').length;document.getElementById('progress').textContent=`${reviewed.toLocaleString()} / ${all.length.toLocaleString()} 검수`;document.getElementById('bar').style.width=(reviewed/all.length*100)+'%';document.getElementById('position').textContent=rows.length?`${index+1} / ${rows.length}`:'0건';if(!rows.length){document.getElementById('app').innerHTML='<div class="card done"><h2>이 조건의 데이터는 모두 확인했어요 🎉</h2><p>필터를 바꾸거나 검수 결과를 저장하세요.</p></div>';return}const r=rows[index];let mentions=cards(r.mentions,m=>labelFor(r,m.theme_id),'이 원문에서 추출된 상품 Theme가 없습니다.');let services=(r.service_mentions||[]).map(m=>`<div class="mention"><div class="mentionhead"><span class="theme">${esc(SERVICE_LABELS[m.service_group]||m.service_group)}</span><span class="pol">${esc(POLARITY_LABELS[m.polarity]||m.polarity)}</span></div><div class="evidence">“${esc(m.evidence_quote)}”</div></div>`).join('')||'<div class="empty">서비스 언급 없음</div>';let themeOpts=r.allowed_themes.map(t=>`<option value="${esc(t.theme_id)}">${esc(t.label)} (${esc(t.theme_id)})</option>`).join('');let editRows=(r.mentions||[]).map((m,i)=>editRow(themeOpts,m,i)).join('');let serviceOpts=Object.entries(SERVICE_LABELS).map(([v,l])=>`<option value="${v}">${l}</option>`).join('');let serviceEditRows=(r.service_mentions||[]).map((m,i)=>serviceEditRow(serviceOpts,m,i)).join('');document.getElementById('app').innerHTML=`<article class="card ${editing?'editing':''}"><div class="meta"><div class="badges"><span class="badge ${r.record_type==='review'?'review':''}">${r.record_type==='review'?'REVIEW':'Q&A'}</span><span class="badge ${r.validation_status==='valid'?'ok':'warn'}">${r.validation_status==='valid'?'상품 구조검증 통과':'상품 구조확인 필요'}</span><span class="badge ${r.service_validation_status==='valid'?'ok':'warn'}">${r.service_validation_status==='valid'?'서비스 구조검증 통과':'서비스 구조확인 필요'}</span><span class="badge">${statusName(r.reviewer_status)}</span>${r.score!==null?`<span class="badge">별점 ${r.score}</span>`:''}</div><div class="category">${esc(r.category_path)}</div><div class="product">${esc(r.product_name)}</div></div><section class="section"><div class="label">실제 원문</div><div class="original">${highlight(r.text,[...(r.mentions||[]),...(r.service_mentions||[])])}</div></section><section class="section analysis"><div class="label">상품 Theme 평가</div>${r.record_type==='qna'?`<div class="stage"><strong>Q&A 단계</strong><span>${esc(r.qna_stage||'미지정')}</span></div>`:''}<div class="readonly mentions">${mentions}</div>${r.validation_errors.length?`<div class="errors"><strong>상품 자동검증 확인사항</strong>\n${esc(r.validation_errors.join('\n'))}</div>`:''}<div class="editbox"><label>Q&A 단계</label><select id="editStage"><option value="">해당 없음</option><option value="pre_purchase">구매 전 질문</option><option value="post_purchase_support">구매 후 고객지원</option><option value="irrelevant">분석 불가</option></select><div id="editMentions">${editRows}</div><button class="smallbtn" onclick="addMention()">+ 상품 Theme 추가</button></div></section><section class="section service"><div class="label">간단 서비스 평가</div><div class="readonly mentions">${services}</div>${r.service_validation_errors.length?`<div class="errors"><strong>서비스 자동검증 확인사항</strong>\n${esc(r.service_validation_errors.join('\n'))}</div>`:''}<div class="editbox"><div id="editServices">${serviceEditRows}</div><button class="smallbtn" onclick="addService()">+ 서비스 그룹 추가</button><label>검수 메모</label><textarea id="editNotes" rows="3">${esc(r.reviewer_notes)}</textarea><button class="smallbtn" onclick="applyEdit()">수정 내용 적용</button></div></section></article>`;if(editing){document.getElementById('editStage').value=r.qna_stage||'';document.querySelectorAll('#editMentions .editrow').forEach((row,i)=>{row.querySelector('select').value=r.mentions[i].theme_id;row.querySelector('.epol').value=r.mentions[i].polarity});document.querySelectorAll('#editServices .editrow').forEach((row,i)=>{row.querySelector('select').value=r.service_mentions[i].service_group;row.querySelector('.epol').value=r.service_mentions[i].polarity})}}
function labelFor(r,id){return (r.allowed_themes.find(t=>t.theme_id===id)||{}).label||id}
function editRow(opts,m={},i=0){return `<div class="editrow" data-row="${i}"><select data-theme="${i}">${opts}</select><select class="epol"><option>positive</option><option>negative</option><option>neutral</option><option>concern</option></select><input class="eevidence" value="${esc(m.evidence_quote||'')}" placeholder="원문 근거"><button class="remove" onclick="this.parentElement.remove()">삭제</button></div>`}
function serviceEditRow(opts,m={},i=0){return `<div class="editrow" data-row="${i}"><select>${opts}</select><select class="epol"><option value="positive">긍정</option><option value="negative">부정</option><option value="neutral">중립</option><option value="concern">문의·우려</option><option value="mixed">복합</option></select><input class="eevidence" value="${esc(m.evidence_quote||'')}" placeholder="서비스 근거 원문"><button class="remove" onclick="this.parentElement.remove()">삭제</button></div>`}
function current(){const rows=filtered();return rows[index]}
function move(d){editing=false;const n=filtered().length;index=Math.max(0,Math.min(n-1,index+d));render()}
function toggleEdit(){editing=!editing;render()}
function applyEdit(){let r=current(),ms=[],ss=[];document.querySelectorAll('#editMentions .editrow').forEach(row=>ms.push({theme_id:row.querySelector('select').value,polarity:row.querySelector('.epol').value,evidence_quote:row.querySelector('.eevidence').value}));document.querySelectorAll('#editServices .editrow').forEach(row=>ss.push({service_group:row.querySelector('select').value,polarity:row.querySelector('.epol').value,evidence_quote:row.querySelector('.eevidence').value}));r.mentions=ms;r.service_mentions=ss;r.qna_stage=document.getElementById('editStage').value||null;r.reviewer_notes=document.getElementById('editNotes').value;r.reviewer_status='corrected';save(r);editing=false;render()}
function addMention(){const r=current(),opts=r.allowed_themes.map(t=>`<option value="${esc(t.theme_id)}">${esc(t.label)} (${esc(t.theme_id)})</option>`).join('');document.getElementById('editMentions').insertAdjacentHTML('beforeend',editRow(opts,{},Date.now()))}
function addService(){const opts=Object.entries(SERVICE_LABELS).map(([v,l])=>`<option value="${v}">${l}</option>`).join('');document.getElementById('editServices').insertAdjacentHTML('beforeend',serviceEditRow(opts,{},Date.now()))}
function setStatus(s){let r=current();r.reviewer_status=s;save(r);render()}
function approve(){let r=current();r.reviewer_status='approved';save(r);editing=false;const rows=filtered();if(document.getElementById('statusFilter').value==='pending'){}else index=Math.min(index+1,Math.max(0,rows.length-1));render()}
function csv(v){return '"'+String(v==null?'':v).split('"').join('""')+'"'}
function exportCSV(){const headers=['reviewer_id','annotation_id','split','product_id','product_name','category_path','record_type','score','text','qna_stage','theme_1','polarity_1','evidence_1','theme_2','polarity_2','evidence_2','theme_3','polarity_3','evidence_3','new_theme_candidates','confidence','validation_status','validation_errors','service_relevant','service_group_1','service_polarity_1','service_evidence_1','service_group_2','service_polarity_2','service_evidence_2','service_group_3','service_polarity_3','service_evidence_3','service_confidence','service_validation_status','service_validation_errors','reviewer_status','reviewer_notes'];let lines=[headers.map(csv).join(',')];for(const r of base.map(get)){let row={...r,validation_errors:(r.validation_errors||[]).join(' | '),service_relevant:(r.service_mentions||[]).length>0,service_validation_errors:(r.service_validation_errors||[]).join(' | ')};(r.mentions||[]).slice(0,3).forEach((m,i)=>{row['theme_'+(i+1)]=m.theme_id;row['polarity_'+(i+1)]=m.polarity;row['evidence_'+(i+1)]=m.evidence_quote});(r.service_mentions||[]).slice(0,3).forEach((m,i)=>{row['service_group_'+(i+1)]=m.service_group;row['service_polarity_'+(i+1)]=m.polarity;row['service_evidence_'+(i+1)]=m.evidence_quote});lines.push(headers.map(h=>csv(row[h])).join(','))}let blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=DOWNLOAD_NAME;a.click();URL.revokeObjectURL(a.href)}
document.getElementById('statusFilter').onchange=()=>{index=0;render()};document.getElementById('serviceFilter').onchange=()=>{index=0;render()};document.getElementById('typeFilter').onchange=()=>{index=0;render()};document.getElementById('search').oninput=()=>{index=0;render()};document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1);if(e.key.toLowerCase()==='a')approve();if(e.key.toLowerCase()==='r')setStatus('rejected');if(e.key.toLowerCase()==='e')toggleEdit()});render();
</script>
</body></html>'''


if __name__ == "__main__":
    main()
