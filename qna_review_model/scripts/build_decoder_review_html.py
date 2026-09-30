#!/usr/bin/env python
"""Build a single-reviewer HTML for product-level Decoder report drafts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows = []
    for split in ("train", "validation", "test"):
        path = args.dataset_dir / f"{split}.drafts.jsonl"
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    embedded = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PALDA Decoder 보고서 검수</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#20242a;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#17202b;color:white;padding:12px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,select{padding:8px 12px;border:1px solid #bbc2cc;border-radius:8px;background:white;cursor:pointer}
header button{border:0}.primary{background:#2764d7;color:white}.danger{background:#b23b3b;color:white}
main{max-width:1500px;margin:auto;padding:16px}.meta{background:white;border-radius:12px;padding:14px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:white;border-radius:12px;padding:14px;min-width:0}
.group{border:1px solid #dde1e7;border-radius:9px;padding:10px;margin:9px 0}.tag{display:inline-block;background:#edf2ff;padding:3px 7px;border-radius:99px;margin-right:5px;font-size:12px}
.evidence{background:#f7f8fa;border-left:3px solid #8da2c0;margin:6px 0;padding:7px;font-size:13px;white-space:pre-wrap}
textarea{width:100%;min-height:620px;border:1px solid #bbc2cc;border-radius:9px;padding:12px;font:13px/1.5 ui-monospace,monospace}
.actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}.ok{color:#087443}.warn{color:#a05a00}
@media(max-width:900px){.grid{grid-template-columns:1fr}textarea{min-height:420px}}
</style></head><body>
<header>
<strong>PALDA Decoder 보고서 검수</strong>
<span id="progress"></span>
<select id="split"><option value="all">전체 split</option><option>train</option><option>validation</option><option>test</option></select>
<select id="filter"><option value="all">전체 상태</option><option value="pending">미검수</option><option value="approved">승인</option><option value="needs_review">수정 필요</option></select>
<button id="prev">이전</button><button id="next">다음</button>
<button id="export" class="primary">CSV 내보내기</button>
</header>
<main>
<div class="meta" id="meta"></div>
<div class="grid"><section class="panel"><h2>집계 입력과 근거</h2><div id="input"></div></section>
<section class="panel"><h2>보고서 정답 후보</h2><div id="machine"></div><textarea id="output"></textarea>
<div class="actions"><button id="approve" class="primary">승인 후 다음</button><button id="review" class="danger">수정 필요</button><button id="save">현재 내용 저장</button></div>
<p><label>메모 <input id="note" style="width:80%;padding:8px"></label></p></section></div>
</main>
<script>
const rows=__ROWS__;
const key="palda-decoder-review-v1";let saved=JSON.parse(localStorage.getItem(key)||"{}");let index=0;
const $=id=>document.getElementById(id);
function visible(){return rows.map((r,i)=>[r,i]).filter(([r])=>{
 const s=$("split").value,f=$("filter").value,v=saved[r.productGroupId]||{};
 return (s==="all"||r.split===s)&&(f==="all"||(v.status||"pending")===f);
});}
function store(){localStorage.setItem(key,JSON.stringify(saved));}
function evidence(groups){return groups.map(g=>'<div class="group"><b>'+g.displayTheme+'</b> <span class="tag">'+
 (g.sentiment||("비율 "+Math.round(g.ratio*100)+"%"))+'</span> <span class="tag">'+
 (g.mentionCount||g.primaryCount)+'건</span>'+g.evidence.map(e=>'<div class="evidence"><small>'+e.id+
 '</small><br>'+escapeHtml(e.text)+'</div>').join("")+'</div>').join("");}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function saveCurrent(){const pair=visible()[index];if(!pair)return;const r=pair[0];let parsed;
 try{parsed=JSON.parse($("output").value);$("machine").innerHTML='<span class="ok">JSON 형식 정상</span>';}
 catch(e){$("machine").innerHTML='<span class="warn">JSON 오류: '+escapeHtml(e.message)+'</span>';return false;}
 saved[r.productGroupId]={...(saved[r.productGroupId]||{}),reportOutput:parsed,note:$("note").value,updatedAt:new Date().toISOString()};store();return true;}
function render(){const list=visible();if(!list.length){$("meta").textContent="조건에 맞는 상품이 없습니다.";return}
 index=Math.max(0,Math.min(index,list.length-1));const [r,rawIndex]=list[index],v=saved[r.productGroupId]||{};
 const input=r.reportInput;$("progress").textContent=(index+1)+" / "+list.length+" · 전체 승인 "+
 Object.values(saved).filter(x=>x.status==="approved").length+" / "+rows.length;
 $("meta").innerHTML="<b>"+escapeHtml(input.category.path)+"</b><br>상품 "+r.productGroupId+
 " · "+r.split+" · Review "+input.sample.reviewSampleSize+"건 · Q&A "+input.sample.qnaSampleSize+"건";
 $("input").innerHTML="<h3>Review</h3>"+evidence(input.reviewGroups)+"<h3>Q&A</h3>"+evidence(input.qnaGroups);
 $("machine").innerHTML='모델 구조검증: <b class="'+(r.validationStatus==="valid"?"ok":"warn")+'">'+r.validationStatus+
 '</b> · 자동 보정 경고 '+(r.normalizationWarnings||[]).length+"개";
 $("output").value=JSON.stringify(v.reportOutput||r.reportOutputDraft,null,2);$("note").value=v.note||"";
}
$("prev").onclick=()=>{saveCurrent();index--;render()};$("next").onclick=()=>{saveCurrent();index++;render()};
$("save").onclick=()=>saveCurrent();$("approve").onclick=()=>{const pair=visible()[index];if(saveCurrent()&&pair){saved[pair[0].productGroupId].status="approved";store();render()}};
$("review").onclick=()=>{const pair=visible()[index];if(saveCurrent()&&pair){saved[pair[0].productGroupId].status="needs_review";store();render()}};
$("split").onchange=$("filter").onchange=()=>{index=0;render()};
$("output").oninput=()=>{clearTimeout(window.autoSave);window.autoSave=setTimeout(saveCurrent,700)};
$("note").oninput=()=>{clearTimeout(window.noteSave);window.noteSave=setTimeout(saveCurrent,700)};
$("export").onclick=()=>{saveCurrent();const fields=["productGroupId","split","reviewer_status","reviewer_note","report_output_json"];
 const csv=[fields.join(",")];for(const r of rows){const v=saved[r.productGroupId]||{};const vals=[r.productGroupId,r.split,v.status||"pending",v.note||"",JSON.stringify(v.reportOutput||r.reportOutputDraft)];
 csv.push(vals.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(","));}
 const blob=new Blob(["\\ufeff"+csv.join("\\r\\n")],{type:"text/csv;charset=utf-8"}),a=document.createElement("a");
 a.href=URL.createObjectURL(blob);a.download="palda-decoder-reviewed.csv";a.click();URL.revokeObjectURL(a.href);};
render();
</script></body></html>""".replace("__ROWS__", embedded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "products": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
