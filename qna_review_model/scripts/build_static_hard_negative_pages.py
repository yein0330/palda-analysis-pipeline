#!/usr/bin/env python
"""Build JavaScript-free HTML pages for environments that block scripts."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=500)
    return parser.parse_args()


def e(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def labels(row: dict) -> str:
    value = row["labels"]
    if row["sourceType"] == "review":
        found = [
            f'<span class="tag {e(item["sentiment"])}">{e(item["themeId"])} · {e(item["sentiment"])}</span>'
            for item in value.get("reviewLabels", [])
        ]
        return "".join(found) or '<span class="empty">예측 테마 없음</span>'
    primary = value.get("qnaPrimaryTheme") or {}
    secondary = [item["themeId"] for item in value.get("qnaSecondaryThemes", [])]
    return (
        f'<div><strong>대표:</strong> {e(primary.get("themeId") or "없음")}</div>'
        f'<div><strong>보조:</strong> {e(", ".join(secondary) or "없음")}</div>'
    )


def card(row: dict, number: int) -> str:
    issue = (row.get("hardNegative") or {}).get("issue", "")
    return f'''<article class="card">
<div class="head"><strong>#{number:,}</strong><span>{e(row["sourceType"]).upper()}</span><span>{e(issue)}</span><span>confidence {e(row["annotation"]["modelConfidence"])}</span></div>
<div class="path">{e(row["category"]["path"])}</div>
<div class="id">{e(row["itemId"])}</div>
<div class="text">{e(row["text"])}</div>
<div class="labels">{labels(row)}</div>
</article>'''


STYLE = '''
body{margin:0;background:#f4f6fa;color:#182230;font:14px/1.55 system-ui,sans-serif}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d9e0ea;padding:14px 20px;z-index:2}
main{max-width:1050px;margin:18px auto;padding:0 16px}.nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
a{color:#245dd8}.card{background:#fff;border:1px solid #d9e0ea;border-radius:12px;padding:18px;margin:12px 0}
.head{display:flex;gap:7px;flex-wrap:wrap}.head span,.tag{border-radius:99px;padding:4px 8px;background:#edf2fb}
.path{font-weight:700;margin-top:10px}.id{color:#667085;font-size:12px}.text{white-space:pre-wrap;font-size:17px;background:#f8fafc;padding:14px;border-radius:8px;margin:12px 0}
.labels{display:flex;gap:7px;flex-wrap:wrap}.tag.positive{background:#e9f8ef;color:#087443}.tag.negative{background:#fff0ee;color:#b42318}.tag.neutral{background:#f1f3f6}.empty{color:#667085}
'''


def main() -> None:
    args = arguments()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = max(1, math.ceil(len(rows) / args.page_size))
    links = " ".join(f'<a href="page-{index:02d}.html">{index}페이지</a>' for index in range(1, pages + 1))
    for page in range(pages):
        start = page * args.page_size
        values = rows[start:start + args.page_size]
        body = "\n".join(card(row, start + offset + 1) for offset, row in enumerate(values))
        document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PALDA hard-negative {page + 1}/{pages}</title><style>{STYLE}</style></head><body>
<header><strong>PALDA hard-negative · {page + 1}/{pages}페이지 · {start + 1:,}–{start + len(values):,} / {len(rows):,}건</strong><nav class="nav">{links}</nav></header>
<main>{body}</main></body></html>'''
        (args.output_dir / f"page-{page + 1:02d}.html").write_text(document, encoding="utf-8")
    index = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>PALDA hard-negative</title><style>{STYLE}</style></head><body><header><strong>PALDA hard-negative 정적 보기 · {len(rows):,}건</strong></header><main><div class="card"><p>JavaScript 없이 데이터 카드가 표시되는 읽기 전용 페이지입니다.</p><nav class="nav">{links}</nav></div></main></body></html>'''
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps({"records": len(rows), "pages": pages, "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
