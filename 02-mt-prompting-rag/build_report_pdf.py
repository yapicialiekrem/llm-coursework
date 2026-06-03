"""Render report.md -> report.pdf with weasyprint + a print-friendly stylesheet.

Run: `python build_report_pdf.py`
"""
from __future__ import annotations

import os
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

CSS_STR = """
@page {
    size: A4;
    margin: 2.2cm 2cm 2.4cm 2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #222;
}
h1 { font-size: 22pt; color: #1F4E79; margin-top: 0.3em; }
h2 { font-size: 15pt; color: #1F4E79; margin-top: 1.4em; border-bottom: 1px solid #d0dce8; padding-bottom: 0.2em; }
h3 { font-size: 12pt; color: #2C5C8B; margin-top: 1.1em; }
h4 { font-size: 11pt; color: #2C5C8B; }
p  { margin: 0.55em 0; }
code { font-family: 'JetBrains Mono', 'Menlo', monospace; font-size: 9.5pt;
       background: #f3f5f9; padding: 1px 4px; border-radius: 3px; }
pre  { background: #f6f8fb; padding: 8px 10px; border-left: 3px solid #1F4E79;
       border-radius: 3px; overflow-x: auto; font-size: 9pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 0.8em 0; width: 100%; font-size: 9.5pt; }
th, td { border: 1px solid #d0dce8; padding: 5px 8px; text-align: left; }
th { background: #eef2f8; color: #1F4E79; }
blockquote { border-left: 3px solid #d0aa55; background: #fffbf0;
             padding: 6px 12px; margin: 0.8em 0; color: #443; }
img { max-width: 100%; }
hr { border: none; border-top: 1px solid #d0dce8; margin: 1.5em 0; }
"""


def main():
    md_path = Path("report.md")
    out_path = Path("report.pdf")
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"
    HTML(string=html, base_url=str(md_path.parent.resolve())).write_pdf(
        str(out_path),
        stylesheets=[CSS(string=CSS_STR)],
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
