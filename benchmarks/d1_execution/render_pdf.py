"""Render D1_EXECUTION_AUDIT_REPORT.md to PDF (python-markdown + WeasyPrint).

Usage:  python3 benchmarks/d1_execution/render_pdf.py <report.md> <out.pdf>

Requires system python with `markdown` and `weasyprint` installed (they are not
in .venv). Report-tooling only: touches no production code and reads only the
report Markdown.
"""
import re
import sys

import markdown
from weasyprint import HTML, CSS

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

# Split off the title (first H1) so it can be rendered as a cover block.
lines = text.split("\n")
title = lines[0].lstrip("# ").strip() if lines[0].startswith("# ") else "Report"
body_md = "\n".join(lines[1:]) if lines[0].startswith("# ") else text

html_body = markdown.markdown(
    body_md,
    extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc", "md_in_html"],
    extension_configs={"toc": {"title": None, "toc_depth": "2-2"}},
)

md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
                       extension_configs={"toc": {"toc_depth": "2-2"}})
html_body = md.convert(body_md)
toc = md.toc

CSS_TEXT = r"""
@page {
  size: A4;
  margin: 16mm 13mm 15mm 13mm;
  @bottom-center {
    content: "D1 DPE Synchronous Execution + Performance Audit  ·  page " counter(page) " / " counter(pages);
    font-family: "DejaVu Sans"; font-size: 7pt; color: #777;
  }
}
@page :first { @bottom-center { content: ""; } }
@page wide { size: A4 landscape; margin: 14mm 12mm; }

/* Tables with >= 6 columns get a landscape page of their own. */
.wide-table { page: wide; break-before: page; break-after: page; }
.wide-table > .wide-caption {
  font-size: 10.5pt; font-weight: bold; color: #0f1115;
  margin: 0 0 3mm 0; padding-bottom: 1.4mm; border-bottom: 1pt solid #1f6feb;
}
.wide-table table { font-size: 7.4pt; }

html { font-size: 9pt; }
body {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 8.6pt; line-height: 1.42; color: #16191d;
  hyphens: none;
}

/* ---------- cover ---------- */
.cover { page-break-after: always; padding-top: 42mm; }
.cover h1 {
  font-size: 25pt; line-height: 1.18; margin: 0 0 6mm 0;
  border: 0; padding: 0; color: #0f1115; letter-spacing: -0.4pt;
}
.cover .rule { height: 3px; background: #1f6feb; width: 46mm; margin: 0 0 7mm 0; }
.cover .meta { font-size: 10pt; color: #444; line-height: 1.85; }
.cover .meta b { color: #16191d; }
.cover .verdict {
  margin-top: 14mm; padding: 5mm 6mm;
  border: 1.4pt solid #1a7f37; border-left: 5pt solid #1a7f37;
  background: #f2fbf4; color: #10521f;
  font-size: 10.5pt; font-weight: bold; line-height: 1.5;
}
.cover .note { margin-top: 9mm; font-size: 8.2pt; color: #666; line-height: 1.6; }

/* ---------- table of contents ---------- */
.toc-page { page-break-after: always; }
.toc-page h2 { margin-top: 0; }
.toc ul { list-style: none; padding-left: 0; margin: 0; column-count: 2; column-gap: 9mm; }
.toc li { font-size: 8.4pt; margin: 0 0 1.7mm 0; break-inside: avoid; }
.toc a { color: #16191d; text-decoration: none; }

/* ---------- headings ---------- */
h1, h2, h3, h4 { font-family: "DejaVu Sans", sans-serif; color: #0f1115; break-after: avoid; }
h2 {
  font-size: 13pt; margin: 8mm 0 3mm 0; padding-bottom: 1.4mm;
  border-bottom: 1.1pt solid #1f6feb; break-before: page;
}
.toc-page + h2 { break-before: avoid; }
h3 { font-size: 10.2pt; margin: 5mm 0 2mm 0; color: #24303f; }
h4 { font-size: 9pt; margin: 4mm 0 1.5mm 0; }
p { margin: 0 0 2.4mm 0; orphans: 2; widows: 2; }

/* ---------- tables ---------- */
table {
  border-collapse: collapse; width: 100%; margin: 3mm 0 4mm 0;
  font-size: 6.9pt; line-height: 1.34; table-layout: auto;
}
th, td {
  border: 0.5pt solid #c9d1d9; padding: 1.5mm 1.8mm;
  text-align: left; vertical-align: top;
  word-break: normal; overflow-wrap: break-word;
}
th { background: #eef2f7; font-weight: bold; color: #0f1115; }
tr:nth-child(even) td { background: #fafbfc; }
tr { break-inside: avoid; }
thead { display: table-header-group; }

/* ---------- code ---------- */
pre {
  background: #f6f8fa; border: 0.5pt solid #d6dde5; border-left: 2.2pt solid #1f6feb;
  padding: 2.4mm 3mm; margin: 3mm 0; border-radius: 1.5pt;
  break-inside: avoid; white-space: pre; overflow: hidden;
}
pre code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 6.15pt;
  line-height: 1.32; color: #1c2128; background: none; padding: 0; border: 0;
}
code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 0.87em;
  background: #eef1f5; padding: 0.3mm 0.8mm; border-radius: 1.5pt; color: #0a3069;
}
th code, td code { font-size: 0.9em; padding: 0; background: none; color: #0a3069; }

/* ---------- blockquote ---------- */
blockquote {
  margin: 3mm 0; padding: 2.5mm 4mm;
  border-left: 3pt solid #d29922; background: #fffbf0; color: #4a3c10;
  break-inside: avoid;
}
blockquote h3 { margin-top: 0; color: #7a2f2f; }
blockquote p:last-child { margin-bottom: 0; }
blockquote pre { background: #fff; border-left-color: #d29922; }

/* ---------- lists / rules ---------- */
ul, ol { margin: 0 0 2.6mm 0; padding-left: 5.5mm; }
li { margin-bottom: 1.1mm; }
hr { border: 0; border-top: 0.5pt solid #dde3ea; margin: 5mm 0; }
strong { color: #0f1115; }

/* ---------- final verdict line ---------- */
.verdict-final {
  margin-top: 6mm; padding: 4mm 5mm; break-inside: avoid;
  border: 1.4pt solid #1a7f37; border-left: 5pt solid #1a7f37;
  background: #f2fbf4; color: #10521f; font-weight: bold; font-size: 10pt;
  text-align: center; letter-spacing: 0.2pt;
}
"""

VERDICT = "D1 DPE SYNCHRONOUS EXECUTION AUDIT: PASS — READY FOR DPE/HPE CONTRACT DECISION"

# Put wide tables (>= 6 columns) on their own landscape page, captioned with
# the section heading they belong to so they stay self-explanatory.
def _wrap_wide(html: str) -> str:
    out, pos = [], 0
    for m in re.finditer(r"<table>.*?</table>", html, re.S):
        tbl = m.group(0)
        ncols = len(re.findall(r"<th[ >]", tbl.split("</thead>")[0]))
        rows = re.findall(r"<tr>(.*?)</tr>", tbl, re.S)
        widest = max(
            (sum(len(re.sub(r"<[^>]+>", "", c).strip())
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S))
             for r in rows), default=0)
        # Landscape only for tables that are BOTH many-columned and text-heavy;
        # a 7-column table of short state names reads fine in portrait.
        if ncols < 6 or widest < 150:
            continue
        before = html[pos:m.start()]
        heads = re.findall(r"<h([234])[^>]*>(.*?)</h\1>", html[:m.start()], re.S)
        cap = re.sub(r"<[^>]+>", "", heads[-1][1]).strip() if heads else ""
        out.append(before)
        out.append(f'<div class="wide-table"><div class="wide-caption">{cap}</div>{m.group(0)}</div>')
        pos = m.end()
    out.append(html[pos:])
    return "".join(out)

html_body = _wrap_wide(html_body)

# Style the trailing verdict line as a callout rather than a bare paragraph.
html_body = html_body.replace(
    f"<p>{VERDICT}</p>", f'<div class="verdict-final">{VERDICT}</div>'
)

cover = f"""
<div class="cover">
  <h1>{title}</h1>
  <div class="rule"></div>
  <div class="meta">
    <b>Repository</b> &nbsp; depth_perception_engine v1.2.0<br/>
    <b>Branch</b> &nbsp; release/v1.2.0 &nbsp;·&nbsp; <b>HEAD</b> f4ce645<br/>
    <b>Scope</b> &nbsp; Audit + measurement only — production code unchanged<br/>
    <b>Baseline / final regression</b> &nbsp; 983 passed, 0 failed, 0 skipped
  </div>
  <div class="verdict">{VERDICT}</div>
  <div class="note">
    Clean rerun. Every finding and measurement was re-derived from the current working tree;
    no state or conclusion from the lost prior session was reused.<br/>
    Raw machine-readable results: <code>benchmarks/d1_execution/results/d1_execution_audit.json</code>
  </div>
</div>
<div class="toc-page">
  <h2 style="break-before:avoid">Contents</h2>
  {toc}
</div>
"""

full = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body>{cover}{html_body}</body></html>"
HTML(string=full, base_url=".").write_pdf(out, stylesheets=[CSS(string=CSS_TEXT)])
print("wrote", out)
