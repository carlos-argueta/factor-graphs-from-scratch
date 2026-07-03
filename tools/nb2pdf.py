#!/usr/bin/env python3
"""Export a lesson notebook to a print-ready PDF.

Why this exists
---------------
The obvious route, ``jupyter nbconvert --to pdf``, needs a full LaTeX stack
(xelatex + a working luaotfload + a pile of TeX packages). That toolchain is
incomplete on this machine, so instead we take a TeX-free path that renders the
LaTeX math with MathJax and never truncates code or output:

    notebook  --(nbconvert, executed)-->  self-contained HTML
              --(inject a print stylesheet that *wraps* long lines)-->  HTML'
              --(headless Chrome --print-to-pdf)-->  PDF

The injected stylesheet forces every code/output block to wrap, so nothing runs
off the right page edge (the failure mode of a naive HTML->PDF print).

Usage
-----
Run it inside the course pixi env so ``jupyter`` is on PATH::

    pixi run pdf modules/00-ekf-to-least-squares/lesson.ipynb
    pixi run pdf modules/00-ekf-to-least-squares         # dir -> lesson.ipynb
    pixi run pdf modules/01-nonlinear-least-squares/exercises.ipynb

Options::

    --no-execute     convert the notebook as-is (do not re-run its cells)
    --output PATH    write the PDF here (default: alongside the notebook,
                     same basename with a .pdf suffix)
    --keep-html      keep the intermediate HTML next to the PDF (for debugging)
    --timeout SECS   per-cell execution timeout (default 300)

Inputs
------
* One notebook path, OR a module directory (then ``lesson.ipynb`` inside it).
* A Chromium-family browser on PATH (google-chrome / chromium / ...).
* Network access on first render (MathJax is pulled from a CDN).

Output
------
* A PDF next to the notebook (or at ``--output``). US-letter, math typeset,
  all cells executed unless ``--no-execute``, no truncated lines.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Browsers we know how to drive, in preference order.
CHROME_CANDIDATES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
]

# Print stylesheet: wrap every code/output container so long lines never get
# clipped at the page edge. Covers both the JupyterLab and classic HTML themes.
WRAP_CSS = """
<style>
pre, code, .highlight, .highlight pre, .input_area, .output_area,
.jp-RenderedText, .jp-RenderedText pre, .jp-OutputArea-output,
.jp-OutputArea-output pre, .CodeMirror-line, .cm-editor, .jp-InputArea-editor {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
  overflow: visible !important;
}
.jp-Cell, .jp-OutputArea, .jp-Notebook, body { overflow: visible !important; }
@page { margin: 14mm; }
</style>
"""


def find_chrome() -> str:
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    sys.exit(
        "error: no Chromium-family browser found on PATH (looked for: "
        + ", ".join(CHROME_CANDIDATES)
        + ").\nInstall google-chrome or chromium, then re-run."
    )


def resolve_notebook(arg: str) -> Path:
    p = Path(arg).resolve()
    if p.is_dir():
        p = p / "lesson.ipynb"
    if p.suffix != ".ipynb" or not p.is_file():
        sys.exit(f"error: not a notebook: {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a lesson notebook to a print-ready PDF.")
    ap.add_argument("notebook", help="path to a .ipynb, or a module dir (uses lesson.ipynb)")
    ap.add_argument("--output", help="output PDF path (default: notebook with .pdf suffix)")
    ap.add_argument("--no-execute", action="store_true", help="do not re-run cells before export")
    ap.add_argument("--keep-html", action="store_true", help="keep the intermediate HTML next to the PDF")
    ap.add_argument("--timeout", type=int, default=300, help="per-cell execution timeout in seconds")
    args = ap.parse_args()

    nb = resolve_notebook(args.notebook)
    out_pdf = Path(args.output).resolve() if args.output else nb.with_suffix(".pdf")
    chrome = find_chrome()

    with tempfile.TemporaryDirectory(prefix="nb2pdf_") as td:
        tmp = Path(td)
        html = tmp / (nb.stem + ".html")

        # 1) notebook -> self-contained HTML (optionally executed).
        cmd = [
            "jupyter", "nbconvert", str(nb),
            "--to", "html",
            "--embed-images",
            "--output-dir", str(tmp),
            "--output", html.name,
        ]
        if not args.no_execute:
            cmd += ["--execute", f"--ExecutePreprocessor.timeout={args.timeout}"]
        print("[nb2pdf] converting:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # 2) inject the wrap stylesheet so nothing is clipped when printed.
        doc = html.read_text(encoding="utf-8")
        if "</head>" in doc:
            doc = doc.replace("</head>", WRAP_CSS + "</head>", 1)
        else:
            doc = WRAP_CSS + doc
        html.write_text(doc, encoding="utf-8")

        # 3) headless Chrome -> PDF. virtual-time-budget lets MathJax typeset.
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        chrome_cmd = [
            chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=25000",
            f"--print-to-pdf={out_pdf}",
            html.as_uri(),
        ]
        print("[nb2pdf] printing PDF via", Path(chrome).name)
        subprocess.run(chrome_cmd, check=True)

        if args.keep_html:
            kept = out_pdf.with_suffix(".html")
            shutil.copy(html, kept)
            print("[nb2pdf] kept HTML:", kept)

    if not out_pdf.is_file():
        sys.exit("error: Chrome did not produce a PDF (see output above).")
    print(f"[nb2pdf] wrote {out_pdf} ({out_pdf.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
