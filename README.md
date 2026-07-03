# factor-graphs-from-scratch

A hands-on course that takes you from the classic **Gaussian filters** (Kalman / EKF /
Information filter, bundled in [`src/gaussian_filters/`](src/gaussian_filters)) to
**factor-graph** state estimation, SLAM, calibration, and the production **GTSAM/iSAM2** stack.

We build a minimal factor-graph engine from scratch in numpy (`src/fgslam/`) to earn the
intuition, then map every concept onto GTSAM.

## Run it in the browser (Google Colab)

Every lesson runs on Colab with no local setup — open the notebook straight from GitHub.
Each notebook carries an "Open in Colab" badge in its top cell:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlos-argueta/factor-graphs-from-scratch/blob/main/modules/00-ekf-to-least-squares/lesson.ipynb)
```

The first code cell clones this repo on Colab so the `gaussian_filters` and `fgslam`
packages import cleanly. Or, from any notebook:
`!pip install git+https://github.com/carlos-argueta/factor-graphs-from-scratch.git`.

## Setup (local)

This course has its own [`pixi`](https://pixi.sh) environment:

```bash
cd factor-graphs-from-scratch
pixi install          # numpy, scipy, sympy, matplotlib, JupyterLab, GTSAM
pixi run lab          # open JupyterLab
pixi run check        # execute every lesson notebook headless (smoke test)
```

Not using pixi? The project is also pip-installable (`pip install -e .`), which puts the
`gaussian_filters` and `fgslam` packages on your path.

## Printing a lesson (PDF)

Prefer to study on paper? Export any notebook to a print-ready PDF — cells are
executed, the math is typeset (via MathJax), and long code/output lines wrap so
nothing is clipped at the page edge:

```bash
pixi run pdf modules/00-ekf-to-least-squares/lesson.ipynb   # a notebook
pixi run pdf modules/00-ekf-to-least-squares                # a dir -> lesson.ipynb
```

- **Input**: one notebook path, or a module directory (uses its `lesson.ipynb`).
- **Output**: a `.pdf` next to the notebook (same basename), US-letter.
- **Options** (see [`tools/nb2pdf.py`](tools/nb2pdf.py)): `--no-execute` (export as-is,
  skip re-running cells), `--output PATH`, `--keep-html`, `--timeout SECS`.
- **Needs**: a Chromium-family browser on `PATH` (`google-chrome`/`chromium`) and,
  on first render, network access (MathJax is loaded from a CDN).

> Why not `nbconvert --to pdf`? That needs a full LaTeX stack (xelatex + luaotfload +
> many TeX packages) which isn't complete on this machine. `pixi run pdf` takes a
> TeX-free path instead: execute → self-contained HTML → headless-Chrome print.

## Layout

```
.
  pixi.toml        the course environment
  pyproject.toml   pip-installable packaging (gaussian_filters + fgslam)
  tools/nb2pdf.py  export a notebook to a print-ready PDF (`pixi run pdf`)
  src/gaussian_filters/  the reference Kalman/EKF/Information filters (course starting point)
  src/fgslam/      the factor-graph engine we build, module by module
  modules/
    00-ekf-to-least-squares/   lesson.ipynb · exercises.ipynb · solutions/
    ...                        (more modules added as the course grows)
```

## License

- **Code** (`src/`, `tools/`, and notebook code cells): [MIT](LICENSE).
- **Course content** (notebook prose, math, figures, exercises):
  [CC BY 4.0](LICENSE-CONTENT.md).
