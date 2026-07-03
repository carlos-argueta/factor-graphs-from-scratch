"""fgslam — a minimal factor-graph / nonlinear-least-squares engine built from
scratch, module by module, alongside the course.

The point is not to compete with GTSAM but to *earn the intuition*: every piece we
later use in GTSAM, we first build here in plain numpy.

Built so far:
- M1: :mod:`fgslam.nls` — Gauss-Newton / Levenberg-Marquardt with a Huber robust kernel.

Coming:
- M3: variables, factors, the bipartite graph, batch solve.
- M4: SE(2)/SE(3) manifolds and on-manifold optimization.
- M5: variable elimination, incremental (iSAM-style) updates.
"""

from fgslam.nls import (
    HuberKernel,
    NLSResult,
    gauss_newton,
    levenberg_marquardt,
)

__all__ = [
    "HuberKernel",
    "NLSResult",
    "gauss_newton",
    "levenberg_marquardt",
]
