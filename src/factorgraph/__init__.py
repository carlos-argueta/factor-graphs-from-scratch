"""factorgraph — a minimal factor-graph / nonlinear-least-squares engine built from
scratch, module by module, alongside the course.

The point is not to compete with GTSAM but to *earn the intuition*: every piece we
later use in GTSAM, we first build here in plain numpy.

Built so far:
- M1: :mod:`factorgraph.nls` — Gauss-Newton / Levenberg-Marquardt with a Huber robust kernel.
- M4: :mod:`factorgraph.lie` — SO(2)/SE(2)/SO(3)/SE(3) groups and the ``Pose2``/``Pose3`` types;
  :mod:`factorgraph.graph` — factors, the factor graph, and the on-manifold Gauss-Newton solve.

Coming:
- M5: variable elimination, the Bayes tree, incremental (iSAM-style) updates.
"""

from factorgraph.nls import (
    HuberKernel,
    NLSResult,
    gauss_newton,
    levenberg_marquardt,
)
from factorgraph.lie import (
    Pose2,
    Pose3,
    wrap_to_pi,
    so2_exp, so2_log, so2_boxplus, so2_boxminus,
    se2_exp, se2_log, se2_adjoint,
    hat, vee,
    so3_exp, so3_log,
    se3_exp, se3_log,
)
from factorgraph.graph import (
    numerical_jacobian,
    Factor,
    PriorFactor,
    BetweenFactor,
    Ordering,
    FactorGraph,
)

__all__ = [
    # nls (M1)
    "HuberKernel",
    "NLSResult",
    "gauss_newton",
    "levenberg_marquardt",
    # lie groups + pose types (M4)
    "Pose2",
    "Pose3",
    "wrap_to_pi",
    "so2_exp", "so2_log", "so2_boxplus", "so2_boxminus",
    "se2_exp", "se2_log", "se2_adjoint",
    "hat", "vee",
    "so3_exp", "so3_log",
    "se3_exp", "se3_log",
    # factor graph engine (M4)
    "numerical_jacobian",
    "Factor",
    "PriorFactor",
    "BetweenFactor",
    "Ordering",
    "FactorGraph",
]
