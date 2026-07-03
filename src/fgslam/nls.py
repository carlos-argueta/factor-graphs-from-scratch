"""Nonlinear least squares — the engine under every estimator in this course.

We minimise a sum of squared (optionally robustified) residuals

    C(x) = 1/2  sum_i  rho( || r_i(x) ||^2 )

where ``r(x)`` is a stacked residual vector and ``J = dr/dx`` its Jacobian. The
two workhorses:

* :func:`gauss_newton` — solve ``(J^T J) dx = -J^T r`` and step ``x <- x + dx``.
* :func:`levenberg_marquardt` — damped GN, ``(J^T J + lambda diag) dx = -J^T r``,
  trust-region-style adaptation of ``lambda``. Robust to a poor initial guess.

Robustness uses iteratively-reweighted least squares (IRLS): a :class:`HuberKernel`
down-weights rows whose residual is large, so a few outliers can't dominate the fit.

This is deliberately small and readable. The whole of Kalman filtering is *one*
Gauss-Newton iteration of a two-factor problem (see Module 0); a batch smoother is
many iterations over many factors. Everything else in the course is this idea, scaled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# A residual function maps the parameter vector x -> (r, J):
#   r : (m,)   stacked residuals
#   J : (m, n) Jacobian dr/dx
ResidualFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]


# --------------------------------------------------------------------------- #
# Robust kernel
# --------------------------------------------------------------------------- #

@dataclass
class HuberKernel:
    """Huber M-estimator, applied per scalar residual via IRLS weights.

    Quadratic for small residuals (``|r| <= delta``) and linear beyond, so a
    gross outlier contributes ~linearly instead of quadratically. We realise it
    by scaling each residual and its Jacobian row by ``w = sqrt(weight)`` so the
    ordinary normal equations on the scaled system implement the reweighting
    (Triggs' "square-rooting the kernel").
    """

    delta: float = 1.0

    def weights(self, r: np.ndarray) -> np.ndarray:
        """Per-residual IRLS weight ``w`` such that the scaled row is ``w * r``."""
        a = np.abs(np.asarray(r, float))
        # weight on the squared cost is 1 inside the band, delta/|r| outside;
        # the residual/Jacobian get sqrt of that.
        w = np.ones_like(a)
        out = a > self.delta
        w[out] = np.sqrt(self.delta / a[out])
        return w


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass
class NLSResult:
    x: np.ndarray                       # final parameters
    cost: float                         # final 1/2 ||r||^2 (robustified)
    iterations: int
    converged: bool
    cost_history: list = field(default_factory=list)   # cost per iteration (incl. start)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "converged" if self.converged else "stopped"
        return (f"NLSResult({status} in {self.iterations} it, "
                f"cost={self.cost:.6g}, x={np.array2string(self.x, precision=4)})")


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _eval(residual_fn: ResidualFn, x: np.ndarray, kernel: HuberKernel | None):
    """Return (r, J, cost) with the robust kernel folded in by row-scaling."""
    r, J = residual_fn(x)
    r = np.asarray(r, float).reshape(-1)
    J = np.asarray(J, float).reshape(r.shape[0], -1)
    if kernel is not None:
        w = kernel.weights(r)
        r = w * r
        J = w[:, None] * J
    cost = 0.5 * float(r @ r)
    return r, J, cost


# --------------------------------------------------------------------------- #
# Gauss-Newton
# --------------------------------------------------------------------------- #

def gauss_newton(residual_fn: ResidualFn, x0, *, max_iters: int = 50,
                 tol: float = 1e-9, kernel: HuberKernel | None = None,
                 verbose: bool = False) -> NLSResult:
    """Minimise ``1/2||r(x)||^2`` by Gauss-Newton.

    Each step solves the normal equations ``(J^T J) dx = -J^T r`` (via a least
    squares solve for numerical stability) and applies ``x <- x + dx``. Stops
    when the step or the relative cost change falls below ``tol``.
    """
    x = np.array(x0, float).reshape(-1)
    r, J, cost = _eval(residual_fn, x, kernel)
    history = [cost]

    converged = False
    it = 0
    for it in range(1, max_iters + 1):
        # Solve J dx = -r in the least-squares sense (equivalent to the normal
        # equations but better conditioned than forming J^T J explicitly).
        dx, *_ = np.linalg.lstsq(J, -r, rcond=None)
        x_new = x + dx
        r, J, cost_new = _eval(residual_fn, x_new, kernel)

        if verbose:
            print(f"  GN it {it:2d}: cost {cost:.6g} -> {cost_new:.6g}, "
                  f"|dx|={np.linalg.norm(dx):.3e}")

        x = x_new
        rel = abs(cost - cost_new) / max(cost, 1e-12)
        cost = cost_new
        history.append(cost)
        if np.linalg.norm(dx) < tol or rel < tol:
            converged = True
            break

    return NLSResult(x=x, cost=cost, iterations=it, converged=converged,
                     cost_history=history)


# --------------------------------------------------------------------------- #
# Levenberg-Marquardt
# --------------------------------------------------------------------------- #

def levenberg_marquardt(residual_fn: ResidualFn, x0, *, max_iters: int = 100,
                        tol: float = 1e-9, lambda0: float = 1e-3,
                        lambda_up: float = 10.0, lambda_down: float = 10.0,
                        kernel: HuberKernel | None = None,
                        verbose: bool = False) -> NLSResult:
    """Minimise ``1/2||r(x)||^2`` by Levenberg-Marquardt (damped Gauss-Newton).

    Solves ``(H + lambda * diag(H)) dx = -g`` with ``H = J^T J`` and ``g = J^T r``.
    A step that lowers the cost is accepted and the damping is *relaxed*
    (trust region grows, -> Gauss-Newton); a step that doesn't is rejected and the
    damping is *raised* (-> gradient descent). Far more robust to a bad ``x0`` than GN.
    """
    x = np.array(x0, float).reshape(-1)
    r, J, cost = _eval(residual_fn, x, kernel)
    history = [cost]
    lam = lambda0

    converged = False
    it = 0
    for it in range(1, max_iters + 1):
        H = J.T @ J
        g = J.T @ r
        diag = np.diag(np.diag(H))

        # Try a step; shrink the trust region until it helps or we give up.
        stepped = False
        for _ in range(30):
            try:
                dx = np.linalg.solve(H + lam * diag, -g)
            except np.linalg.LinAlgError:
                lam *= lambda_up
                continue
            x_new = x + dx
            _, _, cost_new = _eval(residual_fn, x_new, kernel)
            if cost_new < cost:
                # Accept and relax the damping.
                x = x_new
                r, J, _ = _eval(residual_fn, x, kernel)
                lam = max(lam / lambda_down, 1e-12)
                rel = abs(cost - cost_new) / max(cost, 1e-12)
                cost = cost_new
                stepped = True
                if verbose:
                    print(f"  LM it {it:2d}: cost -> {cost:.6g}, lam={lam:.2e}, "
                          f"|dx|={np.linalg.norm(dx):.3e}")
                if np.linalg.norm(dx) < tol or rel < tol:
                    converged = True
                break
            # Reject and tighten the damping.
            lam *= lambda_up
        else:
            stepped = False

        history.append(cost)
        if converged or not stepped:
            break

    return NLSResult(x=x, cost=cost, iterations=it, converged=converged,
                     cost_history=history)
