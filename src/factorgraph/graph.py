"""On-manifold factor graph and Gauss-Newton solver.

A **factor** defines only its whitened error — a vector in the tangent space. The
engine gets each factor's Jacobian by **central-differencing through the variable's
``retract``** (:func:`numerical_jacobian`), so the manifold structure enters in exactly
one place and every factor is correct-by-construction. This mirrors how GTSAM validates
its hand-derived Jacobians, and it lets the *same* engine run over any variable type
that exposes ``dof``, ``retract`` and ``local`` — :class:`~factorgraph.lie.Pose2`,
:class:`~factorgraph.lie.Pose3`, or a plain vector.

The solve is Module 1's Gauss-Newton with a single change: the update is
``x <- x boxplus delta`` (retract) instead of ``x <- x + delta``. Everything else —
residuals, normal equations, iteration — is unchanged.

The batch :meth:`FactorGraph.solve` here forms and factorises the *dense* system; Module 5
replaces that with sparse variable elimination and incremental (iSAM-style) updates.
"""

from __future__ import annotations

import numpy as np


def numerical_jacobian(error_fn, values, key, dof, eps=1e-6):
    """d(error)/d(tangent perturbation of values[key]), via central differences
    taken *through* the variable's retract -- i.e. the derivative on the manifold."""
    base = values[key]
    cols = []
    for i in range(dof):
        d = np.zeros(dof); d[i] = eps
        vp, vm = dict(values), dict(values)
        vp[key] = base.retract(d)
        vm[key] = base.retract(-d)
        cols.append((error_fn(vp) - error_fn(vm)) / (2 * eps))
    return np.stack(cols, axis=1)


def _W(sigma, dim):
    """Whitening matrix W with W^T W = Sigma^{-1} for a diagonal noise model."""
    sigma = np.asarray(sigma, float)
    if sigma.ndim == 0:
        return np.eye(dim) / sigma
    return np.diag(1.0 / sigma)


class Factor:
    """Base class: define `error(values)` (whitened, tangent-space); the engine
    linearises numerically through each variable's retract."""
    keys, dim = (), 0

    def error(self, values):
        raise NotImplementedError

    def linearize(self, values):
        r = self.error(values)
        blocks = {k: numerical_jacobian(self.error, values, k, values[k].dof)
                  for k in self.keys}
        return r, blocks


class PriorFactor(Factor):
    """Pull a pose toward a measured absolute pose: r = W * log(measured^{-1} X)."""
    def __init__(self, key, measured, sigma):
        self.keys = (key,)
        self.measured = measured
        self.dim = measured.dof
        self.W = _W(sigma, self.dim)

    def error(self, values):
        return self.W @ self.measured.local(values[self.keys[0]])


class BetweenFactor(Factor):
    """Constrain the relative pose X_i^{-1} X_j to a measured value:
       r = W * log( measured^{-1} (X_i^{-1} X_j) )."""
    def __init__(self, k1, k2, measured, sigma):
        self.keys = (k1, k2)
        self.measured = measured
        self.dim = measured.dof
        self.W = _W(sigma, self.dim)

    def error(self, values):
        predicted = values[self.keys[0]].between(values[self.keys[1]])   # X_i^{-1} X_j
        return self.W @ self.measured.local(predicted)


class Ordering:
    """Assign each variable key a contiguous slice of the stacked *tangent* vector."""
    def __init__(self, dofs):                 # dofs: {key: tangent dimension}
        self.slices, i = {}, 0
        for k, d in dofs.items():
            self.slices[k] = slice(i, i + d)
            i += d
        self.n = i


class FactorGraph:
    def __init__(self):
        self.factors = []

    def add(self, factor):
        self.factors.append(factor); return self

    def linearize(self, values, ordering):
        m = sum(f.dim for f in self.factors)
        J = np.zeros((m, ordering.n)); r = np.zeros(m); row = 0
        for f in self.factors:
            r_f, blocks = f.linearize(values)
            r[row:row + f.dim] = r_f
            for key, Jb in blocks.items():
                J[row:row + f.dim, ordering.slices[key]] = Jb
            row += f.dim
        return r, J

    def error(self, values):                  # total 1/2||r||^2, for convergence tracking
        m = sum(f.dim for f in self.factors); r = np.zeros(m); row = 0
        for f in self.factors:
            r[row:row + f.dim] = f.error(values); row += f.dim
        return 0.5 * float(r @ r)

    def solve(self, values, max_iters=25, tol=1e-10):
        """On-manifold Gauss-Newton: solve for a tangent step, retract, repeat."""
        values = dict(values)
        ordering = Ordering({k: v.dof for k, v in values.items()})
        history = [self.error(values)]
        for _ in range(max_iters):
            r, J = self.linearize(values, ordering)
            delta, *_ = np.linalg.lstsq(J, -r, rcond=None)     # J delta = -r
            for k in values:                                   # x <- x boxplus delta
                values[k] = values[k].retract(delta[ordering.slices[k]])
            history.append(self.error(values))
            if np.linalg.norm(delta) < tol:
                break
        return values, history
