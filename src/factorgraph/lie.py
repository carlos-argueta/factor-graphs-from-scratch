"""Lie groups for on-manifold estimation — SO(2), SE(2), SO(3), SE(3).

Rotations and rigid-body poses do not live in a flat vector space: a heading wraps
at :math:`\\pm\\pi`, a 3-D orientation lives on a curved surface. Adding corrections
as numbers is silently wrong near the wrap. The fix (Module 4) is to keep each
unknown *on its manifold* and optimise in the flat tangent space that touches it,
via two operators:

    X boxplus xi = X exp(xi)          (retract: nudge along the manifold)
    Y boxminus X = log(X^{-1} Y)      (local:   the tangent step from X to Y)

Each group provides ``exp``/``log`` (the bridge to its tangent, the Lie algebra) and
a small pose type (:class:`Pose2`, :class:`Pose3`) that carries exactly the operations
a factor needs — ``compose``, ``inverse``, ``between``, ``retract``, ``local`` — plus a
``dof`` attribute so the solver knows how many tangent dimensions the variable spans.

The pose types share one interface, so the same :mod:`factorgraph.graph` engine runs
unchanged in the plane and in space.
"""

from __future__ import annotations

import numpy as np

# =========================================================================== #
# SO(2) — planar rotations
# =========================================================================== #

def wrap_to_pi(a):
    """Fold an angle back into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def so2_exp(theta):
    """Tangent angle -> rotation matrix (wraps the line onto the circle)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def so2_log(R):
    """Rotation matrix -> tangent angle in (-pi, pi] (atan2 = the wrap, done once)."""
    return np.arctan2(R[1, 0], R[0, 0])


def so2_boxplus(R, delta):
    """Retract: nudge R by a tangent angle, stay on the circle."""
    return R @ so2_exp(delta)


def so2_boxminus(S, R):
    """Local: the shortest tangent angle from R to S."""
    return so2_log(R.T @ S)


# =========================================================================== #
# SE(2) — planar rigid-body poses
# =========================================================================== #

def _V_se2(w):
    """The SE(2) left-Jacobian block; -> I as w -> 0 (Taylor guard near 0)."""
    if abs(w) < 1e-8:
        A = 1.0 - w * w / 6.0            # sin(w)/w
        B = w / 2.0 - w**3 / 24.0        # (1 - cos w)/w
    else:
        A = np.sin(w) / w
        B = (1.0 - np.cos(w)) / w
    return np.array([[A, -B], [B, A]])


def se2_exp(xi):
    """Twist (vx, vy, omega) -> SE(2) matrix."""
    v, w = np.asarray(xi[:2], float), float(xi[2])
    T = np.eye(3)
    T[:2, :2] = so2_exp(w)
    T[:2, 2] = _V_se2(w) @ v
    return T


def se2_log(T):
    """SE(2) matrix -> twist (vx, vy, omega)."""
    w = so2_log(T[:2, :2])
    v = np.linalg.solve(_V_se2(w), T[:2, 2])
    return np.array([v[0], v[1], w])


def se2_adjoint(T):
    """Ad_T for SE(2): maps a twist through a change of frame,
       exp(Ad_T xi) == T exp(xi) T^{-1}."""
    R, t = T[:2, :2], T[:2, 2]
    Ad = np.eye(3)
    Ad[:2, :2] = R
    Ad[0, 2] =  t[1]      # [ R | [ y ; -x] ]
    Ad[1, 2] = -t[0]
    return Ad


class Pose2:
    """A planar rigid pose (x, y, theta) stored as a 3x3 homogeneous matrix."""
    dof = 3

    def __init__(self, x=0.0, y=0.0, theta=0.0, T=None):
        if T is not None:
            self.T = np.asarray(T, float)
        else:
            self.T = np.eye(3)
            self.T[:2, :2] = so2_exp(theta)
            self.T[:2, 2] = [x, y]

    # readable accessors
    @property
    def x(self):     return self.T[0, 2]
    @property
    def y(self):     return self.T[1, 2]
    @property
    def theta(self): return so2_log(self.T[:2, :2])
    def xytheta(self):  return np.array([self.x, self.y, self.theta])

    # group operations
    def compose(self, other):  return Pose2(T=self.T @ other.T)
    def inverse(self):         return Pose2(T=np.linalg.inv(self.T))
    def between(self, other):  return Pose2(T=np.linalg.inv(self.T) @ other.T)  # this^{-1} * other

    # manifold operations  (right-perturbation convention)
    def retract(self, xi):     return Pose2(T=self.T @ se2_exp(xi))             # X boxplus xi
    def local(self, other):    return se2_log(np.linalg.inv(self.T) @ other.T)  # other boxminus X

    def __repr__(self):
        return f"Pose2(x={self.x:.3f}, y={self.y:.3f}, theta={np.rad2deg(self.theta):.1f}deg)"


# =========================================================================== #
# SO(3) — spatial rotations
# =========================================================================== #

def hat(w):
    """so(3): map a 3-vector to its skew-symmetric matrix ( [w]_x u = w x u )."""
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(W):
    """Inverse of :func:`hat`: skew-symmetric matrix -> 3-vector."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def so3_exp(phi):
    """Rotation-vector -> rotation matrix (Rodrigues)."""
    phi = np.asarray(phi, float)
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K                       # Taylor near 0
    return np.eye(3) + (np.sin(th) / th) * K + ((1 - np.cos(th)) / th**2) * K @ K


def so3_log(R):
    """Rotation matrix -> rotation-vector (axis * angle)."""
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-8:
        return 0.5 * vee(R - R.T)                                # Taylor near 0
    return (th / (2.0 * np.sin(th))) * vee(R - R.T)


# =========================================================================== #
# SE(3) — full 6-DOF poses
# =========================================================================== #

def _V_se3(phi):
    """SE(3) left-Jacobian block; -> I as ||phi|| -> 0."""
    phi = np.asarray(phi, float)
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-8:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * K @ K
    A = (1 - np.cos(th)) / th**2
    B = (th - np.sin(th)) / th**3
    return np.eye(3) + A * K + B * K @ K


def se3_exp(xi):
    """Twist (rho[3], phi[3]) -> 4x4 SE(3) matrix."""
    rho, phi = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = _V_se3(phi) @ rho
    return T


def se3_log(T):
    """4x4 SE(3) matrix -> twist (rho[3], phi[3])."""
    phi = so3_log(T[:3, :3])
    rho = np.linalg.solve(_V_se3(phi), T[:3, 3])
    return np.concatenate([rho, phi])


class Pose3:
    """A spatial rigid pose stored as a 4x4 homogeneous matrix."""
    dof = 6

    def __init__(self, T=None, R=None, t=None):
        if T is not None:
            self.T = np.asarray(T, float)
        else:
            self.T = np.eye(4)
            if R is not None: self.T[:3, :3] = R
            if t is not None: self.T[:3, 3] = t

    @property
    def t(self): return self.T[:3, 3]
    @property
    def R(self): return self.T[:3, :3]

    def compose(self, other): return Pose3(T=self.T @ other.T)
    def inverse(self):        return Pose3(T=np.linalg.inv(self.T))
    def between(self, other): return Pose3(T=np.linalg.inv(self.T) @ other.T)
    def retract(self, xi):    return Pose3(T=self.T @ se3_exp(xi))
    def local(self, other):   return se3_log(np.linalg.inv(self.T) @ other.T)

    def __repr__(self):
        return f"Pose3(t={np.round(self.t, 3)}, rot={np.round(so3_log(self.R), 3)})"
