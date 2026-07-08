"""gaussian_filters — the reference Gaussian filters the course starts from.

These are the classic Probabilistic-Robotics estimators (Kalman / Extended
Kalman / Information filters) written in plain numpy. Module 0 uses them as the
"before" picture: it shows that one Extended Kalman Filter update is exactly one
Gauss-Newton step on a two-factor least-squares problem. From there the course
builds the from-scratch factor-graph engine in :mod:`factorgraph`.

Provenance: adapted from the `rse_prob_robotics` / `rse_gaussian_filters`
package. ``ekf.py`` retains its original GPLv3 header.
"""

from gaussian_filters.ekf import ExtendedKalmanFilter
from gaussian_filters.inf import InformationFilter

__all__ = ["ExtendedKalmanFilter", "InformationFilter"]
