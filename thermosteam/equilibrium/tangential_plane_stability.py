# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2025, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import numpy as np
import thermosteam as tmo
from scipy.optimize import minimize
from typing import NamedTuple
from itertools import combinations

__all__ = ('TangentPlaneStabilityAnalysis',)

class StabilityReport(NamedTuple):
    unstable: bool
    candidate: np.ndarray
    tpd: float
    early_exit: bool = False

def edge_points_simplex_masked(z: np.ndarray,
                               points_per_edge: int = 12,
                               epsilon: float = 1e-3,
                               min_active: int = 2) -> np.ndarray:
    """
    Sample along edges of the simplex, but only for components with z_i > mean(z).
    Inactive components keep their original composition (scaled so total = 1).
    Always uses Chebyshev spacing along edges (no pure vertices).

    Parameters
    ----------
    z : array_like
        Overall composition (mole fractions, will be normalized).
    points_per_edge : int
        Number of points along each edge.
    epsilon : float
        Small offset to avoid exact zeros (safe for logs/fugacities).
    min_active : int
        Minimum number of components to activate if few exceed average.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_components).
    """
    z = np.asarray(z, float)
    z = z / z.sum()
    n = z.size

    # Ignore components present in small amounts
    active = np.where(z >= 0.15 * z.mean())[0]

    # fallback: ensure at least `min_active` components
    if active.size < min_active:
        order = np.argsort(-z)
        active = order[:min(n, min_active)]

    # Chebyshev nodes mapped to [0, 1]
    k = np.arange(points_per_edge)
    t = 0.5 * (1 - np.cos(np.pi * k / (points_per_edge - 1)))

    pts = []
    # edges only among the selected components
    for i, j in combinations(active, 2):
        for tau in t:
            w = np.zeros(n)

            # inactive components fixed to original z_i
            for idx in range(n):
                if idx not in (i, j):
                    w[idx] = z[idx]

            # fraction available for the edge = leftover after inactive
            inactive_sum = w.sum()
            free = 1.0 - inactive_sum

            # assign varying parts to i, j
            w[i] = (1 - tau) * free
            w[j] = tau * free

            # avoid exact zeros
            w = np.maximum(w, epsilon)
            w /= w.sum()

            pts.append(w)

    return np.array(pts)

# Tangent plane distances within this tolerance of zero are considered
# stable (e.g., the conjugate phase of a composition on the binodal).
tpd_tolerance = 1e-12

def near_pure_points(n, epsilon=1e-3):
    """Return trial compositions rich in one component each."""
    points = np.full((n, n), epsilon)
    np.fill_diagonal(points, 1.)
    points /= points.sum(axis=1, keepdims=True)
    return points

def lle_tangential_plane_analysis(gamma, z, T, P, samples=None):
    """
    Yield trial liquids that make `z` unstable as a single liquid (i.e.,
    with negative tangent plane distance). Candidates are computed lazily,
    in order of increasing cost:

    1. The given samples.
    2. Local minima of the tangent plane distance starting from the given
       samples, edge points of the simplex (only those that improve on the
       best tangent plane distance so far), and near-pure liquids (which
       edge points, biased towards major components, may miss).

    """
    logfz = np.log(z * gamma(z, T, P) + 1e-30)

    def tpd(w):
        return np.dot(w, np.log(w * gamma(w, T, P) + 1e-30) - logfz)

    def tpd_and_gradient(v):
        # Softmax parameterization for unconstrained optimization. By the
        # Gibbs-Duhem equation, the gradient with respect to v is
        # w * (g - tpd), where g is the log fugacity difference.
        w = np.exp(v - v.max())
        w /= w.sum()
        g = np.log(w * gamma(w, T, P) + 1e-30) - logfz
        value = np.dot(w, g)
        return value, w * (g - value)

    def minimize_tpd(w):
        result = minimize(
            tpd_and_gradient, np.log(np.maximum(w, 1e-300)),
            method="L-BFGS-B", jac=True, options=dict(maxiter=20),
        )
        w = np.exp(result.x - result.x.max())
        return w / w.sum(), result.fun

    def is_unstable(w, value):
        return value < -tpd_tolerance and np.abs(w - z).sum() > 1e-9

    if samples is None: samples = ()
    for w in samples:
        if is_unstable(w, tpd(w)): yield w
    MW = np.array([i.MW for i in gamma.chemicals])
    edge_points = edge_points_simplex_masked(z * MW) / MW
    edge_points /= edge_points.sum(axis=1, keepdims=True)
    best = np.inf
    for points, always in ((samples, True),
                           (edge_points, False),
                           (near_pure_points(z.size), True)):
        for w in points:
            if always or tpd(w) < best:
                w, value = minimize_tpd(w)
                best = min(best, value)
                if is_unstable(w, value): yield w

class TangentPlaneStabilityAnalysis:
    __slots__ = ('fugacity_models', 'MW')
    def __init__(self, phases, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.MW = np.array([i.MW for i in chemicals])
        self.fugacity_models =  {
            i: thermo.fugacities(i, chemicals)
            for i in phases
        }
    
    def objective(self, w, T, P, model, logfz, reduce):
        return np.dot(w, np.log(model(w, T, P, reduce) + 1e-30) - logfz)

    def objective_and_gradient(self, v, T, P, model, logfz, reduce):
        # Softmax parameterization for unconstrained optimization. By the
        # Gibbs-Duhem equation, the gradient with respect to v is
        # w * (g - tpd), where g is the log fugacity difference.
        w = np.exp(v - np.max(v))
        w /= w.sum()
        g = np.log(model(w, T, P, reduce) + 1e-30) - logfz
        tpd = np.dot(w, g)
        return tpd, w * (g - tpd)

    def __call__(self, z, T, P, reference_phase='l', potential_phase='L', samples=None):
        reference_model = self.fugacity_models[reference_phase]
        same_phase = reference_phase.lower() == potential_phase.lower()
        logfz = np.log(reference_model(z, T, P, same_phase) + 1e-30)
        objective = self.objective
        model = self.fugacity_models[potential_phase]
        args = (T, P, model, logfz, same_phase)
        if samples is None: samples = ()
        best_val = np.inf
        best_result = None
        for sample in samples:
            val = objective(sample, *args)
            if val < best_val:
                best_result = sample
                best_val = val
            if best_val < -tpd_tolerance and np.abs(best_result - z).sum() > 1e-9:
                return StabilityReport(
                    unstable=True,
                    candidate=sample,
                    tpd=best_val,
                    early_exit=True,
                )
        MW = self.MW
        if best_result is not None: 
            # Minimize from the best sample (otherwise samples could prevent
            # the search below from minimizing at all)
            result = minimize(
                self.objective_and_gradient,
                np.log(np.maximum(best_result, 1e-300)),
                method="L-BFGS-B",
                jac=True,
                options=dict(maxiter=20),
                args=args,
            )
            if result.fun < best_val:
                w = np.exp(result.x - np.max(result.x))
                best_result = w / w.sum()
                best_val = result.fun
        w = z * MW
        w /= w.sum()
        samples = edge_points_simplex_masked(w)
        samples /= MW
        samples /= samples.sum(axis=1, keepdims=True)
        for sample in samples:
            value = objective(sample, *args)
            if value < best_val:
                best_val = value
                best_result = sample
                result = minimize(
                    self.objective_and_gradient,
                    np.log(np.maximum(best_result, 1e-300)),
                    method="L-BFGS-B",
                    jac=True,
                    options=dict(maxiter=20),
                    args=args,
                )
                value = result.fun
                if value < best_val:
                    w = result.x
                    w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
                    w /= w.sum()
                    best_result = w
                    best_val = value
        return StabilityReport(
            unstable=best_val < -tpd_tolerance and np.abs(best_result - z).sum() > 1e-9,
            candidate=best_result,
            tpd=best_val,
        )


if __name__ == '__main__':
    tmo.settings.set_thermo(['Water', 'Octanol', 'Ethanol'])
    TPSA = TangentPlaneStabilityAnalysis('lL', tmo.settings.chemicals)
    report = TPSA(np.array([0.25, 0.8, 0.05]), 298.15, 101325)