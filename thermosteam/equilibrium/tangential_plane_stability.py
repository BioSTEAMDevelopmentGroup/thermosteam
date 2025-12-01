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
    sample_unstable: bool = False

def edge_points_simplex_masked(z: np.ndarray,
                               points_per_edge: int = 8,
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

# Light weight
def lle_tangential_plan_analysis(gamma, z, T, P, sample=None):
    MW = np.array([i.MW for i in gamma.chemicals])
    logfz = np.log(z * gamma(z, T, P) + 1e-30)
    
    def objective(w, T, P, logfz, softmax=False):
        if softmax:
            w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
            w /= w.sum()
        return np.dot(w, np.log(w * gamma(w, T, P) + 1e-30) - logfz)
    
    args = (T, P, logfz)
    if sample is None:
        best_val = np.inf
        best_result = None
    else:
        best_result = sample
        best_val = objective(sample, *args)
        result = minimize(
            objective, 
            best_result, 
            method="L-BFGS-B", 
            options=dict(maxiter=20),
            args=(*args, True),
        )
        value = result.fun
        if value < best_val:
            w = result.x
            w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
            w /= w.sum()
            best_result = w
            best_val = value
        unstable = best_val < 0
        if unstable:
            return StabilityReport(
                unstable=unstable,
                candidate=sample,
                tpd=best_val,
                sample_unstable=True,
            )
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
                objective, 
                best_result, 
                method="L-BFGS-B", 
                options=dict(maxiter=20),
                args=(*args, True),
            )
            value = result.fun
            if value < best_val:
                w = result.x
                w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
                w /= w.sum()
                best_result = w
                best_val = value
    return StabilityReport(
        unstable=(best_val < 0),
        candidate=best_result,
        tpd=best_val
    )

class TangentPlaneStabilityAnalysis:
    __slots__ = ('fugacity_models', 'MW')
    def __init__(self, phases, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.MW = np.array([i.MW for i in chemicals])
        self.fugacity_models =  {
            i: thermo.fugacities(i, chemicals)
            for i in phases
        }
    
    def objective(self, w, T, P, model, logfz, reduce, softmax=False):
        if softmax:
            w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
            w /= w.sum()
        return np.dot(w, np.log(model(w, T, P, reduce) + 1e-30) - logfz)
    
    def __call__(self, z, T, P, reference_phase='l', potential_phase='L', sample=None):
        reference_model = self.fugacity_models[reference_phase]
        same_phase = reference_phase.lower() == potential_phase.lower()
        logfz = np.log(reference_model(z, T, P, same_phase) + 1e-30)
        objective = self.objective
        model = self.fugacity_models[potential_phase]
        args = (T, P, model, logfz, same_phase)
        if sample is None:
            best_val = np.inf
            best_result = None
        else:
            best_result = sample
            best_val = objective(sample, *args)
            result = minimize(
                objective, 
                best_result, 
                method="L-BFGS-B", 
                options=dict(maxiter=20),
                args=(*args, True),
            )
            value = result.fun
            if value < best_val:
                w = result.x
                w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
                w /= w.sum()
                best_result = w
                best_val = value
            unstable = best_val < 0
            if unstable:
                return StabilityReport(
                    unstable=unstable,
                    candidate=sample,
                    tpd=best_val,
                    sample_unstable=True,
                )
        MW = self.MW
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
                    objective, 
                    best_result, 
                    method="L-BFGS-B", 
                    options=dict(maxiter=20),
                    args=(*args, True),
                )
                value = result.fun
                if value < best_val:
                    w = result.x
                    w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
                    w /= w.sum()
                    best_result = w
                    best_val = value
        return StabilityReport(
            unstable=(best_val < 0),
            candidate=best_result,
            tpd=best_val
        )


if __name__ == '__main__':
    tmo.settings.set_thermo(['Water', 'Octanol', 'Ethanol'])
    TPSA = TangentPlaneStabilityAnalysis('lL', tmo.settings.chemicals)
    report = TPSA(np.array([0.25, 0.8, 0.05]), 298.15, 101325)