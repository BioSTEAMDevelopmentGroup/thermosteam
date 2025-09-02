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

def edge_points_simplex_masked(z: np.ndarray,
                               points_per_edge: int = 5,
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

    # filter: components strictly greater than average
    avg = z.mean()
    active = np.where(z >= avg)[0]

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


class TangentPlaneStabilityAnalysis:
    
    def __init__(self, phases, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.fugacity_models =  {
            i: thermo.fugacities(i, chemicals)
            for i in phases
        }
    
    def objective(self, w, T, P, model, logfz, reduce, softmax=False):
        if softmax:
            w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
            w /= w.sum()
        return np.dot(w, np.log(model(w, T, P, reduce) + 1e-30) - logfz)
    
    def __call__(self, z, T, P, reference_phase='l', potential_phase='L'):
        reference_model = self.fugacity_models[reference_phase]
        same_phase = reference_phase.lower() == potential_phase.lower()
        logfz = np.log(reference_model(z, T, P, same_phase) + 1e-30)
        best_val = np.inf
        best_result = None
        samples = edge_points_simplex_masked(z)
        objective = self.objective
        model = self.fugacity_models[potential_phase]
        args = (T, P, model, logfz, same_phase)
        for sample in samples:
            value = objective(sample, *args)
            if value < best_val:
                best_val = value
                best_result = sample
        result = minimize(
            objective, 
            best_result, 
            method="L-BFGS-B", 
            options=dict(maxiter=5),
            args=(*args, True),
        )
        value = result.fun
        if value < best_val:
            w = result.x
            w = np.exp(w - np.max(w)) # Softmax for unconstrained optimization
            w /= w.sum()
            best_result = w
        return StabilityReport(
            unstable=(best_val < 0),
            candidate=best_result,
            tpd=best_val
        )


if __name__ == '__main__':
    tmo.settings.set_thermo(['Water', 'Octanol'])
    TPSA = TangentPlaneStabilityAnalysis('lL', tmo.settings.chemicals)
    report = TPSA(np.array([0.2, 0.8]), 298.15, 101325)