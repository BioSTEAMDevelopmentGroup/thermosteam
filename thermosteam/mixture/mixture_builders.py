# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
All Mixture object builders.

"""
from ..base import PhaseMixtureHandle
from .ideal_mixture_model import IdealMixtureModel
from .mixture import Mixture

__all__ = ('ideal_mixture',)

# %% Functions

def group_handles_by_phase(phase_handles):
    hasfield = hasattr
    getfield = getattr
    iscallable = callable
    handles_by_phase = {'s': [],
                        'l': [],
                        'g': []}
    for phase, handles in handles_by_phase.items():
        for phase_handle in phase_handles:
            if iscallable(phase_handle) and hasfield(phase_handle, phase):
                prop = getfield(phase_handle, phase)
            else:
                prop = phase_handle
            handles.append(prop)
    return handles_by_phase
    
def build_ideal_PhaseMixtureHandle(chemicals, var):
    setfield = object.__setattr__
    getfield = getattr
    phase_handles = [getfield(i, var) for i in chemicals]
    new = PhaseMixtureHandle.__new__(PhaseMixtureHandle)
    for phase, handles in group_handles_by_phase(phase_handles).items():
        setfield(new, phase, IdealMixtureModel(handles, var))
    setfield(new, 'var', var)
    return new

# %% Ideal mixture model builder 

def ideal_mixture(chemicals,
                  rigorous_energy_balance=True,
                  include_excess_energies=False):
    """
    Create a Mixture object that computes mixture properties using ideal mixing rules.
    
    Parameters
    ----------
    chemicals : Chemicals
        For retrieving pure component chemical data.
    rigorous_energy_balance=True : bool
        Whether to rigorously solve for temperature in energy balance or simply approximate.
    include_excess_energies=False : bool
        Whether to include excess energies in enthalpy and entropy calculations.

    See also
    --------
    :class:`~.mixture.Mixture`
    :class:`~.IdealMixtureModel`

    Examples
    --------
    >>> from thermosteam import Chemicals
    >>> from thermosteam.mixture import ideal_mixture
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> ideal_mixture_model = ideal_mixture(chemicals)
    >>> ideal_mixture_model.Hvap([0.2, 0.8], 350)
    39601.089191849824


    """
    chemicals = tuple(chemicals)
    getfield = getattr
    Cn =  build_ideal_PhaseMixtureHandle(chemicals, 'Cn')
    H =  build_ideal_PhaseMixtureHandle(chemicals, 'H')
    S = build_ideal_PhaseMixtureHandle(chemicals, 'S')
    H_excess = build_ideal_PhaseMixtureHandle(chemicals, 'H_excess')
    S_excess = build_ideal_PhaseMixtureHandle(chemicals, 'S_excess')
    mu = build_ideal_PhaseMixtureHandle(chemicals, 'mu')
    V = build_ideal_PhaseMixtureHandle(chemicals, 'V')
    kappa = build_ideal_PhaseMixtureHandle(chemicals, 'kappa')
    Hvap = IdealMixtureModel([getfield(i, 'Hvap') for i in chemicals], 'Hvap')
    sigma = IdealMixtureModel([getfield(i, 'sigma') for i in chemicals], 'sigma')
    epsilon = IdealMixtureModel([getfield(i, 'epsilon') for i in chemicals], 'epsilon')
    return Mixture('ideal mixing', Cn, H, S, H_excess, S_excess,
                   mu, V, kappa, Hvap, sigma, epsilon,
                   rigorous_energy_balance, include_excess_energies)