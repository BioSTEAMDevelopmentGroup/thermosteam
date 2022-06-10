# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
This module contains functions for modeling separations in unit operations.

"""
from warnings import warn
from numba import njit
import thermosteam as tmo
import flexsolve as flx
import numpy as np
import pandas as pd
from .utils import thermo_user
from .exceptions import InfeasibleRegion
from .equilibrium import phase_fraction as compute_phase_fraction

__all__ = (
    'mix_and_split',
    'adjust_moisture_content', 
    'mix_and_split_with_moisture_content',
    'partition_coefficients',
    'vle_partition_coefficients',
    'lle_partition_coefficients',
    'partition', 'lle', 'vle', 
    'material_balance',
    'StageEquilibrium',
    'MultiStageEquilibrium',
    'flow_rates_for_multi_stage_equilibrium',
    'chemical_splits',
    'phase_fraction',
)

def check_partition_infeasibility(infeasible_index, strict, stacklevel=1):
    if infeasible_index.any():
        if strict:
            raise InfeasibleRegion('negative flow rates in equilibrium '
                                   'solution; partition data')
        else:
            warning = RuntimeWarning(
                'phase equilibrium solution results in negative flow rates; '
                'negative flows have been removed from solution'
            )
            warn(warning, stacklevel=stacklevel+1)


def handle_infeasible_flow_rates(mol, maxmol, strict, stacklevel=1):
    infeasible_index = mol < 0.
    check_partition_infeasibility(infeasible_index, strict, stacklevel+1)
    mol[infeasible_index] = 0.
    infeasible_index = mol > maxmol
    check_partition_infeasibility(infeasible_index, strict, stacklevel+1)
    mol[infeasible_index] = maxmol[infeasible_index]


# %% Mixing, splitting, and moisture content

CAS_water = '7732-18-5'

def mix_and_split_with_moisture_content(ins, retentate, permeate,
                                        split, moisture_content, ID=None):
    """
    Run splitter mass and energy balance with mixing all input streams and 
    and ensuring retentate moisture content.
    
    Parameters
    ----------
    ins : Iterable[Stream]
        Inlet fluids with solids.
    retentate : Stream
    permeate : Stream
    split : array_like
        Component splits to the retentate.
    moisture_content : float
        Fraction of water in retentate.

    Examples
    --------
    >>> import thermosteam as tmo
    >>> Solids = tmo.Chemical('Solids', default=True, search_db=False, phase='s')
    >>> tmo.settings.set_thermo(['Water', Solids])
    >>> feed = tmo.Stream('feed', Water=100, Solids=10, units='kg/hr')
    >>> wash_water = tmo.Stream('wash_water', Water=10, units='kg/hr')
    >>> retentate = tmo.Stream('retentate')
    >>> permeate = tmo.Stream('permeate')
    >>> split = [0., 1.]
    >>> moisture_content = 0.5
    >>> tmo.separations.mix_and_split_with_moisture_content(
    ...     [feed, wash_water], retentate, permeate, split, moisture_content
    ... )
    >>> retentate.show(flow='kg/hr')
    Stream: retentate
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kg/hr): Water   10
                   Solids  10
    >>> permeate.show(flow='kg/hr')
    Stream: permeate
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kg/hr): Water  100

    """
    mix_and_split(ins, retentate, permeate, split)
    adjust_moisture_content(retentate, permeate, moisture_content, ID)

def adjust_moisture_content(retentate, permeate, moisture_content, ID=None):
    """
    Remove water from permate to adjust retentate moisture content.
    
    Parameters
    ----------
    retentate : Stream
    permeate : Stream
    moisture_content : float
        Fraction of water in retentate.

    Examples
    --------
    >>> import thermosteam as tmo
    >>> Solids = tmo.Chemical('Solids', default=True, search_db=False, phase='s')
    >>> tmo.settings.set_thermo(['Water', Solids])
    >>> retentate = tmo.Stream('retentate', Solids=20, units='kg/hr')
    >>> permeate = tmo.Stream('permeate', Water=50, Solids=0.1, units='kg/hr')
    >>> moisture_content = 0.5
    >>> tmo.separations.adjust_moisture_content(retentate, permeate, moisture_content)
    >>> retentate.show(flow='kg/hr')
    Stream: retentate
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kg/hr): Water   20
                   Solids  20
    >>> permeate.show(flow='kg/hr')
    Stream: permeate
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kg/hr): Water   30
                   Solids  0.1
    
    Note that if not enough water is available, an InfeasibleRegion error is raised:
        
    >>> retentate.imol['Water'] = permeate.imol['Water'] = 0
    >>> tmo.separations.adjust_moisture_content(retentate, permeate, moisture_content)
    Traceback (most recent call last):
    InfeasibleRegion: not enough water; permeate moisture content is infeasible

    """
    F_mass = retentate.F_mass
    mc = moisture_content
    if ID is None: 
        ID = CAS_water
        MW = 18.01528
        retentate_water = retentate.imol[ID]
        dry_mass = F_mass - MW * retentate_water
        retentate.imol[ID] = water = (dry_mass * mc/(1-mc)) / MW
        permeate.imol[ID] -= water - retentate_water
    else:
        retentate_moisture = retentate.imass[ID]
        dry_mass = F_mass - retentate_moisture
        retentate.imass[ID] = moisture = dry_mass * mc/(1-mc)
        permeate.imass[ID] -= moisture - retentate_moisture
    if permeate.imol[ID] < 0:
        raise InfeasibleRegion(f'not enough {ID}; permeate moisture content')

def mix_and_split(ins, top, bottom, split):
    """
    Run splitter mass and energy balance with mixing all input streams.
    
    Parameters
    ----------
    ins : Iterable[Stream]
        All inlet fluids.
    top : Stream
        Top inlet fluid.
    bottom : Stream
        Bottom inlet fluid
    split : array_like
        Component-wise split of feed to the top stream.
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> feed_a = tmo.Stream(Water=20, Ethanol=5)
    >>> feed_b = tmo.Stream(Water=15, Ethanol=5)
    >>> split = 0.8
    >>> effluent_a = tmo.Stream('effluent_a')
    >>> effluent_b = tmo.Stream('effluent_b')
    >>> tmo.separations.mix_and_split([feed_a, feed_b], effluent_a, effluent_b, split)
    >>> effluent_a.show()
    Stream: effluent_a
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    28
                     Ethanol  8
    >>> effluent_b.show()
    Stream: effluent_b
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    7
                     Ethanol  2
    
    """
    top.mix_from(ins)
    top.split_to(top, bottom, split, energy_balance=True)

def phase_split(feed, outlets):
    """
    Split the feed to outlets by phase.
    
    Parameters
    ----------
    feed : stream
    outlets : streams
        
    Notes
    -----
    Phases allocate to outlets in alphabetical order. For example,
    if the feed.phases is 'gls' (i.e. gas, liquid, and solid), the phases
    of the outlets will be 'g', 'l', and 's'.
        
    Examples
    --------
    Split gas and liquid phases to streams:
    
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> feed = tmo.Stream('feed', Water=10, Ethanol=10)
    >>> feed.vle(V=0.5, P=101325)
    >>> vapor = tmo.Stream('vapor')
    >>> liquid = tmo.Stream('liquid')
    >>> outlets = [vapor, liquid]
    >>> tmo.separations.phase_split(feed, outlets)
    >>> vapor.show()
    Stream: vapor
     phase: 'g', T: 353.88 K, P: 101325 Pa
     flow (kmol/hr): Water    3.86
                     Ethanol  6.14
    >>> liquid.show()
    Stream: liquid
     phase: 'l', T: 353.88 K, P: 101325 Pa
     flow (kmol/hr): Water    6.14
                     Ethanol  3.86
    
    Note that the number of phases in the feed should be equal to the number of 
    outlets:
        
    >>> tmo.separations.phase_split(feed, [vapor])
    Traceback (most recent call last):
    RuntimeError: number of phases in feed must be equal to the number of outlets
    
    """
    phases = feed.phases
    if len(outlets) != len(phases):
        raise RuntimeError('number of phases in feed must be equal to the number of outlets')
    for i,j in zip(feed, outlets): j.copy_like(i)

# %% Single stage equilibrium

def partition_coefficients(IDs, top, bottom):
    """
    Return partition coefficients given streams in equilibrium.
    
    Parameters
    ----------
    top : Stream
        Vapor fluid.
    bottom : Stream
        Liquid fluid.
    IDs : tuple[str]
        IDs of chemicals in equilibrium.
    
    Returns
    -------
    K : 1d array
        Patition coefficients in mol fraction in top stream over mol fraction in bottom stream.

    Examples
    --------
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', tmo.Chemical('O2', phase='g')], cache=True)
    >>> s = tmo.Stream('s', Water=20, Ethanol=20, O2=0.1)
    >>> s.vle(V=0.5, P=101325)
    >>> tmo.separations.partition_coefficients(('Water', 'Ethanol'), s['g'], s['l'])
    array([0.629, 1.59 ])

    """
    numerator = top.get_normalized_mol(IDs)
    denominator = bottom.get_normalized_mol(IDs)
    denominator[denominator < 1e-24] = 1e-24
    return numerator / denominator

def chemical_splits(a, b=None, mixed=None):
    """
    Return a ChemicalIndexer with splits for all chemicals to stream `a`.
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> stream = tmo.Stream('stream', Water=10, Ethanol=10)
    >>> stream.vle(V=0.5, P=101325)
    >>> isplits = tmo.separations.chemical_splits(stream['g'], stream['l'])
    >>> isplits.show()
    ChemicalIndexer:
     Water    0.3861
     Ethanol  0.6139
    >>> isplits = tmo.separations.chemical_splits(stream['g'], mixed=stream)
    >>> isplits.show()
    ChemicalIndexer:
     Water    0.3861
     Ethanol  0.6139
     
    """
    mixed_mol = mixed.mol.copy() if mixed else a.mol + b.mol
    mixed_mol[mixed_mol==0.] = 1.
    return tmo.indexer.ChemicalIndexer.from_data(a.mol / mixed_mol)

def vle_partition_coefficients(top, bottom):
    """
    Return VLE partition coefficients given vapor and liquid streams in equilibrium.
    
    Parameters
    ----------
    top : Stream
        Vapor fluid.
    bottom : Stream
        Liquid fluid.
    
    Returns
    -------
    IDs : tuple[str]
        IDs for chemicals in vapor-liquid equilibrium.
    K : 1d array
        Patition coefficients in mol fraction in vapor over mol fraction in liquid.

    Examples
    --------
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', tmo.Chemical('O2', phase='g')], cache=True)
    >>> s = tmo.Stream('s', Water=20, Ethanol=20, O2=0.1)
    >>> s.vle(V=0.5, P=101325)
    >>> IDs, K = tmo.separations.vle_partition_coefficients(s['g'], s['l'])
    >>> IDs
    ('Water', 'Ethanol')
    >>> K
    array([0.629, 1.59 ])

    """
    IDs = tuple([i.ID for i in bottom.vle_chemicals])
    return IDs, partition_coefficients(IDs, top, bottom)

def lle_partition_coefficients(top, bottom):
    """
    Return LLE partition coefficients given two liquid streams in equilibrium.
    
    Parameters
    ----------
    top : Stream
        Liquid fluid.
    bottom : Stream
        Other liquid fluid.
    
    Returns
    -------
    IDs : tuple[str]
        IDs for chemicals in liquid-liquid equilibrium.
    K : 1d array
        Patition coefficients in mol fraction in top liquid over mol fraction in bottom liquid.

    Examples
    --------
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Octanol'], cache=True)
    >>> s = tmo.Stream('s', Water=20, Octanol=20, Ethanol=1)
    >>> s.lle(T=298.15, P=101325)
    >>> IDs, K = tmo.separations.lle_partition_coefficients(s['l'], s['L'])
    >>> IDs
    ('Water', 'Ethanol', 'Octanol')
    >>> K
    array([6.82e+00, 2.38e-01, 3.00e-04])

    """
    IDs = tuple([i.ID for i in bottom.lle_chemicals])
    return IDs, partition_coefficients(IDs, top, bottom)

def phase_fraction(feed, IDs, K, phi=None, top_chemicals=None, 
                   bottom_chemicals=None, strict=False, stacklevel=1):
    """
    Return the phase fraction given a stream and partition coeffiecients.

    Parameters
    ----------
    feed : Stream
        Mixed feed.
    IDs : tuple[str]
        IDs of chemicals in equilibrium.
    K : 1d array
        Partition coefficeints corresponding to IDs.
    phi : float, optional
        Guess phase fraction in top phase.
    top_chemicals : tuple[str], optional
        Chemicals that remain in the top fluid.
    bottom_chemicals : tuple[str], optional
        Chemicals that remain in the bottom fluid.
    strict : bool, optional
        Whether to raise an InfeasibleRegion exception when solution results
        in negative flow rates or to remove negative flows and issue a warning. 
        Defaults to False.

    Returns
    -------
    phi : float
        Phase fraction in top phase.

    Notes
    -----
    Chemicals not in equilibrium end up in the top phase.

    Examples
    --------
    >>> import numpy as np
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', tmo.Chemical('NaCl', default=True), 
    ...                          tmo.Chemical('O2', phase='g')], cache=True)
    >>> IDs = ('Water', 'Ethanol')
    >>> K = np.array([0.629, 1.59])
    >>> feed = tmo.Stream('feed', Water=20, Ethanol=20, O2=0.1)
    >>> tmo.separations.phase_fraction(feed, IDs, K)
    0.500
    >>> feed = tmo.Stream('feed', Water=20, Ethanol=20, NaCl=0.1, O2=0.1)
    >>> tmo.separations.phase_fraction(feed, IDs, K, 
    ...                                top_chemicals=('O2',),
    ...                                bottom_chemicals=('NaCl'))
    0.500
    
    """
    mol = feed.imol[IDs]
    F_mol = mol.sum()
    Fa = feed.imol[top_chemicals].sum() if top_chemicals else 0.
    if bottom_chemicals:
        bottom_flows = feed.imol[bottom_chemicals]
        Fb = bottom_flows.sum()
    else:
        Fb = 0.
    F_mol += Fa + Fb
    z_mol = mol / F_mol
    phi = compute_phase_fraction(z_mol, K, phi, Fa/F_mol, Fb/F_mol)
    if phi <= 0.:
        phi = 0.
    elif phi < 1.:
        x = z_mol / (phi * K + (1. - phi))
        bottom_mol = x * (1. - phi) * F_mol
        handle_infeasible_flow_rates(bottom_mol, mol, strict, stacklevel+1)
    else:
        phi = 1.
    return phi

def partition(feed, top, bottom, IDs, K, phi=None, top_chemicals=None, 
              bottom_chemicals=None, strict=False, stacklevel=1):
    """
    Run equilibrium of feed to top and bottom streams given partition 
    coeffiecients and return the phase fraction.

    Parameters
    ----------
    feed : Stream
        Mixed feed.
    top : Stream
        Top fluid.
    bottom : Stream
        Bottom fluid.
    IDs : tuple[str]
        IDs of chemicals in equilibrium.
    K : 1d array
        Partition coefficeints corresponding to IDs.
    phi : float, optional
        Guess phase fraction in top phase.
    top_chemicals : tuple[str], optional
        Chemicals that remain in the top fluid.
    bottom_chemicals : tuple[str], optional
        Chemicals that remain in the bottom fluid.
    strict : bool, optional
        Whether to raise an InfeasibleRegion exception when solution results
        in negative flow rates or to remove negative flows and issue a warning. 
        Defaults to False.

    Returns
    -------
    phi : float
        Phase fraction in top phase.

    Notes
    -----
    Chemicals not in equilibrium end up in the top phase.

    Examples
    --------
    >>> import numpy as np
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', tmo.Chemical('NaCl', default=True), 
    ...                          tmo.Chemical('O2', phase='g')], cache=True)
    >>> IDs = ('Water', 'Ethanol')
    >>> K = np.array([0.629, 1.59])
    >>> feed = tmo.Stream('feed', Water=20, Ethanol=20, O2=0.1)
    >>> top = tmo.Stream('top')
    >>> bottom = tmo.Stream('bottom')
    >>> tmo.separations.partition(feed, top, bottom, IDs, K)
    0.500
    >>> top.show()
    Stream: top
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    7.73
                     Ethanol  12.3
                     O2       0.1
    >>> bottom.show()
    Stream: bottom
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    12.3
                     Ethanol  7.72
    >>> feed = tmo.Stream('feed', Water=20, Ethanol=20, NaCl=0.1, O2=0.1)
    >>> tmo.separations.partition(feed, top, bottom, IDs, K, 
    ...                           top_chemicals=('O2',),
    ...                           bottom_chemicals=('NaCl'))
    0.500
    >>> top.show()
    Stream: top
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    7.73
                     Ethanol  12.3
                     O2       0.1
    >>> bottom.show()
    Stream: bottom
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    12.3
                     Ethanol  7.72
                     NaCl     0.1

    """
    feed_mol = feed.mol
    mol = feed.imol[IDs]
    F_mol = mol.sum()
    if not bottom.shares_flow_rate_with(feed): bottom.empty()
    Fa = feed.imol[top_chemicals].sum() if top_chemicals else 0.
    if bottom_chemicals:
        bottom.imol[bottom_chemicals] = bottom_flows = feed.imol[bottom_chemicals]
        Fb = bottom_flows.sum()
    else:
        Fb = 0.
    F_mol += Fa + Fb
    z_mol = mol / F_mol
    phi = compute_phase_fraction(z_mol, K, phi, Fa/F_mol, Fb/F_mol)
    if phi <= 0.:
        bottom.imol[IDs] = mol
        phi = 0.
    elif phi < 1.:
        x = z_mol / (phi * K + (1. - phi))
        bottom_mol = x * (1. - phi) * F_mol
        handle_infeasible_flow_rates(bottom_mol, mol, strict, stacklevel+1)
        bottom.imol[IDs] = bottom_mol
    else:
        phi = 1.        
    top.mol[:] = feed_mol - bottom.mol
    return phi

def lle(feed, top, bottom, top_chemical=None, efficiency=1.0, multi_stream=None):
    """
    Run LLE mass and energy balance.
    
    Parameters
    ----------
    feed : Stream
        Mixed feed.
    top : Stream
        Top fluid.
    bottom : Stream
        Bottom fluid.
    top_chemical : str, optional
        Identifier of chemical that will be favored in the top fluid.
    efficiency=1. : float,
        Fraction of feed in liquid-liquid equilibrium.
        The rest of the feed is divided equally between phases
    multi_stream : MultiStream, optional
        Data from feed is passed to this stream to perform liquid-liquid equilibrium.
    
    Examples
    --------
    Perform liquid-liquid equilibrium around water and octanol and split the phases:
    
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Octanol'], cache=True)
    >>> feed = tmo.Stream('feed', Water=20, Octanol=20, Ethanol=1)
    >>> top = tmo.Stream('top')
    >>> bottom = tmo.Stream('bottom')
    >>> tmo.separations.lle(feed, top, bottom)
    >>> top.show()
    Stream: top
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    3.55
                     Ethanol  0.861
                     Octanol  20
    >>> bottom.show()
    Stream: bottom
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water    16.5
                     Ethanol  0.139
                     Octanol  0.00409
    
    Assume that 1% of the feed is not in equilibrium (possibly due to poor mixing):
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Octanol'], cache=True)
    >>> feed = tmo.Stream('feed', Water=20, Octanol=20, Ethanol=1)
    >>> top = tmo.Stream('top')
    >>> bottom = tmo.Stream('bottom')
    >>> ms = tmo.MultiStream('ms', phases='lL') # Store flow rate data here as well
    >>> tmo.separations.lle(feed, top, bottom, efficiency=0.99, multi_stream=ms)
    >>> ms.show()
    MultiStream: ms
     phases: ('L', 'l'), T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): (L) Water    3.55
                         Ethanol  0.861
                         Octanol  20
                     (l) Water    16.5
                         Ethanol  0.139
                         Octanol  0.00408
    
    """
    if multi_stream:
        ms = multi_stream
        ms.copy_like(feed)
    else:
        ms = feed.copy()
    ms.lle(feed.T, top_chemical=top_chemical)
    top_phase, bottom_phase = ms.phases
    if not top_chemical:
        rho_l = ms['l'].rho
        rho_L = ms['L'].rho
        top_L = rho_L < rho_l
        if top_L:
            top_phase = 'L'
            bottom_phase = 'l'
    top.mol[:] = ms.imol[top_phase]
    bottom.mol[:] = ms.imol[bottom_phase]
    top.T = bottom.T = feed.T
    top.P = bottom.P = feed.P
    if efficiency < 1.:
        top.mol *= efficiency
        bottom.mol *= efficiency
        mixing = (1. - efficiency) / 2. * feed.mol
        top.mol += mixing
        bottom.mol += mixing
        
def vle(feed, vap, liq, T=None, P=None, V=None, Q=None, x=None, y=None,
        multi_stream=None):
    """
    Run VLE mass and energy balance.
    
    Parameters
    ----------
    feed : Stream
        Mixed feed.
    vap : Stream
        Vapor fluid.
    liq : Stream
        Liquid fluid.
    P=None : float
        Operating pressure [Pa].
    Q=None : float
        Duty [kJ/hr].
    T=None : float
        Operating temperature [K].
    V=None : float
        Molar vapor fraction.
    x=None : float
        Molar composition of liquid (for binary mixtures).
    y=None : float
        Molar composition of vapor (for binary mixtures).
    multi_stream : MultiStream, optional
        Data from feed is passed to this stream to perform vapor-liquid equilibrium.
    
    Examples
    --------
    Perform vapor-liquid equilibrium on water and ethanol and split phases
    to vapor and liquid streams:
    
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> feed = tmo.Stream('feed', Water=20, Ethanol=20)
    >>> vapor = tmo.Stream('top')
    >>> liquid = tmo.Stream('bottom')
    >>> tmo.separations.vle(feed, vapor, liquid, V=0.5, P=101325)
    >>> vapor.show()
    Stream: top
     phase: 'g', T: 353.88 K, P: 101325 Pa
     flow (kmol/hr): Water    7.72
                     Ethanol  12.3
    >>> liquid.show()
    Stream: bottom
     phase: 'l', T: 353.88 K, P: 101325 Pa
     flow (kmol/hr): Water    12.3
                     Ethanol  7.72
    
    It is also possible to save flow rate data in a multi-stream as well:
        
    >>> ms = tmo.MultiStream('ms', phases='lg')
    >>> tmo.separations.vle(feed, vapor, liquid, V=0.5, P=101325, multi_stream=ms)
    >>> ms.show()
    MultiStream: ms
     phases: ('g', 'l'), T: 353.88 K, P: 101325 Pa
     flow (kmol/hr): (g) Water    7.72
                         Ethanol  12.3
                     (l) Water    12.3
                         Ethanol  7.72
    
    """
    if multi_stream:
        ms = multi_stream
        ms.copy_like(feed)
    else:
        ms = feed.copy()
    H = feed.H + Q if Q is not None else None
    ms.vle(P=P, H=H, T=T, V=V, x=x, y=y)

    # Set Values
    vap.phase = 'g'
    liq.phase = 'l'
    vap.mol[:] = ms.imol['g']
    liq.mol[:] = ms.imol['l']
    vap.T = liq.T = ms.T
    vap.P = liq.P = ms.P
    
def material_balance(chemical_IDs, variable_inlets, constant_inlets=(),
                     constant_outlets=(), is_exact=True, balance='flow'):
    """
    Solve stream mass balance by iteration.
    
    Parameters
    ----------
    chemical_IDs : tuple[str]
        Chemicals that will be used to solve mass balance linear equations.
        The number of chemicals must be same as the number of input streams varied.
    variable_inlets : Iterable[Stream]
        Inlet streams that can vary in net flow rate to accomodate for the
        mass balance.
    constant_inlets: Iterable[Stream], optional
        Inlet streams that cannot vary in flow rates.
    constant_outlets: Iterable[Stream], optional
        Outlet streams that cannot vary in flow rates.
    is_exact=True : bool, optional
        True if exact flow rate solution is required for the specified IDs.
    balance='flow' : {'flow', 'composition'}, optional
          * 'flow': Satisfy output flow rates
          * 'composition': Satisfy net output molar composition
    
    Examples
    --------
    Vary inlet flow rates to satisfy outlet flow rates:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> in_a = tmo.Stream('in_a', Water=1)
    >>> in_b = tmo.Stream('in_b', Ethanol=1)
    >>> variable_inlets = [in_a, in_b]
    >>> in_c = tmo.Stream('in_c', Water=100)
    >>> constant_inlets = [in_c]
    >>> out_a = tmo.Stream('out_a', Water=200, Ethanol=2)
    >>> out_b = tmo.Stream('out_b', Ethanol=100)
    >>> constant_outlets = [out_a, out_b]
    >>> chemical_IDs = ('Water', 'Ethanol')
    >>> tmo.separations.material_balance(chemical_IDs, variable_inlets, constant_inlets, constant_outlets)
    >>> tmo.Stream.sum([in_a, in_b, in_c]).mol - tmo.Stream.sum([out_a, out_b]).mol # Molar flow rates entering and leaving are equal
    array([0., 0.])
    
    Vary inlet flow rates to satisfy outlet composition:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> in_a = tmo.Stream('in_a', Water=1)
    >>> in_b = tmo.Stream('in_b', Ethanol=1)
    >>> variable_inlets = [in_a, in_b]
    >>> in_c = tmo.Stream('in_c', Water=100)
    >>> constant_inlets = [in_c]
    >>> out_a = tmo.Stream('out_a', Water=200, Ethanol=2)
    >>> out_b = tmo.Stream('out_b', Ethanol=100)
    >>> constant_outlets = [out_a, out_b]
    >>> chemical_IDs = ('Water', 'Ethanol')
    >>> tmo.separations.material_balance(chemical_IDs, variable_inlets, constant_inlets, constant_outlets, balance='composition')
    >>> tmo.Stream.sum([in_a, in_b, in_c]).z_mol - tmo.Stream.sum([out_a, out_b]).z_mol # Molar composition entering and leaving are equal
    array([0., 0.])
    
    """
    # SOLVING BY ITERATION TAKES 15 LOOPS FOR 2 STREAMS
    # SOLVING BY LEAST-SQUARES TAKES 40 LOOPS
    solver = np.linalg.solve if is_exact else np.linalg.lstsq

    # Set up constant and variable streams
    if not variable_inlets:
        raise ValueError('variable_inlets must contain at least one stream')
    index = variable_inlets[0].chemicals.get_index(chemical_IDs)
    mol_out = sum([s.mol for s in constant_outlets])

    if balance == 'flow':
        # Perform the following calculation: Ax = b = f - g
        # Where:
        #    A = flow rate array
        #    x = factors
        #    b = target flow rates
        #    f = output flow rates
        #    g = constant inlet flow rates

        # Solve linear equations for mass balance
        A = np.array([s.mol for s in variable_inlets]).transpose()[index, :]
        f = mol_out[index]
        g = sum([s.mol[index] for s in constant_inlets])
        b = f - g
        x = solver(A, b)

        # Set flow rates for input streams
        for factor, s in zip(x, variable_inlets):
            s.mol[:] = s.mol * factor

    elif balance == 'composition':
        # Perform the following calculation:
        # Ax = b
        #    = sum( A_ * x_guess + g_ )f - g
        #    = A_ * x_guess * f - O
        # O  = sum(g_)*f - g
        # Where:
        # A_ is flow array for all species
        # g_ is constant flows for all species
        # Same variable definitions as in 'flow'

        # Set all variables
        A_ = np.array([s.mol for s in variable_inlets]).transpose()
        A = np.array([s.mol for s in variable_inlets]).transpose()[index, :]
        F_mol_out = mol_out.sum()
        z_mol_out = mol_out / F_mol_out if F_mol_out else mol_out
        f = z_mol_out[index]
        g_ = sum([s.mol for s in constant_inlets])
        g = g_[index]
        O = sum(g_) * f - g

        # Solve by iteration
        x_guess = np.ones_like(index)
        not_converged = True
        while not_converged:
            # Solve linear equations for mass balance
            b = (A_ * x_guess).sum()*f + O
            x_new = solver(A, b)
            infeasibles = x_new < 0.
            if infeasibles.any(): x_new -= x_new[infeasibles].min()
            denominator = x_guess.copy()
            denominator[denominator == 0.] = 1.
            not_converged = sum(((x_new - x_guess)/denominator)**2) > 1e-6
            x_guess = x_new

        # Set flow rates for input streams
        for factor, s in zip(x_new, variable_inlets):
            s.mol = s.mol * factor
    
    else:
        raise ValueError( "balance must be one of the following: 'flow', 'composition'")
        
# %% Equilibrium objects.

@njit(cache=True)
def _vle_phi_K(vapor, liquid):
    F_vapor = vapor.sum()
    F_liquid = liquid.sum()
    phi = F_vapor / (F_vapor + F_liquid)
    y = vapor / F_vapor
    x = liquid / F_liquid
    return phi, y / x 

def _get_specification(name, value):
    if name == 'Duty':
        B = None
        Q = value
    elif name == 'Reflux':
        B = 1 / value
        Q = None
    elif name == 'Boilup':
        B = value
        Q = None
    else:
        raise RuntimeError(f"specification '{name}' not implemented for condenser")
    return B, Q

@thermo_user
class StageEquilibrium:
    __slots__ = ('feeds', 'multi_stream', 
                 '_thermo', 'phi', 'K', 'IDs',)
    
    strict_infeasibility_check = False
    
    def __init__(self, feeds=None, phases=None, thermo=None):
        if feeds is None: feeds = []
        self.feeds = feeds
        thermo = self._load_thermo(thermo)
        self.multi_stream = tmo.MultiStream(
            None, phases=phases or ('g', 'l'), thermo=thermo
        )
        self.phi = None
        self.IDs = None
        self.K = None
    
    @property
    def extract(self):
        return self.multi_stream['L']
    @property
    def raffinate(self):
        return self.multi_stream['l']
    
    @property
    def vapor(self):
        return self.multi_stream['g']
    @property
    def liquid(self):
        return self.multi_stream['l']
        
    @property
    def T(self):
        return self.multi_stream.T
    @T.setter
    def T(self, T):
        self.multi_stream.T = T
        
    @property
    def P(self):
        return self.multi_stream.P
    @P.setter
    def P(self, P):
        self.multi_stream.P = P
        
    def partition_lle(self, partition_data=None, stacklevel=1, P=None, solvent=None):
        ms = self.multi_stream
        data = ms._imol._data.copy()
        ms.mix_from(self.feeds, energy_balance=False)
        top, bottom = ms
        if partition_data:
            self.K = K = partition_data['K']
            self.IDs = IDs = partition_data['IDs']
            args = (IDs, K, self.phi or partition_data['phi'], 
                    partition_data.get('extract_chemicals') or partition_data.get('vapor_chemicals'),
                    partition_data.get('raffinate_chemicals') or partition_data.get('liquid_chemicals'),
                    self.strict_infeasibility_check, stacklevel+1)
            self.phi = partition(ms, top, bottom, *args)
        else:
            eq = ms.lle
            lle_chemicals, K_new, phi = eq(T=ms.T, P=P, top_chemical=solvent, update=False)
            IDs = tuple([i.ID for i in lle_chemicals])
            IDs_last = self.IDs
            if IDs_last and IDs_last != IDs:
                Ks = self.K
                for ID, K in zip(IDs, K_new): Ks[IDs_last.index(ID)] = K
            else:
                self.K = K_new
                self.IDs = IDs
                self.phi = phi
        ms._imol._data[:] = data
        
    def partition_vle(self, partition_data=None, stacklevel=1, P=None):
        ms = self.multi_stream
        data = ms._imol._data.copy()
        top, bottom = ms
        if partition_data:
            self.K = K = partition_data['K']
            self.IDs = IDs = partition_data['IDs']
            args = (IDs, K, self.phi or partition_data['phi'], 
                    partition_data.get('extract_chemicals') or partition_data.get('vapor_chemicals'),
                    partition_data.get('raffinate_chemicals') or partition_data.get('liquid_chemicals'),
                    self.strict_infeasibility_check, stacklevel+1)
            self.phi = partition(ms, top, bottom, *args)
        else:
            try: bp = bottom.bubble_point_at_P(P, IDs=self.IDs)
            except: pass
            else:
                z = bp.z
                z[z == 0] = 1.
                self.K = bp.y / bp.z
                ms.T = bp.T
        ms._imol._data[:] = data
        
    def balance_flows(self, top_split, bottom_split):
        feed, *other_feeds = self.feeds
        total_mol = sum([i.mol for i in other_feeds], feed.mol)
        ms = self.multi_stream
        top, bottom = ms
        top_mol = top.mol / top_split
        handle_infeasible_flow_rates(top_mol, total_mol, self.strict_infeasibility_check, 2)
        top.mol = top_mol * top_split
        bottom.mol[:] = (total_mol - top_mol) * bottom_split
        
    def update_splits(self, top_split, bottom_split):
        ms = self.multi_stream
        top, bottom = ms
        top.mol *= top_split
        bottom.mol *= bottom_split
    
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T}, P={self.P})"
    
@thermo_user
class MultiStageEquilibrium:
    """
    Create a MultiStageEquilibrium object that models counter-current 
    equilibrium stages.
    
    Parameters
    ----------
    N_stages : int
        Number of stages.
    feeds : tuple[Stream]
        All feeds, inlcuding feed with solute and solvent.
    feed_stages : tuple[int]
        Respective stage where feeds enter. Defaults to (0, -1).
    partition_data : {'IDs': tuple[str], 'K': 1d array}, optional
        IDs of chemicals in equilibrium and partition coefficients (molar 
        composition ratio of the extract over the raffinate or vapor over liquid). If given,
        The mixer-settlers will be modeled with these constants. Otherwise,
        partition coefficients are computed based on temperature and composition.
    solvent : str
        Name of main chemical in the solvent.
        
    Examples
    --------
    Simulate 2-stage extraction of methanol from water using octanol:
    
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Methanol', 'Octanol'], cache=True)
    >>> N_stages = 2
    >>> feed = tmo.Stream('feed', Water=500, Methanol=50)
    >>> solvent = tmo.Stream('solvent', Octanol=500)
    >>> stages = tmo.separations.MultiStageEquilibrium(N_stages, [feed, solvent], phases=('L', 'l'))
    >>> stages.simulate()
    >>> stages.extract.imol['Methanol'] / feed.imol['Methanol'] # Recovery
    0.83
    >>> stages.extract.imol['Octanol'] / solvent.imol['Octanol'] # Solvent stays in extract
    0.99
    >>> stages.raffinate.imol['Water'] / feed.imol['Water'] # Carrier remains in raffinate
    0.82
    
    Simulate 10-stage extraction with user defined partition coefficients:
    
    >>> import numpy as np
    >>> tmo.settings.set_thermo(['Water', 'Methanol', 'Octanol'])
    >>> N_stages = 10
    >>> feed = tmo.Stream('feed', Water=5000, Methanol=500)
    >>> solvent = tmo.Stream('solvent', Octanol=5000)
    >>> stages = tmo.separations.MultiStageEquilibrium(N_stages, [feed, solvent], phases=('L', 'l'),
    ...     partition_data={
    ...         'K': np.array([1.451e-01, 1.380e+00, 2.958e+03]),
    ...         'IDs': ('Water', 'Methanol', 'Octanol'),
    ...         'phi': 0.5899728891780545, # Initial phase fraction guess. This is optional.
    ...     }
    ... )
    >>> stages.simulate()
    >>> stages.extract.imol['Methanol'] / feed.imol['Methanol'] # Recovery
    0.99
    >>> stages.extract.imol['Octanol'] / solvent.imol['Octanol'] # Solvent stays in extract
    0.99
    >>> stages.raffinate.imol['Water'] / feed.imol['Water'] # Carrier remains in raffinate
    0.82
    
    Because octanol and water do not mix well, it may be a good idea to assume
    that these solvents do not mix at all:
        
    >>> N_stages = 20
    >>> stages = tmo.separations.MultiStageEquilibrium(N_stages, [feed, solvent], phases=('L', 'l'),
    ...     partition_data={
    ...         'K': np.array([1.38]),
    ...         'IDs': ('Methanol',),
    ...         'raffinate_chemicals': ('Water',),
    ...         'extract_chemicals': ('Octanol',),
    ...     }
    ... )
    >>> stages.simulate()
    >>> stages.extract.imol['Methanol'] / feed.imol['Methanol'] # Recovery
    0.99
    >>> stages.extract.imol['Octanol'] / solvent.imol['Octanol'] # Solvent stays in extract
    1.0
    >>> stages.raffinate.imol['Water'] / feed.imol['Water'] # Carrier remains in raffinate
    1.0
       
    Simulate with a feed at the 4th stage:
    
    >>> N_stages = 5
    >>> dilute_feed = tmo.Stream('dilute_feed', Water=100, Methanol=2)
    >>> stages = tmo.separations.MultiStageEquilibrium(N_stages, [feed, dilute_feed, solvent], 
    ...     feed_stages=[0, 3, -1],
    ...     phases=('L', 'l'),
    ...     partition_data={
    ...         'K': np.array([1.38]),
    ...         'IDs': ('Methanol',),
    ...         'raffinate_chemicals': ('Water',),
    ...         'extract_chemicals': ('Octanol',),
    ...     }
    ... )
    >>> stages.simulate()
    >>> stages.extract.imol['Methanol'] / (feed.imol['Methanol'] + dilute_feed.imol['Methanol']) # Recovery
    0.93
    
    Simulate with a 60% extract side draw at the 2nd stage:
    
    >>> N_stages = 5
    >>> stages = tmo.separations.MultiStageEquilibrium(N_stages, [feed, solvent],                         
    ...     top_side_draws=[(1, 0.6)],
    ...     phases=('L', 'l'),
    ...     partition_data={
    ...         'K': np.array([1.38]),
    ...         'IDs': ('Methanol',),
    ...         'raffinate_chemicals': ('Water',),
    ...         'extract_chemicals': ('Octanol',),
    ...     }
    ... )
    >>> stages.simulate()
    >>> (extract_side_draw,),  raffinate_side_draws = stages.get_side_draws()
    >>> (stages.extract.imol['Methanol'] + extract_side_draw.imol['Methanol']) / feed.imol['Methanol'] # Recovery
    1.0
    
    # This feature is not yet ready for users.
    # Simulate distillation column with 9 stages, a 0.673 reflux ratio, 
    # 2.57 boilup ratio, and feed at stage 4:
    
    # >>> import thermosteam as tmo
    # >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    # >>> feed = tmo.Stream('feed', Ethanol=80, Water=100, T=80.215 + 273.15)
    # >>> stages = tmo.separations.MultiStageEquilibrium(9, [feed], [4],
    # ...     specifications={0: ('Reflux', 0.673), -1: ('Boilup', 2.58)},
    # ...     phases=('g', 'l'),
    # ... )
    # >>> stages.simulate()
    # >>> stages.vapor.imol['Ethanol'] / feed.imol['Ethanol'] # Recovery
    
    """
    __slots__ = ('stages', 'multi_stream', 'iter', 'solvent', 'feeds', 'feed_stages', 'P',
                 'partition_data', 'top_side_draws', 'bottom_side_draws', 'specifications',
                 'maxiter', 'molar_tolerance', 'relative_molar_tolerance', 'use_cache',
                 '_thermo', '_iter_args', '_update_args', '_top_only')
    
    default_maxiter = 20
    default_molar_tolerance = 0.1
    default_relative_molar_tolerance = 0.001
    
    def __init__(self, N_stages, feeds, feed_stages=None, phases=None, P=101325,
                 top_side_draws=None, bottom_side_draws=None, specifications=None, partition_data=None, 
                 thermo=None, solvent=None, use_cache=None):
        thermo = self._load_thermo(thermo)
        # For VLE look for best published algorithm (don't try simple methods that fail often)
        if phases is None: phases = ('g', 'l')
        if specifications is None: specifications = ()
        if top_side_draws is None: top_side_draws = ()
        if bottom_side_draws is None: bottom_side_draws = ()
        if feed_stages is None: feed_stages = (0, -1)
        self.multi_stream = tmo.MultiStream(None, P=P, phases=phases, thermo=thermo)
        self.P = P
        phases = self.multi_stream.phases # Corrected order
        self.stages = stages = [
            StageEquilibrium(
                thermo=thermo, phases=phases,
            )
            for i in range(N_stages)
        ]
        self.solvent = solvent
        self.partition_data = partition_data
        self.feeds = feeds
        self.feed_stages = feed_stages
        self.reset_side_draws(top_side_draws, bottom_side_draws)
        top, bottom = phases
        for i in range(N_stages-1):
            stage = stages[i]
            next_stage = stages[i + 1]
            stage.feeds.append(next_stage.multi_stream[top])
            next_stage.feeds.append(stage.multi_stream[bottom])
        for feed, stage in zip(feeds, feed_stages):
            stages[stage].feeds.append(feed)
        
        #: dict[int, tuple(str, float)] Specifications for VLE by stage
        self.specifications = specifications
        for i in tuple(specifications):
            if i < 0: specifications[N_stages + i] = specifications.pop(i)
        
        #: [int] Maximum number of iterations.
        self.maxiter = self.default_maxiter

        #: [float] Molar tolerance (kmol/hr)
        self.molar_tolerance = self.default_molar_tolerance

        #: [float] Relative molar tolerance
        self.relative_molar_tolerance = self.default_relative_molar_tolerance
        
        self.use_cache = True if use_cache else False
        
    def reset_side_draws(self, top, bottom):
        N_stages = len(self.stages)
        self.top_side_draws = t = np.zeros(N_stages)
        self.bottom_side_draws = b = t.copy()
        for i, j in top: t[i] = j
        for i, j in bottom: b[i] = j
        
    def get_side_draws(self, top_streams=None, bottom_streams=None):
        top_index, = np.where(self.top_side_draws != 0.)
        if top_streams is None: top_streams = [tmo.Stream(None) for i in top_index]
        stages = self.stages
        top, bottom = self.multi_stream.phases
        for i, index in enumerate(top_index):
            s = self.top_side_draws[index]
            top_streams[i].mol = (s / (1 - s)) * stages[index].multi_stream[top].mol 
        bottom_index, = np.where(self.bottom_side_draws != 0.)
        if bottom_streams is None: bottom_streams = [tmo.Stream(None) for i in bottom_index]
        for i, index in enumerate(bottom_index):
            s = self.bottom_side_draws[index]
            bottom_streams[i].mol = (s / (1 - s)) * stages[index].multi_stream[bottom].mol 
        return top_streams, bottom_streams
    
    def _get_net_feeds(self):
        feed, *other_feeds = self.feeds
        return sum([i.mol for i in other_feeds], feed.mol)
    
    def _get_net_outlets(self):
        top_index, = np.where(self.top_side_draws != 0.)
        stages = self.stages
        top, bottom = self.multi_stream.phases
        mol = self.stages[0].multi_stream[top].mol + self.stages[-1].multi_stream[bottom].mol
        for index in top_index:
            s = self.top_side_draws[index]
            mol += (s / (1 - s)) * stages[index].multi_stream[top].mol 
        bottom_index, = np.where(self.bottom_side_draws != 0.)
        for index in bottom_index:
            s = self.bottom_side_draws[index]
            mol += (s / (1 - s)) * stages[index].multi_stream[bottom].mol 
        return mol
    
    def material_errors(self):
        errors = []
        stages = self.stages
        top_splits, bottom_splits, index = self._update_args
        columns = self.multi_stream.chemicals.IDs
        for i in range(len(stages)):
            stage = stages[i]
            top, bottom = stage.multi_stream
            mol = top.mol / top_splits[i] + bottom.mol / bottom_splits[i]
            errors.append(sum([i.mol for i in stage.feeds]) - mol)
        return pd.DataFrame(errors, columns=columns)
        
    def __len__(self):
        return len(self.stages)
    def __iter__(self):
        return iter(self.stages)
    def __getitem__(self, key):
        return self.stages[key]
    
    @property
    def extract(self):
        return self.stages[0].extract
    @property
    def raffinate(self):
        return self.stages[-1].raffinate
    @property
    def vapor(self):
        return self.stages[0].vapor
    @property
    def liquid(self):
        return self.stages[-1].liquid
    
    def correct_overall_mass_balance(self):
        top, bottom = self.multi_stream.phases
        stages = self.stages
        mol = self._get_net_outlets()
        mol[mol == 0] = 1
        factor = self._get_net_feeds() / mol
        stages[0].multi_stream[top].mol *= factor
        stages[-1].multi_stream[bottom].mol *= factor
        n = len(stages) - 1
        for i, s in enumerate(self.top_side_draws):
            if s and i != 0: stages[i].multi_stream[top].mol *= factor
        for i, s in enumerate(self.bottom_side_draws):
            if s and i != n: stages[i].multi_stream[bottom].mol *= factor
    
    # TODO: Numba all array operations here
    def update(self, top_flow_rates):
        top, bottom = self.multi_stream.phases
        stages = self.stages
        range_stages = range(len(stages))
        top_splits, bottom_splits, index = self._update_args
        for i in range_stages:
            s = stages[i].multi_stream[top]
            s.mol[index] = top_flow_rates[i]
            if self._top_only:
                IDs, flows = self._top_only
                s.imol[IDs] = flows
            s.mol *= top_splits[i]
        for i in range_stages:
            stages[i].balance_flows(top_splits[i], bottom_splits[i])
        self.correct_overall_mass_balance()
        for i in (*reversed(range_stages), *range_stages):
            stage = stages[i]
            top, bottom = stage.multi_stream
            mol = top.mol / top_splits[i] + bottom.mol / bottom_splits[i]
            mol[mol == 0] = 1
            factor = sum([i.mol for i in stage.feeds]) / mol
            stage.multi_stream.imol.data[:] *= factor
        self.correct_overall_mass_balance()
            
    def simulate(self):
        f = self.multi_stage_equilibrium_iter
        top_flow_rates = self.initialize()
        top_flow_rates = flx.conditional_wegstein(f, top_flow_rates)
        self.update(top_flow_rates)
    
    def initialize(self):
        self.iter = 1
        ms = self.multi_stream
        feeds = self.feeds
        stages = self.stages
        N_stages = len(stages)
        top_phase, bottom_phase = ms.phases
        eq = 'vle' if top_phase == 'g' else 'lle'
        ms.mix_from(feeds)
        ms.P = self.P
        if eq == 'lle':
            self.solvent = solvent = self.solvent or feeds[-1].main_chemical
            for i in stages: i.multi_stream.T = ms.T
        self._top_only = None
        data = self.partition_data
        if data:
            top_chemicals = data.get('extract_chemicals') or data.get('vapor_chemicals')
            bottom_chemicals = data.get('raffinate_chemicals') or data.get('liquid_chemicals')
        if self.use_cache and all([not i.multi_stream.isempty() for i in stages]): # Use last set of data
            if eq == 'lle':
                IDs = data['IDs'] if data else [i.ID for i in ms.lle_chemicals]
                for i in stages: 
                    i.IDs = IDs
                    i.partition_lle(self.partition_data, P=self.P, solvent=self.solvent)
                phase_ratios = np.array([i.phi / (1 - i.phi) for i in stages], dtype=float)
            else:
                IDs = data['IDs'] if data else [i.ID for i in ms.vle_chemicals]
                for i in stages: 
                    i.IDs = IDs
                    i.partition_vle(self.partition_data, P=self.P)
                phase_ratios = self.get_vle_phase_ratios()
            IDs = tuple(IDs)
            index = ms.chemicals.get_index(IDs)
            partition_coefficients = np.array([i.K for i in stages], dtype=float)
            N_chemicals = partition_coefficients.shape[1]
        else:
            if data: 
                top, bottom = ms
                IDs = data['IDs']
                K = data['K']
                phi = data.get('phi') or top.imol[IDs].sum() / ms.imol[IDs].sum()
                data['phi'] = phi = partition(ms, top, bottom, IDs, K, phi,
                                              top_chemicals, bottom_chemicals)
                index = ms.chemicals.get_index(IDs)
            elif eq == 'lle':
                lle = ms.lle
                lle(ms.T, top_chemical=solvent)
                IDs = tuple([i.ID for i in lle._lle_chemicals])
                index = ms.chemicals.get_index(IDs)
                K = lle._K
                phi = lle._phi
            else:
                P = self.P
                IDs = tuple([i.ID for i in ms.vle_chemicals])
                dp = ms.dew_point_at_P(P=P, IDs=IDs)
                T_top = dp.T
                bp = ms.bubble_point_at_P(P=P, IDs=IDs)
                T_bot = bp.T
                dT_stage = (T_bot - T_top) / N_stages
                for i in range(N_stages):
                    stages[i].multi_stream.T = T_top - i * dT_stage
                index = ms.chemicals.get_index(IDs)
                phi = 0.5
                K = bp.y / bp.z
            N_chemicals = len(index)
            phase_ratios = np.ones(N_stages) * (phi / (1 - phi))
            partition_coefficients = np.ones([N_stages, N_chemicals]) * K[np.newaxis, :]
        if data:
            if top_chemicals:
                top_flows = sum([i.imol[top_chemicals] for i in feeds])
                for i in stages: i.multi_stream.imol[top_phase, top_chemicals] = top_flows
                self._top_only = top_chemicals, top_flows
            if bottom_chemicals:
                bottom_flows = sum([i.imol[bottom_chemicals] for i in feeds])
                for i in stages: i.multi_stream.imol[bottom_phase, bottom_chemicals] = bottom_flows
        feed_flows = feed_flows = np.zeros([N_stages, N_chemicals])
        feeds = self.feeds
        feed_stages = self.feed_stages
        for feed, stage in zip(feeds, feed_stages):
            feed_flows[stage, :] += feed.mol[index]
        top_splits = 1. - self.top_side_draws
        bottom_splits = 1. - self.bottom_side_draws
        top_flow_rates = flow_rates_for_multi_stage_equilibrium(
            phase_ratios, partition_coefficients, feed_flows, -top_splits, -bottom_splits,
        )
        self._iter_args = (feed_flows, -top_splits, -bottom_splits)
        self._update_args = (top_splits, bottom_splits, index)
        return top_flow_rates
    
    def get_vle_phase_ratios(self):
        # ENERGY BALANCE
        # Intermediate stage:
        # Hv1*V1 + Hl1*L1 - Hv2*V2 - Hl0*L0 = Q1
        # Hv1*L1*B1 + Hl1*L1 - Hv2*L2*B2 - Hl0*L0 = Q1
        # Hv1*L1*B1 - Hv2*L2*B2 = Q1 + Hl0*L0 - Hl1*L1
        # Top stage:
        # Hv1*V1 - Hv2*V2 - Hl0*L0 = Q1
        # Hv1*L1*B1 + Hl1*L1 - Hv2*L2*B2 - Hl0*L0 = Q1
        # Hv1*L1*B1 - Hv2*L2*B2 = Q1 + Hl0*L0
        # Bottom stage:
        # Hv1*V1 + Hl1*L1 - Hl0*L0 = Q1
        # Hv1*L1*B1 + Hl1*L1 - Hl0*L0 = Q1
        # Hv1*L1*B1 = Q1 + Hl0*L0 - Hl1*L1
        # If top stage (or bottom stage) B1 given, do not include include in equations.
        # Second to last (or fist) stage equations do not change. 
        specifications = self.specifications
        
        stages = self.stages
        N_stages = len(stages)
        phase_ratios = np.zeros(N_stages)
        B_last = 0
        vapor_last = liquid_last = None
        for i in range(N_stages - 1, -1, -1):
            stage = stages[i]
            if i in specifications:
                name, value = specifications[i]
                B, Q = _get_specification(name, value)
            else:
                B = None
                Q = 0.
            vapor, liquid = stage.multi_stream
            if B is None:
                if B_last != 0:
                    Q += vapor_last.h * liquid_last.F_mol * B_last
                B = sum([i.H for i in stage.feeds if i is not vapor_last], Q - liquid.H) / (vapor.h * liquid.F_mol)
                if B < 0.: 
                    B = B_last / 2 + 1e-6
            phase_ratios[i] = B_last = B
            vapor_last = vapor
            liquid_last = liquid 
        return phase_ratios
    
    def multi_stage_equilibrium_iter(self, top_flow_rates):
        self.iter += 1
        self.update(top_flow_rates)
        stages = self.stages
        P = self.P
        eq = 'vle' if self.multi_stream.phases[0] == 'g' else 'lle'
        if eq == 'vle': 
            for i in stages: i.partition_vle(self.partition_data, P=self.P)
            phase_ratios = self.get_vle_phase_ratios()
            partition_coefficients = np.array([i.K for i in stages], dtype=float) 
        else:
            for i in stages: i.partition_lle(self.partition_data, P=P, solvent=self.solvent)
            phase_ratios = np.array([i.phi / (1 - i.phi) for i in stages], dtype=float)
        partition_coefficients = np.array([i.K for i in stages], dtype=float) 
        new_top_flow_rates = flow_rates_for_multi_stage_equilibrium(
            phase_ratios, partition_coefficients, *self._iter_args,
        )
        mol = top_flow_rates[0] 
        mol_new = new_top_flow_rates[0]
        mol_errors = np.abs(mol - mol_new)
        positive_index = mol_errors > 1e-16
        mol_errors = mol_errors[positive_index]
        if mol_errors.size == 0:
            mol_error = 0.
            rmol_error = 0.
        else:
            mol_error = mol_errors.max()
            if mol_error > 1e-12:
                rmol_error = (mol_errors / np.maximum.reduce([np.abs(mol[positive_index]), np.abs(mol_new[positive_index])])).max()
            else:
                rmol_error = 0.
        not_converged = (
            self.iter < self.maxiter and (mol_error > self.molar_tolerance
             or rmol_error > self.relative_molar_tolerance)
        )
        return new_top_flow_rates, not_converged


# %% General functional algorithms based on MESH equations to solve multi-stage 

@njit(cache=True)
def solve_TDMA(a, b, c, d): # Tridiagonal matrix solver
    """
    http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """
    n = d.shape[0] - 1 # number of equations minus 1
    for i in range(n):
        inext = i + 1
        m = a[i] / b[i]
        b[inext] = b[inext] - m * c[i] 
        d[inext] = d[inext] - m * d[i]
        
    b[n] = d[n] / b[n]

    for i in range(n-1, -1, -1):
        b[i] = (d[i] - c[i] * b[i+1]) / b[i]
    
    return b

@njit(cache=True)
def flow_rates_for_multi_stage_equilibrium(
        phase_ratios,
        partition_coefficients, 
        feed_flows,
        asplit,
        bsplit,
    ):
    """
    Solve b-phase flow rates for a single component across equilibrium stages with side draws. 

    Parameters
    ----------
    phase_ratios : 1d array
        Phase ratios by stage. The phase ratio for a given stage is 
        defined as F_a / F_b; where F_a and F_b are the flow rates 
        of phase a (extract or vapor) and b (raffinate or liquid) leaving the stage 
        respectively.
    partition_coefficients : Iterable[1d array]
        Partition coefficients with stages by row and components by column.
        The partition coefficient for a component in a given stage is defined 
        as x_a / x_b; where x_a and x_b are the fraction of the component in 
        phase a (extract or vapor) and b (raffinate or liquid) leaving the stage.
    feed_flows : Iterable [1d array]
        Flow rates of all components feed across stages. Shape should be 
        (N_stages, N_chemicals).
    asplit : 1d array
        Side draw split from phase a minus 1 by stage.
    bsplit : 1d array
        Side draw split from phase b minus 1 by stage.

    Returns
    -------
    flow_rates_a : 2d array
        Flow rates of phase a with stages by row and components by column.

    """
    phase_ratios[phase_ratios < 1e-16] = 1e-16
    phase_ratios[phase_ratios > 1e16] = 1e16
    N_stages, N_chemicals = partition_coefficients.shape
    phase_ratios = np.expand_dims(phase_ratios, -1)
    component_ratios = 1 / (phase_ratios * partition_coefficients)
    b = 1. +  component_ratios
    a = b.copy()
    c = b.copy()
    d = feed_flows.copy()
    for i in range(N_stages-1):
        c[i] = bsplit[i + 1]
        a[i] = asplit[i] *  component_ratios[i] 
    return solve_TDMA(a, b, c, d)