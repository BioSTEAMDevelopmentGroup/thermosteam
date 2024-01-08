# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
This module contains functions for modeling separations in unit operations.

"""
from warnings import warn
import thermosteam as tmo
import numpy as np

from .exceptions import InfeasibleRegion
from .equilibrium import phase_fraction as compute_phase_fraction

__all__ = (
    'mix_and_split',
    'adjust_moisture_content', 
    'mix_and_split_with_moisture_content',
    'handle_infeasible_flow_rates',
    'partition_coefficients',
    'vle_partition_coefficients',
    'lle_partition_coefficients',
    'partition', 'lle', 'vle', 
    'material_balance',
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
    mol = mol
    maxmol = maxmol
    infeasible_index, = np.where(mol < 0.)
    check_partition_infeasibility(infeasible_index, strict, stacklevel+1)
    mol[infeasible_index] = 0.
    infeasible_index, = np.where(mol > maxmol)
    check_partition_infeasibility(infeasible_index, strict, stacklevel+1)
    mol[infeasible_index] = maxmol[infeasible_index]


# %% Mixing, splitting, and moisture content

CAS_water = '7732-18-5'

def mix_and_split_with_moisture_content(ins, retentate, permeate,
                                        split, moisture_content, ID=None,
                                        strict=None):
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
    adjust_moisture_content(retentate, permeate, moisture_content, ID, strict)

def adjust_moisture_content(retentate, permeate, moisture_content, ID=None, strict=None):
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
        key = ('l', ID) if isinstance(retentate, tmo.MultiStream) else ID
        retentate.imol[key] = water = (dry_mass * mc/(1-mc)) / MW    
        key = ('l', ID) if isinstance(retentate, tmo.MultiStream) else ID
        permeate.imol[key] -= water - retentate_water
    else:
        retentate_moisture = retentate.imass[ID]
        dry_mass = F_mass - retentate_moisture
        key = ('l', ID) if isinstance(retentate, tmo.MultiStream) else ID
        retentate.imass[key] = moisture = dry_mass * mc/(1-mc)
        key = ('l', ID) if isinstance(retentate, tmo.MultiStream) else ID
        permeate.imass[key] -= moisture - retentate_moisture
    if permeate.imol[key] < 0:
        if strict is None: strict = True
        if strict:
            raise InfeasibleRegion(f'not enough {ID}; permeate moisture content')
        else:
            retentate.imol[key] -= permeate.imol[key]
            permeate.imol[key] = 0.

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
    phase: 'g', T: 353.94 K, P: 101325 Pa
    flow (kmol/hr): Water    3.87
                    Ethanol  6.13
    >>> liquid.show()
    Stream: liquid
    phase: 'l', T: 353.94 K, P: 101325 Pa
    flow (kmol/hr): Water    6.13
                    Ethanol  3.87
    
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
    array([0.632, 1.582])

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
     Water    0.387
     Ethanol  0.613
    >>> isplits = tmo.separations.chemical_splits(stream['g'], mixed=stream)
    >>> isplits.show()
    ChemicalIndexer:
     Water    0.387
     Ethanol  0.613
     
    """
    mixed_mol = mixed.mol.copy() if mixed else a.mol + b.mol
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
    array([0.632, 1.582])

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
    >>> s.lle(T=298.15, P=101325, top_chemical='Octanol') # Top phase is L
    >>> IDs, K = tmo.separations.lle_partition_coefficients(s['L'], s['l'])
    >>> IDs
    ('Water', 'Ethanol', 'Octanol')
    >>> round(K[2], -1) # Octanol
    3330.0

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
        Fb = bottom_flows.sum() if hasattr(bottom_flows, 'sum') else bottom_flows
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
        Fb = bottom_flows.sum() if hasattr(bottom_flows, 'sum') else bottom_flows
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
    >>> tmo.separations.lle(feed, top, bottom, top_chemical='Octanol')
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
    >>> tmo.separations.lle(feed, top, bottom, efficiency=0.99, multi_stream=ms, top_chemical='Octanol')
    >>> ms.show()
    MultiStream: ms
    phases: ('L', 'l'), T: 298.15 K, P: 101325 Pa
    flow (kmol/hr): (L) Water    3.55
                        Ethanol  0.861
                        Octanol  20
                    (l) Water    16.5
                        Ethanol  0.139
                        Octanol  0.00409
    
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
        top_l = rho_l < rho_L
        if top_l:
            top_phase = 'l'
            bottom_phase = 'L'
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
    phase: 'g', T: 353.94 K, P: 101325 Pa
    flow (kmol/hr): Water    7.75
                    Ethanol  12.3
    >>> liquid.show()
    Stream: bottom
    phase: 'l', T: 353.94 K, P: 101325 Pa
    flow (kmol/hr): Water    12.3
                    Ethanol  7.75
    
    It is also possible to save flow rate data in a multi-stream as well:
        
    >>> ms = tmo.MultiStream('ms', phases='lg')
    >>> tmo.separations.vle(feed, vapor, liquid, V=0.5, P=101325, multi_stream=ms)
    >>> ms.show()
    MultiStream: ms
    phases: ('g', 'l'), T: 353.94 K, P: 101325 Pa
    flow (kmol/hr): (g) Water    7.75
                        Ethanol  12.3
                    (l) Water    12.3
                        Ethanol  7.75
    
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
    sparse([0., 0.])
    
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
    mol_out = sum([s.mol for s in constant_outlets]).to_array()
    inlet_mols = np.array([s.mol.to_array() for s in variable_inlets]).transpose()
    if balance == 'flow':
        # Perform the following calculation: Ax = b = f - g
        # Where:
        #    A = flow rate array
        #    x = factors
        #    b = target flow rates
        #    f = output flow rates
        #    g = constant inlet flow rates

        # Solve linear equations for mass balance
        A = inlet_mols[index, :]
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
        A_ = inlet_mols.copy()
        A = inlet_mols[index, :]
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
        