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
    'MultiStageEquilbrium',
    'single_component_flow_rates_for_multi_stage_equilibrium_with_side_draws',
    'flow_rates_for_multi_stage_equilibrum_with_side_draws',
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
    top_phase = 'l'
    bottom_phase = 'L'
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
def _vle_K_phi(vapor, liquid):
    F_vapor = vapor.sum()
    F_liquid = liquid.sum()
    phi = F_vapor / (F_vapor + F_liquid)
    y = vapor / F_vapor
    x = liquid / F_liquid
    return phi, x / y 

@thermo_user
class StageEquilibrium:
    __slots__ = ('feeds', 'multi_stream',
                 'carrier_chemical', '_thermo', '_phi', '_K', '_IDs')
    
    strict_infeasibility_check = False
    
    def __init__(self, T=298.15, P=101325, feeds=None, phases=None,
                 solvent=None, thermo=None, carrier_chemical=None):
        if feeds is None: feeds = []
        self.feeds = feeds
        thermo = self._load_thermo(thermo)
        self.multi_stream = tmo.MultiStream(
            None, T=T, P=P, phases=phases or ('g', 'l'), thermo=thermo
        )
        self.carrier_chemical = carrier_chemical
        self._phi = None
        self._IDs = None
        self._K = None
    
    @property
    def extract(self):
        return self.multi_stream['l']
    @property
    def raffinate(self):
        return self.multi_stream['L']
    
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
        
    def partition(self, partition_data=None, update=True, stacklevel=1):
        ms = self.multi_stream
        if not update: data = ms.get_data()
        phases = ms.phases
        if phases == ('g', 'l'):
            ms.mix_from(self.feeds, energy_balance=True)
            eq = 'vle'
        elif phases == ('l', 'L'):
            ms.mix_from(self.feeds, energy_balance=False)
            eq = 'lle'
        else:
            raise NotImplementedError(f'equilibrium with {phases} not implemented yet')
        if partition_data:
            top, bottom = ms
            self._K = K = partition_data['K']
            self._IDs = IDs = partition_data['IDs']
            args = (IDs, K, self._phi or partition_data['phi'], 
                    partition_data.get('raffinate_chemicals'),
                    partition_data.get('extract_chemicals'),
                    self.strict_infeasibility_check, stacklevel+1)
            self._phi = partition(ms, bottom, top, *args)
        else:
            if eq == 'vle':               
                eq = ms.vle
                eq(P=self.P, H=ms.H)
                IDs = tuple([i.ID for i in eq._vle_chemicals])
                index = vle._index
                phi, K_new = _vle_K_phi(vle._vapor_mol[index], vle._liquid_mol[index])
            elif eq == 'lle':
                eq = ms.lle
                eq(self.T, top_chemical=self.carrier_chemical or self.feed.main_chemical)
                IDs_last = self._IDs
                IDs = tuple([i.ID for i in eq._lle_chemicals])
                K_new = eq._K
                phi = eq._phi
            IDs_last = self._IDs
            if IDs_last and IDs_last != IDs:
                Ks = self._K
                for ID, K in zip(IDs, K_new): Ks[IDs_last.index(ID)] = K
            else:
                self._K = K_new
            self._phi = phi
        if not update: ms.set_data(data)
        
    def balance_flows(self):
        feed, other_feeds = self.feeds
        total_mol = sum(other_feeds, feed.mol)
        ms = self.multi_stream
        top, bottom = ms
        top_mol = top.mol
        handle_infeasible_flow_rates(top_mol, total_mol, self.strict_infeasibility_check, 2)
        bottom.mol[:] = total_mol - top_mol
        
    @property
    def IDs(self):
        return self._IDs
    @property
    def phi(self):
        return self._phi
    @property
    def K(self):
        return self._K
    
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T}, P={self.P})"
    
@thermo_user
class MultiStageEquilbrium:
    """
    Create a MultiStageEquilbrium object that models a counter-current system
    of mixer-settlers for liquid-liquid extraction.
    
    Parameters
    ----------
    N_stages : int
        Number of stages.
    feeds : tuple[Stream]
        All feeds, inlcuding feed with solute and solvent.
    feed_stages : tuple[int]
        Respective stage where feeds enter. Defaults to (0, -1).
    carrier_chemical : str
        Name of main chemical in the feed (which is not selectively extracted by the solvent).
    partition_data : {'IDs': tuple[str], 'K': 1d array}, optional
        IDs of chemicals in equilibrium and partition coefficients (molar 
        composition ratio of the raffinate over the extract). If given,
        The mixer-settlers will be modeled with these constants. Otherwise,
        partition coefficients are computed based on temperature and composition.
    
    Examples
    --------
    Simulate 2-stage extraction of methanol from water using octanol:
    
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Methanol', 'Octanol'], cache=True)
    >>> N_stages = 2
    >>> feed = tmo.Stream('feed', Water=500, Methanol=50)
    >>> solvent = tmo.Stream('solvent', Octanol=500)
    >>> stages = tmo.separations.MultiStageLLE(N_stages, feed, solvent)
    >>> stages.simulate_multi_stage_equilibrium_with_side_draws()
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
    >>> stages = tmo.separations.MultiStageLLE(N_stages, feed, solvent,
    ...     partition_data={
    ...         'K': np.array([6.894, 0.7244, 3.381e-04]),
    ...         'IDs': ('Water', 'Methanol', 'Octanol'),
    ...         'phi': 0.4100271108219455 # Initial phase fraction guess. This is optional.
    ...     }
    ... )
    >>> stages.simulate_multi_stage_equilibrium_with_side_draws()
    >>> stages.extract.imol['Methanol'] / feed.imol['Methanol'] # Recovery
    0.99
    >>> stages.extract.imol['Octanol'] / solvent.imol['Octanol'] # Solvent stays in extract
    0.99
    >>> stages.raffinate.imol['Water'] / feed.imol['Water'] # Carrier remains in raffinate
    0.82
    
    Because octanol and water do not mix well, it may be a good idea to assume
    that these solvents do not mix at all:
        
    >>> N_stages = 20
    >>> stages = tmo.separations.MultiStageLLE(N_stages, feed, solvent,
    ...     partition_data={
    ...         'K': np.array([0.7244]),
    ...         'IDs': ('Methanol',),
    ...         'raffinate_chemicals': ('Water',),
    ...         'extract_chemicals': ('Octanol',),
    ...     }
    ... )
    >>> stages.simulate_multi_stage_equilibrium_with_side_draws()
    >>> stages.extract.imol['Methanol'] / feed.imol['Methanol'] # Recovery
    0.99
    >>> stages.extract.imol['Octanol'] / solvent.imol['Octanol'] # Solvent stays in extract
    1.0
    >>> stages.raffinate.imol['Water'] / feed.imol['Water'] # Carrier remains in raffinate
    1.0
        
    """
    __slots__ = ('stages', 'index', 'multi_stream', 
                 'carrier_chemical', 'extract_flow_rates', 
                 'partition_data', 'feed_flows', 'raffinate_side_draw_splits',
                 'extract_side_draw_splits', '_asplit', '_bsplit',
                 '_thermo', '_K_init')
    
    def __init__(self, N_stages, feeds, feed_stages=(0, -1), phases=None, carrier_chemical=None,
                 thermo=None, partition_data=None):
        thermo = self._load_thermo(thermo)
        if phases is None: phases = ('g', 'l')
        self.multi_stream = tmo.MultiStream(None, phases=phases, thermo=thermo)
        phases = self.multi_stream.phases # Corrected order
        self.stages = stages = [
            StageEquilibrium(
                thermo=thermo, phases=phases, carrier_chemical=carrier_chemical,
            )
            for i in range(N_stages)
        ]
        self.carrier_chemical = carrier_chemical
        self.partition_data = partition_data
        top, bottom = phases
        for i in range(N_stages-1):
            stage = stages[i]
            next_stage = stages[i + 1]
            next_stage.feeds.extend([stage.multi_stream[bottom], next_stage.multi_stream[top]])
        for feed, stage in (feeds, feed_stages):
            stages[stage].feeds.append(feed)
        
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
    
    def update_multi_stage_equilibrium_with_side_draws(self, top_flow_rates):
        index = self.index
        top = self.multi_stream.phases[0]
        for stage, top_flow in zip(self.stages, top_flow_rates):
            stage.multi_stream[top].mol[index] = top_flow
            
    def simulate_multi_stage_equilibrium_with_side_draws(self):
        f = self.multi_stage_equilibrium_with_side_draws_iter
        top_flow_rates = self.initialize_multi_stage_equilibrium_with_side_draws()
        self.top_flow_rates = top_flow_rates = flx.wegstein(f, top_flow_rates, xtol=0.1, maxiter=10, checkiter=False)
        self.update_multi_stage_equilibrium_with_side_draws(top_flow_rates)
        for i in self.stages: i.balance_flows()
    
    def initialize_multi_stage_equilibrium_with_side_draws(self):
        multi_stream = self.multi_stream
        multi_stream.mix_from(self.feeds)
        top, bottom = multi_stream
        stages = self.stages
        N_stages = len(stages)
        T = multi_stream.T
        for i in stages: 
            i.multi_stream.T = T
            i.raffinate.empty()
        if self.partition_data: 
            data = self.partition_data
            IDs = data['IDs']
            K = data['K']
            phi = data.get('phi') or top.imol[IDs].sum() / multi_stream.imol[IDs].sum()
            top_chemicals = data.get('top_chemicals')
            bottom_chemicals = data.get('bottom_chemicals')
            data['phi'] = phi = partition(multi_stream, bottom, top, IDs, K, phi,
                                          bottom_chemicals, top_chemicals)
            if top_chemicals:
                phase = multi_stream.phases[0]
                top_flows = multi_stream.imol[top_chemicals]
                for i in stages: i.multi_stream.imol[phase, top_chemicals] = top_flows
            if bottom_chemicals:
                phase = multi_stream.phases[1]
                bottom_flows = multi_stream.imol[bottom_chemicals]
                for i in stages: i.multi_stream.imol[phase, top_chemicals] = bottom_flows
        else:
            lle = multi_stream.lle
            lle(multi_stream.T, top_chemical=self.carrier_chemical or bottom.main_chemical)
            IDs = tuple([i.ID for i in lle._lle_chemicals])
            K = lle._K
            phi = lle._phi
        self._K_init = K
        index = multi_stream.chemicals.get_index(IDs)
        phase_fractions = np.ones(N_stages) * phi
        N_chemicals = K.size
        partition_coefficients = np.ones([N_chemicals, N_stages]) * K[:, np.newaxis]
        self.feed_flows = feed_flows = np.zeros([N_chemicals, N_stages])
        feed_flows[:, 0] = self.feed.mol[index]
        feed_flows[:, N_stages - 1] = self.solvent.mol[index]
        self.raffinate_side_draw_splits = np.zeros(N_stages) # TODO: Allow user to specify side draws
        self.extract_side_draw_splits = np.zeros(N_stages)
        self._asplit = asplit = 1 - self.raffinate_side_draw_splits
        self._bsplit = bsplit = 1 - self.extract_side_draw_splits
        extract_flow_rates = flow_rates_for_multi_stage_equilibrum_with_side_draws(
            phase_fractions, partition_coefficients, feed_flows, asplit, bsplit
        )
        self.index = index 
        return extract_flow_rates
    
    def multi_stage_equilibrium_with_side_draws_iter(self, extract_flow_rates):
        self.update_multi_stage_equilibrium_with_side_draws(extract_flow_rates)
        stages = self.stages
        for i in stages: i.balance_flows()
        for i in stages: i.partition(self.partition_data, update=False)
        K = np.array([i.K for i in stages], dtype=float)
        K = np.transpose(K) 
        phi = np.array([i.phi for i in stages])
        extract_flow_rates = flow_rates_for_multi_stage_equilibrum_with_side_draws(
            phi, K, self.feed_flows, self._asplit, self._bsplit,
        )
        return extract_flow_rates

# %% General functional algorithms based on MESH equations to solve multi-stage 

@njit(cache=True)
def single_component_flow_rates_for_multi_stage_equilibrium_with_side_draws(
        N_stages,
        phase_ratios,
        partition_coefficients, 
        feed_flows,
        asplit,
        bsplit,
    ):
    """
    Solve flow rates for a single component across equilibrium stages
    with side draws. 

    Parameters
    ----------
    N_stages : int
        Number of stages.
    phase_ratios : 1d array
        Phase ratios by stage. The phase ratio for a given stage is 
        defined as F_a / F_b; where F_a and F_b are the flow rates 
        of phase a (raffinate or liquid) and b (extract or vapor) leaving the stage 
        respectively.
    partition_coefficients : 1d array
        Partition coefficients by stage. The partition coefficient for a given
        stage is defined as x_a / x_b; where x_a and x_b are the fraction of
        the component in each phase leaving the stage.
    feed_flows : Iterable [1d array]
        Flow rates of all components feed across stages. Shape should be 
        N_chemicals by N_stages.
    asplit : 1d array
        1 minus side draw split from phase a by stage.
    bsplit : 1d array
        1 minus side draw split from phase b by stage.

    Returns
    -------
    extract_flow_rates : 1d array
        Extract component flow rates by stage.

    """
    component_ratios = phase_ratios * partition_coefficients
    A = np.eye(N_stages) * (1 + component_ratios) 
    for i in range(N_stages-1):
        i_next = i + 1
        A[i, i_next] = -bsplit[i_next]
        A[i_next, i] = -asplit[i] * component_ratios[i]
    return np.linalg.solve(A, feed_flows)

@njit(cache=True)
def flow_rates_for_multi_stage_equilibrum_with_side_draws(
        phase_fractions,
        partition_coefficients, 
        feed_flows,
        asplit,
        bsplit,
    ):
    """
    Solve flow rates for a single component across a equilibrium stages with side draws. 

    Parameters
    ----------
    phase_fractions : 1d array
        Phase fractions by stage. The phase fraction for a given stage is 
        defined as F_a / (F_a + F_b); where F_a and F_b are the flow rates 
        of phase a (raffinate or liquid) and b (extract or vapor) leaving the stage 
        respectively.
    partition_coefficients : Iterable[1d array]
        Partition coefficients with components by row and stages by column.
        The partition coefficient for a component in a given stage is defined 
        as x_a / x_b; where x_a and x_b are the fraction of the component in 
        phase a (raffinate or liquid) and b (extract or vapor) leaving the stage.
    feed_flows : Iterable [1d array]
        Flow rates of all components feed across stages. Shape should be 
        N_chemicals by N_stages.
    asplit : 1d array
        1 minus side draw split from phase a by stage.
    bsplit : 1d array
        1 minus side draw split from phase b by stage.

    Returns
    -------
    extract_flow_rates : 2d array
        Extract flow rates with stages by row and components by column.

    """
    phase_fractions[phase_fractions < 1e-16] = 1e-16
    phase_fractions[phase_fractions > 1. - 1e-16] = 1. - 1e-16
    phase_ratios = phase_fractions / (1. - phase_fractions)
    N_chemicals, N_stages = partition_coefficients.shape
    extract_flow_rates = np.zeros((N_stages, N_chemicals))
    for i in range(N_chemicals):
        flows_by_stage = single_component_flow_rates_for_multi_stage_equilibrium_with_side_draws(
                N_stages, phase_ratios, partition_coefficients[i], feed_flows[i], asplit, bsplit,
        ) 
        extract_flow_rates[:, i] = flows_by_stage    
    return extract_flow_rates
