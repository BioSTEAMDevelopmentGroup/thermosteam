# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
This module contains functions for modeling separations in unit operations.

"""
import thermosteam as tmo
import flexsolve as flx
import numpy as np
from .utils import thermo_user
from .exceptions import InfeasibleRegion
from .equilibrium import phase_fraction

__all__ = (
    'split', 'mix_and_split',
    'adjust_moisture_content', 
    'mix_and_split_with_moisture_content',
    'partition_coefficients',
    'vle_partition_coefficients',
    'lle_partition_coefficients',
    'partition', 'lle', 'vle', 
    'material_balance',
    'StageLLE',
    'MultiStageLLE',
    'single_component_flow_rates_for_multi_stage_lle_without_side_draws',
    'flow_rates_for_multi_stage_extration_without_side_draws',
    'chemical_splits'
)


# %% Mixing, splitting, and moisture content

CAS_water = '7732-18-5'

def mix_and_split_with_moisture_content(ins, retentate, permeate,
                                        split, moisture_content):
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
    adjust_moisture_content(retentate, permeate, moisture_content)

def adjust_moisture_content(retentate, permeate, moisture_content):
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
    retentate_water = retentate.imol[CAS_water]
    dry_mass = F_mass - 18.01528 * retentate_water
    retentate.imol[CAS_water] = water = (dry_mass * mc/(1-mc))/18.01528
    permeate.imol[CAS_water] -= (water - retentate_water)
    if permeate.imol[CAS_water] < 0:
        raise InfeasibleRegion('not enough water; permeate moisture content')

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
    bottom.copy_like(top)
    top_mol = top.mol
    top_mol[:] *= split
    bottom.mol[:] -= top_mol

def split(feed, top, bottom, split):
    """
    Run splitter mass and energy balance with mixing all input streams.
    
    Parameters
    ----------
    feed : Stream
        Inlet fluid.
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
    >>> feed = tmo.Stream(Water=35, Ethanol=10)
    >>> split = 0.8
    >>> effluent_a = tmo.Stream('effluent_a')
    >>> effluent_b = tmo.Stream('effluent_b')
    >>> tmo.separations.split(feed, effluent_a, effluent_b, split)
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
    if feed is not top: top.copy_like(feed)
    bottom.copy_like(top)
    top_mol = top.mol
    top_mol[:] *= split
    bottom.mol[:] -= top_mol

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
    return top.get_normalized_mol(IDs) / bottom.get_normalized_mol(IDs)

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

def partition(feed, top, bottom, IDs, K, phi=None):
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
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', tmo.Chemical('O2', phase='g')], cache=True)
    >>> IDs = ('Water', 'Ethanol')
    >>> K = np.array([0.629, 1.59])
    >>> feed = tmo.Stream('feed', Water=20, Ethanol=20, O2=0.1)
    >>> top = tmo.Stream('top')
    >>> bottom = tmo.Stream('bottom')
    >>> tmo.separations.partition(feed, top, bottom, IDs, K)
    0.5002512677600628
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

    """
    feed_mol = feed.mol
    mol = feed.imol[IDs]
    F_mol = mol.sum()
    z_mol = mol / F_mol
    phi = phase_fraction(z_mol, K, phi)
    x = z_mol / (phi * K + (1 - phi))
    bottom.empty()
    bottom.imol[IDs] = x * (1 - phi) * F_mol
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
        
# %% General functional algorithms based on MESH equations to solve multi-stage 
# liquid-liquid equilibrium.

@thermo_user
class StageLLE:
    __slots__ = ('feed', 'raffinate', 'solvent', 'extract', 'multi_stream',
                 'carrier_chemical', '_thermo', '_phi', '_K', '_IDs')
    
    def __init__(self, T=298.15, P=101325, feed=None,
                 solvent=None, thermo=None, carrier_chemical=None):
        self.feed = feed
        self.solvent = solvent
        thermo = self._load_thermo(thermo)
        self.multi_stream = multi_stream = tmo.MultiStream(
            None, T=T, P=P, phases=('l', 'L'), thermo=thermo
        )
        self.raffinate = multi_stream['l']
        self.extract = multi_stream['L']
        self.carrier_chemical = carrier_chemical
        
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
        
    def partition(self, partition_data=None):
        multi_stream = self.multi_stream
        multi_stream.mix_from([self.feed, self.solvent])
        lle = multi_stream.lle
        if partition_data:
            top = multi_stream['l']
            bottom = multi_stream['L']
            lle = multi_stream.lle
            phi = partition_data['phi']
            self._K = K = partition_data['K']
            self._IDs = IDs = partition_data['IDs']
            self._phi = partition(multi_stream, top, bottom, IDs, K, phi)
        else:
            lle(self.T, top_chemical=self.carrier_chemical or self.feed.main_chemical)
            self._IDs = tuple([i.ID for i in lle._lle_chemicals])
            self._phi = lle._phi
            self._K = lle._K 
        
    def balance_raffinate_flows(self):
        total_mol = self.feed.mol + self.solvent.mol
        extract_mol = self.extract.mol
        self.raffinate.mol[:] = total_mol - extract_mol
        
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
class MultiStageLLE:
    """
    Create a MultiStageLLE object that models a counter-current system
    of mixer-settlers for liquid-liquid extraction.
    
    Parameters
    ----------
    N_stages : int
        Number of stages.
    feed : Stream
        Feed with solute.
    solvent : Stream
        Solvent to contact feed and recover solute.
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
    >>> stages.simulate_multi_stage_lle_without_side_draws()
    >>> stages.raffinate.show()
    Stream: 
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water     413
                     Methanol  8.4
                     Octanol   0.1
    >>> stages.extract.show()
    Stream: 
     phase: 'L', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water     87.
                     Methanol  41.
                     Octanol   500
    
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
    >>> stages.simulate_multi_stage_lle_without_side_draws()
    >>> stages.raffinate.show()
    Stream: 
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water     4.1e+03
                     Methanol  1.2
                     Octanol   1.5
    >>> stages.extract.show()
    Stream: 
     phase: 'L', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water     871
                     Methanol  499
                     Octanol   5e+03
    
    """
    __slots__ = ('stages', 'index', 'multi_stream', 
                 'carrier_chemical', 'extract_flow_rates', 
                 'partition_data', '_thermo')
    
    def __init__(self, N_stages, feed, solvent, carrier_chemical=None,
                 thermo=None, partition_data=None):
        thermo = self._load_thermo(thermo)
        self.multi_stream = tmo.MultiStream(None, phases=('l', 'L'), thermo=thermo)
        self.stages = stages = [StageLLE(thermo=thermo) for i in range(N_stages)]
        self.carrier_chemical = carrier_chemical
        self.partition_data = partition_data
        for i in range(N_stages-1):
            stage = stages[i]
            next_stage = stages[i + 1]
            next_stage.feed = stage.raffinate
            stage.solvent = next_stage.extract
        stages[0].feed = feed
        stages[-1].solvent = solvent
        
    def __len__(self):
        return len(self.stages)
    def __iter__(self):
        return iter(self.stages)
    def __getitem__(self, key):
        return self.stages[key]
    
    @property
    def feed(self):
        return self.stages[0].feed
    @property
    def extract(self):
        return self.stages[0].extract    
    @property
    def solvent(self):
        return self.stages[-1].solvent
    @property
    def raffinate(self):
        return self.stages[-1].raffinate
    
    def update_multi_stage_lle_without_side_draws(self, extract_flow_rates):
        if (extract_flow_rates < 0.).any(): raise RuntimeError('negative flow rate')
        index = self.index
        for stage, extract_flow in zip(self.stages, extract_flow_rates):
            stage.extract.mol[index] = extract_flow
            stage.balance_raffinate_flows()
    
    def simulate_multi_stage_lle_without_side_draws(self):
        f = self.multi_stage_lle_without_side_draws_iter
        if hasattr(self, 'extract_flow_rates'):
            extract_flow_rates = self.extract_flow_rates
        else:
            extract_flow_rates = self.initialize_multi_stage_lle_without_side_draws()
        extract_flow_rates = flx.wegstein(f, extract_flow_rates, xtol=0.1, maxiter=10, checkiter=False)
        self.extract_flow_rates = extract_flow_rates
        self.update_multi_stage_lle_without_side_draws(extract_flow_rates)
    
    def initialize_multi_stage_lle_without_side_draws(self):
        feed = self.feed
        solvent = self.solvent
        multi_stream = self.multi_stream
        multi_stream.mix_from([feed, solvent]) 
        stages = self.stages
        N_stages = len(stages)
        if self.partition_data:
            data = self.partition_data
            top = multi_stream['l']
            bottom = multi_stream['L']
            IDs = data['IDs']
            K = data['K']
            phi = data['phi']
            data['phi'] = phi = partition(multi_stream, top, bottom, IDs, K, phi)
        else:
            lle = multi_stream.lle
            lle(multi_stream.T, top_chemical=self.carrier_chemical or feed.main_chemical)
            IDs = tuple([i.ID for i in lle._lle_chemicals])
            K = lle._K
            phi = lle._phi
        index = multi_stream.chemicals.get_index(IDs)
        phase_fractions = np.ones(N_stages) * phi
        partition_coefficients = np.ones([K.size, N_stages]) * K[:, np.newaxis]
        extract_flow_rates = flow_rates_for_multi_stage_extration_without_side_draws(
            N_stages, phase_fractions, partition_coefficients, feed.mol[index], solvent.mol[index]
        )
        self.index = index 
        return extract_flow_rates
    
    def multi_stage_lle_without_side_draws_iter(self, extract_flow_rates):
        self.update_multi_stage_lle_without_side_draws(extract_flow_rates)
        stages = self.stages
        for i in stages: i.partition(self.partition_data)
        K = np.transpose([i.K for i in stages]) 
        phi = np.array([i.phi for i in stages])
        index = self.index
        stages = self.stages
        N_stages = len(stages)
        extract_flow_rates = flow_rates_for_multi_stage_extration_without_side_draws(
            N_stages, phi, K, self.feed.mol[index], self.solvent.mol[index]
        )
        return extract_flow_rates
        
@flx.njitable(cache=True)
def single_component_flow_rates_for_multi_stage_lle_without_side_draws(
        N_stages,
        phase_ratios,
        partition_coefficients, 
        feed, 
        solvent
    ):
    """
    Solve flow rates for a single component across a multi stage liquid-liquid
    extraction operation without side draws. 

    Parameters
    ----------
    N_stages : int
        Number of stages.
    phase_ratios : 1d array
        Phase ratios by stage. The phase ratio for a given stage is 
        defined as F_l / F_L; where F_l and F_L are the flow rates 
        of phase l (raffinate) and L (extract) leaving the stage respectively.
    partition_coefficients : 1d array
        Partition coefficients by stage. The partition coefficient for a given
        stage is defined as x_l / x_L; where x_l and x_L are the fraction of
        the component in phase l (raffinate) and L (extract) leaving the stage.
    feed : float
        Component flow rate in feed entering stage 1.
    solvent : float
        Component flow rate in solvent entering stage N.

    Returns
    -------
    extract_flow_rates : 1d array
        Extract component flow rates by stage.

    """
    component_ratios = phase_ratios * partition_coefficients
    A = np.eye(N_stages) * (1 + component_ratios) 
    for i in range(N_stages-1):
        i_next = i + 1
        A[i, i_next] = -1
        A[i_next, i] = -component_ratios[i]
    b = np.zeros(N_stages)
    b[0] = feed
    b[-1] += solvent
    return np.linalg.solve(A, b)

@flx.njitable(cache=True)
def flow_rates_for_multi_stage_extration_without_side_draws(
        N_stages,
        phase_fractions,
        partition_coefficients, 
        feed, 
        solvent
    ):
    """
    Solve flow rates for a single component across a multi stage liquid-liquid
    extraction without side draws. 

    Parameters
    ----------
    N_stages : int
        Number of stages.
    phase_fractions : 1d array
        Phase fractions by stage. The phase fraction for a given stage is 
        defined as F_l / (F_l + F_L); where F_l and F_L are the flow rates 
        of phase l (raffinate) and L (extract) leaving the stage respectively.
    partition_coefficients : Iterable[1d array]
        Partition coefficients with components by row and stages by column.
        The partition coefficient for a component in a given stage is defined 
        as x_l / x_L; where x_l and x_L are the fraction of the component in 
        phase l (raffinate) and L (extract) leaving the stage.
    feed : Iterable[float]
        Flow rates of all components in feed entering stage 1.
    solvent : Iterable[float]
        Flow rates of all components in solvent entering stage N.

    Returns
    -------
    extract_flow_rates : 2d array
        Extract flow rates with stages by row and components by column.

    """
    phase_ratios = phase_fractions / (1. - phase_fractions)
    N_chemicals = feed.size
    extract_flow_rates = np.zeros((N_stages, N_chemicals))
    for i in range(N_chemicals):
        flows_by_stage = single_component_flow_rates_for_multi_stage_lle_without_side_draws(
                N_stages, phase_ratios, partition_coefficients[i], feed[i], solvent[i]
        ) 
        extract_flow_rates[:, i] = flows_by_stage
    return extract_flow_rates
