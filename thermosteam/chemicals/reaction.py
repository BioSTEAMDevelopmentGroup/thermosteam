# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the reaction module from the chemicals's library:
# https://github.com/CalebBell/chemicals
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.

import os
from chemicals import reaction
import pandas as pd

reaction.__all__.extend([
    'Hf', 'S0', 'free_energy_at_phase'
])

folder = os.path.join(os.path.dirname(__file__), 'Reaction')
Hf_biochemicals = pd.read_csv(os.path.join(folder, 'Biochemicals Hf.tsv'), sep='\t', index_col=0)

def Hf(CASRN, phase=None, Hvap=None, Hfus=None):
    r'''
    Return a chemical's heat of formation at given phase. If no phase given,
    return the standard-state heat of formation. Return None if the data is 
    not available. 
    
    Parameters
    ----------
    CASRN : string
        CASRN [-].
    
    Returns
    -------
    Hf : float
        Standard-state heat of formation [J/mol].
    
    Other Parameters
    ----------------
    phase : 's', 'l', or 'g', optional
        Phase.
    Hvap=None : float, optional
        Heat of vaporization [J/mol].
    Hfus=None : float, optional
        Heat of fusion [J/mol].
    
    Notes
    -----
    This is a wrapper around :func:`~chemicals.reaction.Hfs`,
    :func:`~chemicals.reaction.Hfl`, and :func:`~chemicals.reaction.Hfg`.
    
    Examples
    --------
    >>> Hf(CASRN='7732-18-5')
    -241822.0
    
    '''
    if CASRN in Hf_biochemicals.index:
        Hf = Hf_biochemicals.at[CASRN, 'Hf']
        phase_found = Hf_biochemicals.at[CASRN, 'phase']
        if phase and phase_found != phase: 
            Hf = free_energy_at_phase(Hf, phase_found, phase, Hvap, Hfus)
        return Hf
    if not phase:
        for f in (reaction.Hfg, reaction.Hfl, reaction.Hfs):
            Hf = f(CASRN)
            if Hf: break 
    elif phase == 'g':
        Hf = reaction.Hfg(CASRN)
    elif phase == 'l':
        Hf = reaction.Hfl(CASRN)
    elif phase == 's':
        Hf = reaction.Hfs(CASRN)
    else:
        raise ValueError("phase must be either 's', 'l', or 'g'; not '%s'" %phase)
    if Hf is None:
        if phase != 'g':
            Hf = free_energy_at_phase(reaction.Hfg(CASRN), 'g', phase, Hvap, Hfus)
        elif phase != 'l':
            Hf = free_energy_at_phase(reaction.Hfl(CASRN), 'l', phase, Hvap, Hfus)
        elif phase != 's':
            Hf = free_energy_at_phase(reaction.Hfs(CASRN), 's', phase, Hvap, Hfus)
    return Hf
reaction.Hf = Hf

def S0(CASRN, phase=None, Svap=None, Sfus=None):
    r'''
    Return a chemical's absolute entropy of formation at given phase. If no phase given,
    return the standard-state entropy of formation. Return None if the data is 
    not available. 
    
    Parameters
    ----------
    CASRN : string
        CASRN [-].
    
    Returns
    -------
    S0 : float
        Absolute entropy of formation [J/mol].
    
    Other Parameters
    ----------------
    phase : 's', 'l', or 'g', optional
        Phase.
    Svap=None : float, optional
        Entropy of vaporization [J/mol].
    Sfus=None : float, optional
        Entropy of fusion [J/mol].
    
    Notes
    -----
    This is a wrapper around :func:`~chemicals.reaction.S0s`,
    :func:`~chemicals.reaction.S0l`, and :func:`~chemicals.reaction.S0g`.
    
    Examples
    --------
    >>> S0(CASRN='7732-18-5')
    188.8
    
    '''
    if not phase:
        for f in (reaction.S0g, reaction.S0l, reaction.S0s):
            S0 = f(CASRN)
            if S0: break 
    elif phase == 'g':
        S0 = reaction.S0g(CASRN)
    elif phase == 'l':
        S0 = reaction.S0l(CASRN)
    elif phase == 's':
        S0 = reaction.S0s(CASRN)
    else:
        raise ValueError("phase must be either 's', 'l', or 'g'; not '%s'" %phase)
    if S0 is None:
        if phase != 'g':
            S0 = free_energy_at_phase(reaction.S0g(CASRN), 'g', phase, Svap, Sfus)
        elif phase != 'l':
            S0 = free_energy_at_phase(reaction.S0l(CASRN), 'l', phase, Svap, Sfus)
        elif phase != 's':
            S0 = free_energy_at_phase(reaction.S0s(CASRN), 's', phase, Svap, Sfus)
    return S0
reaction.S0 = S0

def free_energy_at_phase(Ef, phase_ref, phase, Evap, Efus):
    r"""
    Return a chemical's free energy at given phase.
    Return None if value cannot be computed.
    
    Parameters
    ----------
    Ef : float
        Standard-state free energy of formation [J/mol].
    phase : 's', 'l', or 'g'
        Phase.
    Evap=None : float, optional
        Free energy of vaporization [J/mol].
    Efus=None : float, optional
        Free energy of fusion [J/mol].
    
    Returns
    -------
    Ef : float
        Free energy of formation at given phase [J/mol].
    
    """
    if Ef is None: return Ef
    elif phase_ref == phase:
        return Ef
    elif phase_ref == 'g' and phase == 'l':
        if Evap: return Ef - Evap
    elif phase_ref == 'g' and phase == 's':
        if Evap and Efus is not None: return Ef - Evap - Efus
    elif phase_ref == 'l' and phase == 'g':
        if Evap: return Ef + Evap
    elif phase_ref == 'l' and phase == 's':
        if Efus is not None: return Ef - Efus
    elif phase_ref == 's' and phase == 'l':
        if Efus is not None: return Ef + Efus
    elif phase_ref == 's' and phase == 'g':
        if Evap and Efus is not None: return Ef + Evap + Efus
    else: 
        raise ValueError("phase must be either 's', 'l', or 'g'; not '%s'" %phase)
reaction.free_energy_at_phase = free_energy_at_phase