# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is an extension of the reaction module from the chemicals's library:
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

from chemicals import reaction

__all__ = reaction.__all__
__all__.extend([
    'Hf', 'Hf_at_phase'
])

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
    -241820.0
    
    '''
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
            Hf = Hf_at_phase(reaction.Hfg(CASRN), 'g', phase, Hvap, Hfus)
        elif phase != 'l':
            Hf = Hf_at_phase(reaction.Hfl(CASRN), 'l', phase, Hvap, Hfus)
        elif phase != 's':
            Hf = Hf_at_phase(reaction.Hfs(CASRN), 's', phase, Hvap, Hfus)
    return Hf

def Hf_at_phase(Hf, phase_ref, phase, Hvap, Hfus):
    r"""
    Return a chemical's heat of formation at given phase.
    Return None if value cannot be computed.
    
    Parameters
    ----------
    Hf : float
        Standard-state heat of formation [J/mol].
    phase : 's', 'l', or 'g'
        Phase.
    Hvap=None : float, optional
        Heat of vaporization [J/mol].
    Hfus=None : float, optional
        Heat of fusion [J/mol].
    
    Returns
    -------
    Hf : float
        Heat of formation at given phase [J/mol].
    
    """
    if Hf is None: return Hf
    elif phase_ref == phase:
        return Hf
    elif phase_ref == 'g' and phase == 'l':
        if Hvap: return Hf - Hvap
    elif phase_ref == 'g' and phase == 's':
        if Hvap and Hfus is not None: return Hf - Hvap - Hfus
    elif phase_ref == 'l' and phase == 'g':
        if Hvap: return Hf + Hvap
    elif phase_ref == 'l' and phase == 's':
        if Hfus is not None: return Hf - Hfus
    elif phase_ref == 's' and phase == 'l':
        if Hfus is not None: return Hf + Hfus
    elif phase_ref == 's' and phase == 'g':
        if Hvap and Hfus is not None: return Hf + Hvap + Hfus
    else: 
        raise ValueError("phase must be either 's', 'l', or 'g'; not '%s'" %phase)
        
import sys
module = sys.modules[__name__]
sys.modules[__name__] = reaction
reaction.Hf = Hf
reaction.Hf_at_phase = Hf_at_phase
del sys