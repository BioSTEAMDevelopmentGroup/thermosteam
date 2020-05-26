# -*- coding: utf-8 -*-
'''
All data and methods for estimating a chemical's heat of formation.

References
----------
.. [1] Albahri, Tareq A., and Abdulla F. Aljasmi. "SGC Method for
   Predicting the Standard Enthalpy of Formation of Pure Compounds from
   Their Molecular Structures." Thermochimica Acta 568
   (September 20, 2013): 46-60. doi:10.1016/j.tca.2013.06.020.
.. [2] Ruscic, Branko, Reinhardt E. Pinzon, Gregor von Laszewski, Deepti
    Kodeboyina, Alexander Burcat, David Leahy, David Montoy, and Albert F.
    Wagner. "Active Thermochemical Tables: Thermochemistry for the 21st
    Century." Journal of Physics: Conference Series 16, no. 1
    (January 1, 2005): 561. doi:10.1088/1742-6596/16/1/078.
.. [3] Frenkelʹ, M. L, Texas Engineering Experiment Station, and
    Thermodynamics Research Center. Thermodynamics of Organic Compounds in
    the Gas State. College Station, Tex.: Thermodynamics Research Center,
    1994.   
    
'''

__all__ = ('heat_of_formation', 
           'heat_of_formation_liquid',
           'heat_of_formation_gas',
)

from ..exceptions import UndefinedPhase, InvalidMethod
from .data import (heat_of_formation_sources,
                   heat_of_formation_gas_sources,
                   heat_of_formation_liquid_sources,
                   get_from_data_sources
)

# TODO: Add the rest of the metals, and fix graphite in data base.
# Currently, graphite points to methane.
elemental_gas_standard_states = {'1333-74-0', '7727-37-9',
                                 '7782-44-7', '7782-41-4', 
                                 '7782-50-5', '7440-59-7'
}
elemental_liquid_standard_states = {'7726-95-6', '7439-97-6'
}
elemental_solid_standard_states = {'100320-09-0', '7553-56-2',
                                   '7440-21-3'
}
standard_elemental_states = (elemental_gas_standard_states 
                             | elemental_liquid_standard_states 
                             | elemental_solid_standard_states)

def heat_of_formation(CASRN, phase_ref,
                      Hvap_298K=None, Hfus=None,
                      method='Any'):
    r'''
    Return a chemical's standard-phase heat of formation.
    The lookup is based on CASRNs. Return None if the data is not available.

    Function has data for 571 chemicals.

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
    method : string, optional
        The method name to use. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.
    phase_ref : {'s', 'l', 'g'}
        Reference phase.
    Hvap_298K=None : float, optional
        Heat of vaporization [J/mol].
    Hfus=None : float, optional
        Heat of fusion [J/mol].

    Notes
    -----
    Multiple sources of information are available for this function:

        * 'API_TDB', a compilation of heats of formation of unspecified phase.
          Not the original data, but as reproduced in [1]_. Some chemicals with
          duplicated CAS numbers were removed.
        * 'ATCT_L', the Active Thermochemical Tables version 1.112. [2]_
        * 'ATCT_G', the Active Thermochemical Tables version 1.112. [2]_
        * 'TRC', from a 1994 compilation. [3]_
        * 'Other', from NIST or calculated by Joback method. 

    Examples
    --------
    >>> Hf(CASRN='7732-18-5')
    -241820.0

    '''
    if phase_ref == 'l':
        if CASRN in elemental_liquid_standard_states: return 0.
    elif phase_ref == 'g':
        if CASRN in elemental_gas_standard_states: return 0. 
    elif phase_ref == 's':
        if CASRN in elemental_solid_standard_states: return 0.
    try: Hf = heat_of_formation_gas(CASRN, method)
    except InvalidMethod: pass
    if Hf:
        Hf = Hf_at_phase_ref(Hf, 'g', phase_ref, Hvap_298K, Hfus)
        if Hf: return Hf
    try: Hf = heat_of_formation_liquid(CASRN, method)
    except InvalidMethod: pass
    if Hf:
        Hf = Hf_at_phase_ref(Hf, 'l', phase_ref, Hvap_298K, Hfus)
        if Hf: return Hf        
    try: Hf, phase = get_from_data_sources(heat_of_formation_sources, CASRN,
                                           ['Hf_298K', 'phase'], method)     
    except InvalidMethod: pass
    if Hf:
        Hf = Hf_at_phase_ref(Hf, phase, phase_ref, Hvap_298K, Hfus)
        if Hf: return Hf        
        
def Hf_at_phase_ref(Hf, phase, phase_ref, Hvap_298K, Hfus):
    if phase == phase_ref: return Hf
    elif phase == 'g' and phase_ref == 'l':
        if Hvap_298K: return Hf - Hvap_298K
    elif phase == 'g' and phase_ref == 's':
        if Hvap_298K and Hfus: return Hf - Hvap_298K - Hfus
    elif phase == 'l' and phase_ref == 'g':
        if Hvap_298K: return Hf + Hvap_298K
    elif phase == 'l' and phase_ref == 's':
        if Hfus: return Hf - Hfus
    elif phase == 's' and phase_ref == 'l':
        if Hfus: return Hf + Hfus
    elif phase == 's' and phase_ref == 'g':
        if Hvap_298K and Hfus: return Hf + Hvap_298K + Hfus
    else: raise UndefinedPhase(phase)

def heat_of_formation_liquid(CASRN, method='Any'):
    r'''
    Return a chemical's liquid standard phase heat of formation. 
    The lookup is based on CASRNs. Selects the only data source available,
    Active Thermochemical Tables (l), if the chemical is in it.
    Return None if the data is not available.

    Function has data for 34 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Hf_l : float
        Liquid standard-state heat of formation, [J/mol]

    Other Parameters
    ----------------
    method : string, optional
        The method name to use. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.

    Notes
    -----
    Only one source of information is available to this function. It is:

        * 'ATCT_L', the Active Thermochemical Tables version 1.112. [2]_

    Examples
    --------
    >>> heat_of_formation_liquid('67-56-1')
    -238400.0

    '''
    return get_from_data_sources(heat_of_formation_liquid_sources,
                                 CASRN, 'Hf_298K', method)


def heat_of_formation_gas(CASRN, method='Any'):
    r'''
    Retrieve a chemical's gas heat of formation. Lookup is based on CASRNs. 
    Automatically select a data source to use if no Method is provided.
    Return None if the data is not available.

    Prefered sources are 'Active Thermochemical Tables (g)' for high accuracy,
    and 'TRC' for less accuracy but more chemicals.
    Function has data for approximately 2000 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Hf_g : float
        Gas phase heat of formation, [J/mol]

    Other Parameters
    ----------------
    method : string, optional
        The method name to use. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.
    
    Notes
    -----
    Sources are:

        * 'ATCT_G', the Active Thermochemical Tables version 1.112. [2]_
        * 'TRC', from a 1994 compilation. [3]_

    Examples
    --------
    >>> Hf_g('67-56-1')
    -200700.0

    '''
    return get_from_data_sources(heat_of_formation_gas_sources,
                                 CASRN, 'Hf_298K', method)
   