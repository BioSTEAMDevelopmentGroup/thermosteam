# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the elements module from the chemicals library:
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
from chemicals import combustion
from ..utils import forward
import numpy as np

@forward(combustion)
def HHV_modified_Dulong(mass_fractions: dict[str, float]) -> float:
    r"""
    Return higher heating value [HHV; in J/g] based on the modified
    Dulong's equation [1]_.

    Parameters
    ----------
    mass_fractions : dict[str, float]
        Dictionary of atomic mass fractions [-].

    Returns
    -------
    HHV : float
        Higher heating value [J/g].

    Notes
    -----
    The heat of combustion in J/g is given by Dulong's equation [1]_:

    .. math::
        Hc (J/mol) = -(338C + 1428(H - O/8)+ 95S)

    This equation is only good for <10 wt. % Oxygen content. Variables C, H, O,
    and S are atom weight fractions.

    Examples
    --------
    Dry bituminous coal:

    >>> HHV_modified_Dulong({'C': 0.716, 'H': 0.054, 'S': 0.016, 'N': 0.016, 'O': 0.093, 'Ash': 0.105})
    -30403.9

    References
    ----------
    .. [1] Green, D. W. Waste management. In Perry`s Chemical Engineers` Handbook,
       9 ed.; McGraw-Hill Education, 2018

    """
    C = mass_fractions.get("C", 0.)
    H = mass_fractions.get("H", 0.)
    O = mass_fractions.get("O", 0.)
    S = mass_fractions.get("S", 0.)
    if O > 0.105:
        raise ValueError("Dulong's formula is only valid at 10 wt. % Oxygen "
                         f"or less ({O} given)")
    return -(338.*C  + 1428.*(H - O/8.)+ 95.*S) * 100

@forward(combustion)
def HHV_Boie(mass_fractions: dict[str, float]) -> float:
    r"""
    Return higher heating value [HHV; in J/g] based on the modified
    Boie's equation [1]_.

    Parameters
    ----------
    mass_fractions : dict[str, float]
        Dictionary of atomic mass fractions [-].

    Returns
    -------
    HHV : float
        Higher heating value [J/g].

    Notes
    -----
    The heat of combustion in J/g is given by Boie's equation [1]_:

    .. math::
        Hc (J/g) = -(347.3C + 1151H + 29N + 42S – 108O)

    Examples
    --------
    Tire rubber:

    >>> HHV_Boie({'C': 0.8033, 'H': 0.0766, 'S': 0.0087, 'N': 0.0035, 'O': 0.1079})
    -35596.6

    References
    ----------
    .. [1] Green, D. W. Waste management. In Perry`s Chemical Engineers` Handbook,
       9 ed.; McGraw-Hill Education, 2018

    """
    C = mass_fractions.get("C", 0.)
    H = mass_fractions.get("H", 0.)
    N = mass_fractions.get("N", 0.)
    O = mass_fractions.get("O", 0.)
    S = mass_fractions.get("S", 0.)
    return -(347.3 * C + 1151 * H + 29 * N + 42 * S - 108 * O) * 100

@forward(combustion)
def HHV_from_LHV(LHV: float, N_H2O: float) -> float:
    r"""
    Return the higher heating value [HHV; in J/mol] of a chemical given
    the lower heating value [LHV; in J/mol] and the number of water
    molecules formed per molecule burned.

    Parameters
    ----------
    LHV : float
        Lower heating value [J/mol].
    N_H2O : int
        Number of water molecules produced [-].

    Returns
    -------
    HHV : float
        Higher heating value [J/mol].

    Notes
    -----
    The HHV is calculated as follows:

    .. math::
        HHV = LHV - H_{vap} \cdot H_2O

    .. math::
        H_{vap} = 44011.496 \frac{J}{mol H_2O}

    .. math::
        H_2O = \frac{mol H_2O}{mol}

    Examples
    --------
    Methanol lower heat of combustion:

    >>> HHV_from_LHV(-638001.008, 2)
    -726024.0

    """
    return LHV - 44011.496 * N_H2O

@forward(combustion)
def combustion_data(formula=None, stoichiometry=None, Hf=None, MW=None,
                    method=None, missing_handling="ash", LHV=None, HHV=None):
    r"""
    Return a CombustionData object that contains the stoichiometry
    coefficients of the reactants and products, the lower and higher
    heating values [LHV, HHV; in J/mol], the heat of formation [Hf; in J/mol],
    and the molecular weight [MW; in g/mol].

    Parameters
    ----------
    formula : str, or dict[str, float], optional
        Chemical formula as a string or a dictionary of atoms and their counts.
    stoichiometry : dict[str, float], optional
        Stoichiometry of combustion reaction.
    Hf : float, optional
        Heat of formation of given chemical [J/mol].
        Required if method is "Stoichiometry".
    MW : float, optional
        Molecular weight of chemical [g/mol].
    method : "Stoichiometry", "Dulong" or "Specification", optional
        Method to estimate LHV and HHV. Use "Specification" if LHV or HHV are specified.
    missing_handling : str, optional
        How to handle compounds which do not appear in the stoichiometric
        reaction below. If 'elemental', return those atoms in the monatomic
        state; if 'Ash', converts all missing attoms to 'Ash' in the output at
        a `MW` of 1 g/mol, [-]
    LHV : float, optional
        Lower heating value of chemical [J/mol].
    HHV : float, optional
        Higher heating value of chemical [J/mol].

    Returns
    -------
    combustion_data : :class:`~chemicals.combustion.CombustionData`
        A combustion data object with the stoichiometric coefficients of
        combustion, higher heating value, heat of formation, and molecular
        weight as attributes named stoichiomery, HHV, Hf, and MW, respectively.

    Notes
    -----
    The combustion reaction is based on the following equation:

    .. math::
        C_c H_h O_o N_n S_s Br_b I_i Cl_x F_f P_p + kO_2 -> cCO_2 + \frac{b}{2}Br_2 + \frac{i}{2}I + xHCl + fHF + sSO_2 + \frac{n}{2}N_2 + \frac{p}{4}P_4O_{10} +\frac{h + x + f}{2}H_2O

    .. math::
        k = c + s + \frac{h}{4} + \frac{5P}{4} - \frac{x + f}{4} - \frac{o}{2}

    If the method is "Stoichiometry", the HHV is found using
    through an energy balance on the reaction (i.e. heat of reaction).
    If the method is "Dulong", Dulong's equation is used [1]_:

    .. math::
        Hc (J/mol) = MW \cdot (338C + 1428(H - O/8)+ 95S)

    The LHV is calculated as follows:

    .. math::
        LHV = HHV + H_{vap} \cdot H_2O

    .. math::
        H_{vap} = 44011.496 \frac{J}{mol H_2O}

    .. math::
        H_2O = \frac{mol H_2O}{mol}

    Examples
    --------
    Liquid methanol burning:

    >>> combustion_data({'H': 4, 'C': 1, 'O': 1}, Hf=-239100)
    CombustionData(stoichiometry={'CO2': 1, 'O2': -1.5, 'H2O': 2.0}, HHV=-726024.0, Hf=-239100, MW=32.04186)

    Dry bituminous coal:

    >>> cd = combustion_data({'C': 0.05961, 'H': 0.053571, 'S': 0.000499, 'N': 0.001142, 'O': 0.005813, 'Ash': 0.105})
    >>> cd.HHV
    -30401.9
    
    Find Hf from HHV for liquid methanol burning:
    
    >>> cd = combustion_data({'H': 4, 'C': 1, 'O': 1}, HHV=-726024.0)
    >>> cd.Hf
    -239100.0
    
    Find Hf from LHV for liquid methanol burning:
    
    >>> cd = combustion_data({'H': 4, 'C': 1, 'O': 1}, LHV=-638001.008)
    >>> cd.Hf
    -239100.0
    
    References
    ----------
    .. [1] Green, D. W. Waste management. In Perry`s Chemical Engineers` Handbook,
       9 ed.; McGraw-Hill Education, 2018

    """
    if formula:
        if stoichiometry:
            raise ValueError("must specify either `formula` or `stoichiometry`, not both")
        atoms = combustion.as_atoms(formula)
    if not stoichiometry:
        try:
            stoichiometry = combustion.combustion_stoichiometry(atoms, MW, missing_handling)
        except NameError:
            raise ValueError("must specify either `formula` or `stoichiometry`, none specified")
    if MW is None:
        MW = combustion.molecular_weight(atoms)
    if method:
        method = method.capitalize()
    elif Hf is None:
        if LHV is None:
            if HHV is None:
                method = 'Dulong'
                z = combustion.mass_fractions(atoms)
                if z.get('O', 0) > 0.105: method = 'Boie'
            else:
                method = "Specification"
        else:
            method = "Specification"
    else:
        method = "Stoichiometry"
    if method == 'Dulong':
        HHV = MW * HHV_modified_Dulong(combustion.mass_fractions(atoms))
        if Hf: raise ValueError("cannot specify Hf if method is 'Dulong'")
        Hf = HHV - combustion.HHV_stoichiometry(stoichiometry, 0)
    elif method == 'Boie':
        HHV = MW * HHV_Boie(combustion.mass_fractions(atoms))
        if Hf: raise ValueError("cannot specify Hf if method is 'Boie'")
        Hf = HHV - combustion.HHV_stoichiometry(stoichiometry, 0)
    elif method == 'Stoichiometry':
        if Hf is None: raise ValueError("must specify Hf if method is 'Stoichiometry'")
        HHV = combustion.HHV_stoichiometry(stoichiometry, Hf)
    elif method == 'Specification':
        if HHV is None: 
            if LHV is None:
                raise ValueError("must specify either LHV or HHV if method is 'Specification'")
            HHV = HHV_from_LHV(LHV, stoichiometry.get("H2O", 0.))
        Hf = combustion.HHV_stoichiometry(stoichiometry, 0) - HHV
    else:
        raise ValueError("method must be either 'Stoichiometric', 'Dulong', or 'Specification'; "
                         f"not {method!r}")
    return combustion.CombustionData(stoichiometry, HHV, Hf, MW)
