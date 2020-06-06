# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# A significant portion of this module originates from:
# Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
# 
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/thermo/blob/master/LICENSE.txt for details.
"""
All data and methods for the internal retrieval of data from CAS numbers.

"""
from collections import namedtuple
from .readers import CASDataReader, load_json, to_nums, get_from_data_sources
import os

__all__ = ('get_from_data_sources',
           'critical_data_IUPAC',
           'critical_data_Matthews',
           'critical_data_CRC',
           'critical_data_PSRKR4',
           'critical_data_PassutDanner',
           'critical_data_Yaws',
           'critical_data_sources',
           'tiple_point_Staveley_data',
           'triple_point_sources',
           'dipole_data_Poling',
           'dipole_data_CCCBDB',
           'dipole_data_Muller',
           'dipole_data_sources',
           'Psat_data_WagnerMcGarry',
           'Psat_data_Wagner',
           'Psat_data_Antoine',
           'Psat_data_AntoineExtended',
           'Psat_data_Perrys2_8',
           'Psat_data_VDI_PPDS_3',
           'Cn_data_Poling',
           'Cn_data_TRC_gas',
           'Cn_data_CRC_standard',
           'Cn_data_PerryI',
           'zabransky_dict_sat_s',
           'zabransky_dict_sat_p',
           'zabransky_dict_const_s',
           'zabransky_dict_const_p',
           'zabransky_dict_iso_s',
           'zabransky_dict_iso_p',
           'Hf_data_API_TDB',
           'Hf_data_ATcT_l',
           'Hf_data_ATcT_g',
           'heat_of_formation_sources',
           'heat_of_formation_solid_sources',
           'heat_of_formation_liquid_sources',
           'heat_of_formation_gas_sources',
           'sigma_data_Mulero_Cachadina',
           'sigma_data_Jasper_Lange',
           'sigma_data_Somayajulu',
           'sigma_data_Somayajulu_2',
           'sigma_data_VDI_PPDS_11',
           # 'kappa_data_Perrys2_314',
           'kappa_data_Perrys2_315',
           'kappa_data_VDI_PPDS_9',
           # 'kappa_data_VDI_PPDS_10',
           'mu_data_Dutt_Prasad',
           'mu_data_VN3',
           'mu_data_VN2',
           # 'mu_data_VN2E_data',
           'mu_data_Perrys2_313',
           'mu_data_Perrys2_312',
           'mu_data_VDI_PPDS_7',
           'mu_data_VDI_PPDS_8',
           'V_data_COSTALD',
           'V_data_SNM0',
           'V_data_Perry_l',
           'V_data_VDI_PPDS_2',
           'V_data_CRC_inorg_l',
           'V_data_CRC_inorg_l_const',
           'V_data_CRC_inorg_s_const',
           'V_data_CRC_virial',
           'permittivity_data_CRC',
           'Lange_cond_pure',
           'Laliberte_Density_ParametersDict',
           'Laliberte_Viscosity_ParametersDict',
           'Laliberte_Heat_Capacity_ParametersDict',
           'inorganic_data_CRC',
           'organic_data_CRC', 
           'VDI_saturation_dict', 
           'VDI_tabular_data',
)


# %% VDI saturation data

read = CASDataReader('Misc')

### CRC Handbook general tables
inorganic_data_CRC = read('Physical Constants of Inorganic Compounds.csv')
organic_data_CRC = read('Physical Constants of Organic Compounds.csv')

### VDI Saturation
VDI_saturation_dict = load_json(read.folder, 'VDI Saturation Compounds Data.json')
#: Read in a dict of assorted chemical properties at saturation for 58
#: industrially important chemicals, from:
#: Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2E. Berlin : Springer, 2010.
#: This listing is the successor to that in:
#: Schlunder, Ernst U, and International Center for Heat and Mass Transfer.
#: Heat Exchanger Design Handbook. Washington: Hemisphere Pub. Corp., 1983.
#: keys are CASRN and data for each chemical are in standard thermosteam units.

def VDI_tabular_data(CASRN, prop):
    r'''This function retrieves the tabular data available for a given chemical
    and a given property. Lookup is based on CASRNs. Length of data returned
    varies between chemicals. All data is at saturation condition from [1]_.

    Function has data for 58 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]
    prop : string
        Property [-]

    Returns
    -------
    Ts : list
        Temperatures where property data is available, [K]
    props : list
        Properties at each temperature, [various]

    Notes
    -----
    The available properties are 'P', 'Density (l)', 'Density (g)', 'Hvap',
    'Cp (l)', 'Cp (g)', 'Mu (l)', 'Mu (g)', 'K (l)', 'K (g)', 'Pr (l)',
    'Pr (g)', 'sigma', 'Beta', 'Volume (l)', and 'Volume (g)'.

    Data is available for all properties and all chemicals; surface tension
    data was missing for mercury, but added as estimated from the a/b
    coefficients listed in Jasper (1972) to simplify the function.

    Examples
    --------
    >>> VDI_tabular_data('67-56-1', 'Mu (g)')
    ([337.63, 360.0, 385.0, 410.0, 435.0, 460.0, 500.0], [1.11e-05, 1.18e-05, 1.27e-05, 1.36e-05, 1.46e-05, 1.59e-05, 2.04e-05])

    References
    ----------
    .. [1] Gesellschaft, VDI, ed. VDI Heat Atlas. 2E. Berlin : Springer, 2010.
    '''
    try:
        d = VDI_saturation_dict[CASRN]
    except KeyError:
        raise Exception('CASRN not in VDI tabulation')
    try:
        props, Ts = d[prop], d['T']
    except:
        raise Exception('Property not specified correctly')
    Ts = [T for p, T in zip(props, Ts) if p]
    props = [p for p in props if p]

    # Not all data series convererge to correct values
    if prop == 'sigma':
        Ts.append(d['Tc'])
        props.append(0)
    return Ts, props


# %% Boiling point
read = CASDataReader('Phase Change')

boiling_point_data_Yaws = read('Yaws Boiling Points.tsv')
melting_point_data_ON = read('OpenNotebook Melting Points.tsv')
vaporization_data_Gharagheizi = read('Ghazerati Appendix Vaporization Enthalpy.tsv')
vaporization_data_CRC = read('CRC Handbook Heat of Vaporization.tsv')
fusion_data_CRC = read('CRC Handbook Heat of Fusion.tsv')
sublimation_data_Gharagheizi = read('Ghazerati Appendix Sublimation Enthalpy.tsv')
vaporization_data_Perrys2_150 = read('Table 2-150 Heats of Vaporization of Inorganic and Organic Liquids.tsv')
vaporization_data_VDI_PPDS_4 = read('VDI PPDS Enthalpies of vaporization.tsv')
vaporization_data_Alibakhshi_Cs = read('Alibakhshi one-coefficient enthalpy of vaporization.tsv')

### Boiling Point at 1 atm

normal_boiling_point_data_sources = {
    'CRC-Inorganic': inorganic_data_CRC,
    'CRC-Organic': organic_data_CRC,
    'Yaws': boiling_point_data_Yaws,
}

melting_point_data_sources = {
    'OPEN-NTBKM': melting_point_data_ON,  
    'CRC-Inorganic': inorganic_data_CRC,
    'CRC-Organic': organic_data_CRC,   
}

fusion_data_sources = {
    'CRC at Tm': fusion_data_CRC,
}

sublimation_data_sources = {
    'Ghazerati Appendix, at 298K': sublimation_data_Gharagheizi,
}

# %% Critical point

# TODO: check out 12E of this data http://pubsdc3.acs.org/doi/10.1021/acs.jced.5b00571

read = CASDataReader('Critical Properties')
critical_data_IUPAC = read('IUPACOrganicCriticalProps.tsv')  # IUPAC Organic data series
critical_data_Matthews = read('Mathews1972InorganicCriticalProps.tsv')
critical_data_CRC = read('CRCCriticalOrganics.tsv') # CRC Handbook from TRC Organic data section (only in 2015)
critical_data_PSRKR4 = read('Appendix to PSRK Revision 4.tsv')
critical_data_PassutDanner = read('PassutDanner1973.tsv')
critical_data_Yaws = read('Yaws Collection.tsv')

critical_data_sources = {
    'IUPAC': critical_data_IUPAC,
    'Matthews': critical_data_Matthews,
    'CRC': critical_data_CRC,
    'PSRK': critical_data_PSRKR4,
    'Passut Danner':critical_data_PassutDanner,
    'Yaws': critical_data_Yaws,
}

# %% Triple point

read = CASDataReader('Triple Properties')
tiple_point_Staveley_data = read('Staveley 1981.tsv')

triple_point_sources = {
    "Staveley": tiple_point_Staveley_data,
}

# %% Dipole moment

read = CASDataReader('Misc')
dipole_data_Poling = read('Poling Dipole.csv')
dipole_data_CCCBDB = read('cccbdb.nist.gov Dipoles.csv')
dipole_data_Muller = read('Muller Supporting Info Dipoles.csv')

dipole_data_sources = {
    'CCCBDB': dipole_data_CCCBDB,
    'Poling': dipole_data_Poling,
    'Muller': dipole_data_Muller
}

# %% Vapor pressure

read = CASDataReader('Vapor Pressure')
Psat_data_WagnerMcGarry = read("Wagner Original McGarry.tsv")
Psat_data_Wagner = read("Wagner Collection Poling.tsv")
Psat_data_Antoine = read("Antoine Collection Poling.tsv")
Psat_data_AntoineExtended = read("Antoine Extended Collection Poling.tsv")
Psat_data_Perrys2_8 = read("Table 2-8 Vapor Pressure of Inorganic and Organic Liquids.tsv")
Psat_data_VDI_PPDS_3 = read("VDI PPDS Boiling temperatures at different pressures.tsv")

# %% Heat capacity

read = CASDataReader('Heat Capacity')
Cn_data_Poling = read('PolingDatabank.tsv')
Cn_data_TRC_gas = read('TRC Thermodynamics of Organic Compounds in the Gas State.tsv')
Cn_data_CRC_standard = read('CRC Standard Thermodynamic Properties of Chemical Substances.tsv')
Cn_data_PerryI = load_json(read.folder, 'Perrys Table 2-151.json')
# Read in a dict of heat capacities of irnorganic and elemental solids.
# These are in section 2, table 151 in:
# Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
# Eighth Edition. McGraw-Hill Professional, 2007.
# Formula:
# Cn(Cal/mol/K) = Const + Lin*T + Quadinv/T^2 + Quadinv*T^2
# Phases:
# c, gls, l, g.

zabransky_dict_sat_s = {}
zabransky_dict_sat_p = {}
zabransky_dict_const_s = {}
zabransky_dict_const_p = {}
zabransky_dict_iso_s = {}
zabransky_dict_iso_p = {}

# C means average heat capacity values, from less rigorous experiments
# sat means heat capacity along the saturation line
# p means constant-pressure values, 
# second argument is whether or not it has a spline
type_to_zabransky_dict = {('C', True): zabransky_dict_const_s, 
                          ('C', False):   zabransky_dict_const_p,
                          ('sat', True):  zabransky_dict_sat_s,
                          ('sat', False): zabransky_dict_sat_p,
                          ('p', True):    zabransky_dict_iso_s,
                          ('p', False):   zabransky_dict_iso_p}

with open(os.path.join(read.folder, 'Zabransky.tsv'), encoding='utf-8') as f:
    next(f)
    for line in f:
        values = to_nums(line.strip('\n').split('\t'))
        (CAS, name, Type, uncertainty, Tmin, Tmax, a1s, a2s, a3s, a4s, a1p, a2p, a3p, a4p, a5p, a6p, Tc) = values
        spline = bool(a1s) # False if Quasypolynomial, True if spline
        d = type_to_zabransky_dict[(Type, spline)]
        if spline:
            if CAS not in d:
                d[CAS] = [(a1s, a2s, a3s, a4s, Tmin, Tmax)]
            else:
                d[CAS].append((a1s, a2s, a3s, a4s, Tmin, Tmax))
        else:
            # No duplicates for quasipolynomials
            d[CAS] = (Tc, a1p, a2p, a3p, a4p, a5p, a6p, Tmin, Tmax)

# %% Heat of formation

read = CASDataReader("Reactions")
Hf_data_API_TDB = read('API TDB Albahri Hf (g).tsv')
Hf_data_ATcT_l = read('ATcT 1.112 (l).tsv')
Hf_data_ATcT_g = read('ATcT 1.112 (g).tsv')
Hf_data_Yaws_g = read('Yaws Hf S0 (g).tsv')
Hf_data_user = read('Example User Hf.tsv')

heat_of_formation_sources = {
    'Other': Hf_data_user
}
heat_of_formation_solid_sources = {    
    'CRC': Cn_data_CRC_standard,
}
heat_of_formation_liquid_sources = {
    'ATCT_L': Hf_data_ATcT_l,
    'CRC': Cn_data_CRC_standard,
}
heat_of_formation_gas_sources = {
    'ATCT_G': Hf_data_ATcT_g,
    'CRC': Cn_data_CRC_standard,
    'YAWS': Hf_data_Yaws_g,
    'TRC': Cn_data_TRC_gas,
    'APO TDB Albahri': Hf_data_API_TDB,
}

# %% Surface tension

read = CASDataReader('Interface')
sigma_data_Mulero_Cachadina = read('MuleroCachadinaParameters.tsv')
sigma_data_Jasper_Lange = read('Jasper-Lange.tsv')
sigma_data_Somayajulu = read('Somayajulu.tsv')
sigma_data_Somayajulu_2 = read('SomayajuluRevised.tsv')
sigma_data_VDI_PPDS_11 = read('VDI PPDS surface tensions.tsv')

# %% Thermal conductivity

read = CASDataReader('Thermal Conductivity')
# kappa_data_Perrys2_314 = read('Table 2-314 Vapor Thermal Conductivity of Inorganic and Organic Substances.tsv')
kappa_data_Perrys2_315 = read('Table 2-315 Thermal Conductivity of Inorganic and Organic Liquids.tsv')
kappa_data_VDI_PPDS_9 = read('VDI PPDS Thermal conductivity of saturated liquids.tsv')
# kappa_data_VDI_PPDS_10 = read('VDI PPDS Thermal conductivity of gases.tsv')

# %% Viscosity

read = CASDataReader('Viscosity')
mu_data_Dutt_Prasad = read('Dutt Prasad 3 term.tsv')
mu_data_VN3 = read('Viswanath Natarajan Dynamic 3 term.tsv')
mu_data_VN2 = read('Viswanath Natarajan Dynamic 2 term.tsv')
# mu_data_VN2E_data = read('Viswanath Natarajan Dynamic 2 term Exponential.tsv')
mu_data_Perrys2_313 = read('Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv')
mu_data_Perrys2_312 = read('Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv')
mu_data_VDI_PPDS_7 = read('VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv')
mu_data_VDI_PPDS_8 = read('VDI PPDS Dynamic viscosity of gases polynomials.tsv')

# %% Volume

read = CASDataReader("Density")
V_data_COSTALD = read('COSTALD Parameters.tsv')
V_data_SNM0 = read('Mchaweh SN0 deltas.tsv')
V_data_Perry_l = read('Perry Parameters 105.tsv')
V_data_VDI_PPDS_2 = read('VDI PPDS Density of Saturated Liquids.tsv')
V_data_CRC_inorg_l = read('CRC Inorganics densties of molten compounds and salts.tsv')
V_data_CRC_inorg_l_const = read('CRC Liquid Inorganic Constant Densities.tsv')
V_data_CRC_inorg_s_const = read('CRC Solid Inorganic Constant Densities.tsv')
V_data_CRC_virial = read('CRC Virial polynomials.tsv')

# %% Permittivity

read = CASDataReader('Electrolytes')

#: [CASDataSource] Permitivity data
permittivity_data_CRC = read('Permittivity (Dielectric Constant) of Liquids.tsv')

# %% Electrical conductivity

read = CASDataReader('Electrolytes')
Lange_cond_pure = read('Lange Pure Species Conductivity.tsv')
Marcus_ion_conductivities = read('Marcus Ion Conductivities.tsv')
CRC_ion_conductivities = read('CRC conductivity infinite dilution.tsv')
Magomedovk_thermal_cond = read('Magomedov Thermal Conductivity.tsv')
CRC_aqueous_thermodynamics = read('CRC Thermodynamic Properties of Aqueous Ions.csv')
electrolyte_dissociation_reactions = read('Electrolyte dissociations.csv', sep=None)

McCleskey_parameters = namedtuple("McCleskey_parameters",
                                  ["Formula", 'lambda_coeffs', 'A_coeffs', 'B', 'multiplier'])

McCleskey_conductivities = {}
with open(os.path.join(read.folder, 'McCleskey Electrical Conductivity.csv')) as f:
    next(f)
    for line in f:
        values = line.strip().split('\t')
        formula, CASRN, lbt2, lbt, lbc, At2, At, Ac, B, multiplier = to_nums(values)
        McCleskey_conductivities[CASRN] = McCleskey_parameters(formula, 
            [lbt2, lbt, lbc], [At2, At, Ac], B, multiplier)

Lange_cond_pure = read('Lange Pure Species Conductivity.tsv')

# %% Electrolytes

read = CASDataReader('Electrolytes')
Laliberte_Density_ParametersDict = {}
Laliberte_Viscosity_ParametersDict = {}
Laliberte_Heat_Capacity_ParametersDict = {}

# Do not re-implement with Pandas, as current methodology uses these dicts in each function
with open(os.path.join(read.folder, 'Laliberte2009.tsv')) as f:
    next(f)
    for line in f:
        values = to_nums(line.split('\t'))

        _name, CASRN, _formula, _MW, c0, c1, c2, c3, c4, Tmin, Tmax, wMax, pts = values[0:13]
        if c0:
            Laliberte_Density_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "C0":c0, "C1":c1, "C2":c2, "C3":c3, "C4":c4, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}

        v1, v2, v3, v4, v5, v6, Tmin, Tmax, wMax, pts = values[13:23]
        if v1:
            Laliberte_Viscosity_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "V1":v1, "V2":v2, "V3":v3, "V4":v4, "V5":v5, "V6":v6, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}

        a1, a2, a3, a4, a5, a6, Tmin, Tmax, wMax, pts = values[23:34]
        if a1:
            Laliberte_Heat_Capacity_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "A1":a1, "A2":a2, "A3":a3, "A4":a4, "A5":a5, "A6":a6, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}
Laliberte_data = read('Laliberte2009.tsv'),


# %% Mixtures

read = CASDataReader('Identifiers')
mixture_dict = load_json(read.folder, 'Mixtures Compositions.json')
# Read in a dict of 90 or so mixutres, their components, and synonyms.
# Small errors in mole fractions not adding to 1 are known.
# Errors in adding mass fraction are less common, present at the 5th decimal.
# TODO: Normalization
# Mass basis is assumed for all mixtures.

def mixture_from_any(ID):
    '''Looks up a string which may represent a mixture in the database of 
    thermo to determine the key by which the composition of that mixture can
    be obtained in the dictionary `mixture_dict`.

    Parameters
    ----------
    ID : str
        A string containing the name which may represent a
        mixture.

    Returns
    -------
    key : str
        Key for access to the data on the mixture in `_MixtureDict`.

    Notes
    -----
    White space, '-', and upper case letters are removed in the search.

    Examples
    --------
    >>> mixture_from_any('R512A')
    'R512A'
    >>> mixture_from_any(u'air')
    'Air'
    '''
    ID = ID.lower().strip()
    ID2 = ID.replace(' ', '')
    ID3 = ID.replace('-', '')
    for i in [ID, ID2, ID3]:
        if i in mixture_dict:
            return mixture_dict[i]
    raise LookupError('Mixture name not recognized')