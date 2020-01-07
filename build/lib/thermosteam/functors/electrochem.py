# -*- coding: utf-8 -*-
import os
from collections import namedtuple
from ..base import HandleBuilder, functor
from math import e, exp, log10
from .._constants import N_A
from .utils import to_nums, CASDataReader
from scipy.interpolate import interp1d
import pandas as pd

F = e*N_A

read = CASDataReader(__file__, 'Electrolytes')
Lange_cond_pure = read('Lange Pure Species Conductivity.tsv')
Marcus_ion_conductivities = read('Marcus Ion Conductivities.tsv')
CRC_ion_conductivities = read('CRC conductivity infinite dilution.tsv')
Magomedovk_thermal_cond = read('Magomedov Thermal Conductivity.tsv')
CRC_aqueous_thermodynamics = read('CRC Thermodynamic Properties of Aqueous Ions.csv')
electrolyte_dissociation_reactions = read('Electrolyte dissociations.csv', sep=None)

_Laliberte_Density_ParametersDict = {}
_Laliberte_Viscosity_ParametersDict = {}
_Laliberte_Heat_Capacity_ParametersDict = {}


# Do not re-implement with Pandas, as current methodology uses these dicts in each function
with open(os.path.join(read.folder, 'Laliberte2009.tsv')) as f:
    next(f)
    for line in f:
        values = to_nums(line.split('\t'))

        _name, CASRN, _formula, _MW, c0, c1, c2, c3, c4, Tmin, Tmax, wMax, pts = values[0:13]
        if c0:
            _Laliberte_Density_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "C0":c0, "C1":c1, "C2":c2, "C3":c3, "C4":c4, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}

        v1, v2, v3, v4, v5, v6, Tmin, Tmax, wMax, pts = values[13:23]
        if v1:
            _Laliberte_Viscosity_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "V1":v1, "V2":v2, "V3":v3, "V4":v4, "V5":v5, "V6":v6, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}

        a1, a2, a3, a4, a5, a6, Tmin, Tmax, wMax, pts = values[23:34]
        if a1:
            _Laliberte_Heat_Capacity_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "A1":a1, "A2":a2, "A3":a3, "A4":a4, "A5":a5, "A6":a6, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}
Laliberte_data = pd.read_csv(os.path.join(read.folder, 'Laliberte2009.tsv'),
                          sep='\t', index_col=0)


### Laliberty Viscosity Functions

def Laliberte_water_viscosity(T):
    r'''Return the viscosity of a water (Pa*s) at arbitrary temperatures (K) using the form proposed by [1]_. 

    .. math::
        \mu_w = \frac{T - 27.15}{(0.05594T-25.27581)t + 2867.723}

    Notes
    -----
    Original source or pure water viscosity is not cited.
    No temperature range is given for this equation.

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    return (T - 27.15)/((55.94*T-25275.81)*T + 2867723)

@functor
def Laliberte_partial_viscosity(T, w_w, v1, v2, v3, v4, v5, v6):
    t = T-273.15
    mu_i = exp((v1*(1-w_w)**v2 + v3)/(v4*t+1))/(v5*(1-w_w)**v6 + 1)
    return mu_i/1000.

### Laliberty Density Functions

@functor
def Laliberte_water_density(T):
    r'''Return the density of water using the form proposed by [1]_.
    No parameters are needed, just a temperature. Units are Kelvin and kg/m^3h.

    .. math::
        \rho_w = \frac{\left\{\left([(-2.8054253\times 10^{-10}\cdot t +
        1.0556302\times 10^{-7})t - 4.6170461\times 10^{-5}]t
        -0.0079870401\right)t + 16.945176   \right\}t + 999.83952}
        {1 + 0.01687985\cdot t}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    rho_w : float
        Water density, [kg/m^3]

    Notes
    -----
    Original source not cited
    No temperature range is used.

    Examples
    --------
    >>> Laliberte_density_w(298.15)
    997.0448954179155
    >>> Laliberte_density_w(273.15 + 50)
    988.0362916114763

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    t = T-273.15
    rho_w = (((((-2.8054253E-10*t + 1.0556302E-7)*t - 4.6170461E-5)*t - 0.0079870401)*t + 16.945176)*t + 999.83952) \
        / (1 + 0.01687985*t)
    return rho_w

@functor
def Laliberte_density(T, w_w, c0, c1, c2, c3, c4):
    t = T - 273.15
    return ((c0*(1 - w_w)+c1)*exp(1E-6*(t + c4)**2))/((1 - w_w) + c2 + c3*t)


### Laliberty Heat Capacity Functions

_T_array = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
_Cp_array = [4294.03, 4256.88, 4233.58, 4219.44, 4204.95, 4195.45, 4189.1, 4184.8, 4181.9, 4180.02, 4178.95, 4178.86, 4178.77, 4179.56, 4180.89, 4182.77, 4185.17, 4188.1, 4191.55, 4195.52, 4200.01, 4205.02, 4210.57, 4216.64, 4223.23, 4230.36, 4238.07, 4246.37, 4255.28, 4264.84, 4275.08, 4286.04]
Laliberte_heat_capacity_w_interp = interp1d(_T_array, _Cp_array, kind='cubic')

@functor
def Laliberte_water_heat_capacity(T):
    r'''Return the heat capacity of water using the interpolation proposed by [1]_.

    .. math::
        Cp_w = Cp_1 + (Cp_2-Cp_1) \left( \frac{t-t_1}{t_2-t_1}\right)
        + \frac{(Cp_3 - 2Cp_2 + Cp_1)}{2}\left( \frac{t-t_1}{t_2-t_1}\right)
        \left( \frac{t-t_1}{t_2-t_1}-1\right)

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    Cp_w : float
        Water heat capacity, [J/kg/K]

    Notes
    -----
    Units are Kelvin and J/kg/K.
    Original source not cited
    No temperature range is used.
    The original equation is not used, but rather a cubic scipy interpolation routine.

    Examples
    --------
    >>> Laliberte_heat_capacity_w(273.15+3.56)
    4208.878020261102

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    return float(Laliberte_heat_capacity_w_interp(T - 273.15))

@functor
def Laliberte_Heat_Capacity(T, w_w, a1, a2, a3, a4, a5, a6):
    t = T - 273.15
    alpha = a2*t + a3*exp(0.01*t) + a4*(1. - w_w)
    return 1000. * (a1*exp(alpha) + a5*(1. - w_w)**a6)


### Electrical Conductivity


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

Lange_cond_pure = pd.read_csv(os.path.join(read.folder, 'Lange Pure Species Conductivity.tsv'),
                          sep='\t', index_col=0)


LANGE_COND = "LANGE_COND"
NONE = 'None'

conductivity_methods = [LANGE_COND]

def conductivity(CASRN):
    r'''This function handles the retrieval of a chemical's conductivity.
    Lookup is based on CASRNs.

    Function has data for approximately 100 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    kappa : float
        Electrical conductivity of the fluid, [S/m]
    T : float, only returned if full_info == True
        Temperature at which conductivity measurement was made
        
    Notes
    -----
    Only one source is available in this function. It is:

        * 'LANGE_COND' which is from Lange's Handbook, Table 8.34 Electrical 
        Conductivity of Various Pure Liquids', a compillation of data in [1]_.

    Examples
    --------
    >>> conductivity('7732-18-5')
    (4e-06, 291.15)

    References
    ----------
    .. [1] Speight, James. Lange's Handbook of Chemistry. 16 edition.
       McGraw-Hill Professional, 2005.
    '''
    kappa = float(Lange_cond_pure.at[CASRN, 'Conductivity'])
    T = float(Lange_cond_pure.at[CASRN, 'T'])
    return kappa, T
    

