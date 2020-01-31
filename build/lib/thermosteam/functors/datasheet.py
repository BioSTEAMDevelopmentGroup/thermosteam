# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''
import numpy as np
import pandas as pd

__all__ = ['tabulate_solid', 'tabulate_liq', 'tabulate_gas', 
           'tabulate_constants']
           

def tabulate_solid(chemical, Tmin=None, Tmax=None, pts=10):

    (rhos, Cps) = [[] for i in range(2)]
    if not Tmin:  # pragma: no cover
        if chemical.Tm:
            Tmin = chemical.Tm-100
        else:
            Tmin = 150.
    if not Tmax:  # pragma: no cover
        if chemical.Tm:
            Tmax = chemical.Tm
        else:
            Tmax = 350

    Ts = np.linspace(Tmin, Tmax, pts)
    for T in Ts:
        chemical.T = T
        rhos.append(chemical.rhos)
        Cps.append(chemical.Cps)

    data = {}
    data['Density, kg/m^3'] = rhos
    data['Constant-pressure heat capacity, J/kg/K'] = Cps

    df = pd.DataFrame(data, index=Ts)
    df.index.name = 'T, K'
    return df


def tabulate_liq(chemical, Tmin=None, Tmax=None, pts=10):
    (rhos, Cps, mugs, kgs, Prs, alphas, isobarics, JTs, Psats, sigmas, Hvaps,
     permittivities) = [[] for i in range(12)]
    if not Tmin:  # pragma: no cover
        if chemical.Tm:
            Tmin = chemical.Tm
        else:
            Tmin = 273.15
    if not Tmax:  # pragma: no cover
        if chemical.Tc:
            Tmax = chemical.Tc
        else:
            Tmax = 450

    Ts = np.linspace(Tmin, Tmax, pts)
    for T in Ts:
        chemical.T = T

        rhos.append(chemical.rhol)
        Cps.append(chemical.Cpl)
        mugs.append(chemical.mul)
        kgs.append(chemical.kl)
        Prs.append(chemical.Prl)
        alphas.append(chemical.alphal)
        isobarics.append(chemical.isobaric_expansion_l)
        JTs.append(chemical.JTg)
        Psats.append(chemical.Psat)
        Hvaps.append(chemical.Hvap)
        sigmas.append(chemical.sigma)
        permittivities.append(chemical.permittivity)

    data = {}
    data['Saturation pressure, Pa'] = Psats
    data['Density, kg/m^3'] = rhos
    data['Constant-pressure heat capacity, J/kg/K'] = Cps
    data['Heat of vaporization, J/kg'] = Hvaps
    data['Viscosity, Pa*S'] = mugs
    data['Thermal conductivity, W/m/K'] = kgs
    data['Surface tension, N/m'] = sigmas
    data['Prandtl number'] = Prs
    data['Thermal diffusivity, m^2/s'] = alphas
    data['Isobaric expansion, 1/K'] = isobarics
    data['Joule-Thompson expansion coefficient, K/Pa'] = JTs
    data['Permittivity'] = permittivities

    df = pd.DataFrame(data, index=Ts)
    df.index.name = 'T, K'
    return df


def tabulate_gas(chemical, Tmin=None, Tmax=None, pts=10):
    (rhos, Cps, Cvs, mugs, kgs, Prs, alphas, isobarics, isentropics, JTs) = [[] for i in range(10)]
    if not Tmin:  # pragma: no cover
        if chemical.Tm:
            Tmin = chemical.Tm
        else:
            Tmin = 273.15
    if not Tmax:  # pragma: no cover
        if chemical.Tc:
            Tmax = chemical.Tc
        else:
            Tmax = 450

    Ts = np.linspace(Tmin, Tmax, pts)
    for T in Ts:
        chemical.T = T

        rhos.append(chemical.rhog)
        Cps.append(chemical.Cpg)
        Cvs.append(chemical.Cvg)
        mugs.append(chemical.mug)
        kgs.append(chemical.kg)
        Prs.append(chemical.Prg)
        alphas.append(chemical.alphag)
        isobarics.append(chemical.isobaric_expansion_g)
        isentropics.append(chemical.isentropic_exponent)
        JTs.append(chemical.JTg)
    data = {}
    data['Density, kg/m^3'] = rhos
    data['Constant-pressure heat capacity, J/kg/K'] = Cps
    data['Constant-volume heat capacity, J/kg/K'] = Cvs
    data['Viscosity, Pa*S'] = mugs
    data['Thermal consuctivity, W/m/K'] = kgs
    data['Prandtl number'] = Prs
    data['Thermal diffusivity, m^2/s'] = alphas
    data['Isobaric expansion, 1/K'] = isobarics
    data['Isentropic exponent'] = isentropics
    data['Joule-Thompson expansion coefficient, K/Pa'] = JTs

    df = pd.DataFrame(data, index=Ts)  # add orient='index'
    df.index.name = 'T, K'
    return df


def tabulate_constants(chemical, full=False, vertical=False):
    pd.set_option('display.max_rows', 100000)
    pd.set_option('display.max_columns', 100000)

    all_chemicals = {}

    if isinstance(chemical, str):
        cs = [chemical]
    else:
        cs = chemical

    for chemical in cs:
        data = {}
        data['CAS'] = chemical.CAS
        data['Formula'] = chemical.formula
        data['MW, g/mol'] = chemical.MW
        data['Tm, K'] = chemical.Tm
        data['Tb, K'] = chemical.Tb
        data['Tc, K'] = chemical.Tc
        data['Pc, Pa'] = chemical.Pc
        data['Vc, m^3/mol'] = chemical.Vc
        data['Zc'] = chemical.Zc
        data['rhoc, kg/m^3'] = chemical.rhoc
        data['Acentric factor'] = chemical.omega
        data['Triple temperature, K'] = chemical.Tt
        data['Triple pressure, Pa'] = chemical.Pt
        data['Heat of vaporization at Tb, J/mol'] = chemical.Hvap_Tbm
        data['Heat of fusion, J/mol'] = chemical.Hfusm
        data['Heat of sublimation, J/mol'] = chemical.Hsubm
        data['Heat of formation, J/mol'] = chemical.Hf
        data['Dipole moment, debye'] = chemical.dipole
        data['Molecular Diameter, Angstrom'] = chemical.molecular_diameter
        data['Stockmayer parameter, K'] = chemical.Stockmayer
        data['Refractive index'] = chemical.RI
        data['Lower flammability limit, fraction'] = chemical.LFL
        data['Upper flammability limit, fraction'] = chemical.UFL
        data['Flash temperature, K'] = chemical.Tflash
        data['Autoignition temperature, K'] = chemical.Tautoignition
        data['Time-weighted average exposure limit'] = str(chemical.TWA)
        data['Short-term exposure limit'] = str(chemical.STEL)
        data['logP'] = chemical.logP

        if full:
            data['smiles'] = chemical.smiles
            data['InChI'] = chemical.InChI
            data['InChI key'] = chemical.InChI_Key
            data['IUPAC name'] = chemical.IUPAC_name
            data['solubility parameter, Pa^0.5'] = chemical.solubility_parameter
            data['Parachor'] = chemical.Parachor
            data['Global warming potential'] = chemical.GWP
            data['Ozone depletion potential'] = chemical.ODP
            data['Electrical conductivity, S/m'] = chemical.conductivity

        all_chemicals[chemical.name] = data

    if vertical:
        df = pd.DataFrame.from_dict(all_chemicals)
    else:
        df = pd.DataFrame.from_dict(all_chemicals, orient='index')
    return df




#chemicals = ['Sodium Hydroxide', 'sodium chloride', 'methanol',
#'hydrogen sulfide', 'methyl mercaptan', 'Dimethyl disulfide', 'dimethyl sulfide',
# 'alpha-pinene', 'chlorine dioxide', 'sulfuric acid', 'SODIUM CHLORATE', 'carbon dioxide', 'Cl2', 'formic acid',
# 'sodium sulfate']
#for i in chemicals:
#    print tabulate_solid(i)
#    print tabulate_liq(i)
#    print tabulate_gas(i)
#    tabulate_constants(i)

#tabulate_constants('Methylene blue')