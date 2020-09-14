---
title: "Thermosteam: BioSTEAM's Premier Thermodynamic Engine"
tags:
  - Python
  - BioSTEAM
  - thermodynamics
  - process modeling
  - mass and energy balance
  - chemical properties
  - mixture properties
  - phase equilibrium
authors:
  - name: Yoel Cortés-Peña
    orcid: 0000-0003-1742-5059
    affiliation: "1, 2"
affiliations:
  - name: Department of Civil and Environmental Engineering, University of Illinois at Urbana-Champaign
    index: 1
  - name: DOE Center for Advanced Bioenergy and Bioproducts Innovation (CABBI)
    index: 2
date: 12 May 2020
bibliography: paper.bib
---

# Summary

The design and simulation of unit operations in a chemical process often 
requires the estimation of mixture properties and phase equilibrium (e.g. 
vapor-liquid and liquid-liquid equilibrium) to correctly estimate material and 
energy balances, as well as the unit operation design requirements to achieve 
a set of design specifications. For example, estimating both phase equilibrium
and fluid viscosities, densities, and surface tensions are required to design a 
distillation column that can achieve a specified recovery of chemicals 
[@Perry]. The overarching goal of thermosteam is to enable the rigorous design 
and simulation of unit operations by creating thermodynamic property packages
from both user-defined chemicals and databanks from the `chemicals` library [@chemicals], an open-source compilation of data and functions for the estimation of thermodynamic and transport properties for both pure chemicals and mixtures.

Roughly 20,000 chemicals with temperature- and pressure-dependent property data are
included in the `chemicals` library. Thermosteam builds upon `chemicals` with a robust and flexible framework that facilitates the creation of property packages. Its extendable framework allows for easy integration of new models for estimating pure component properties, thermodynamic equilibrium coefficients, and mixture properties. 
The Biorefinery Simulation and Techno-Economic Analysis Modules (BioSTEAM) 
has adopted thermosteam as its premier thermodynamic engine [@BioSTEAM].
Published biorefinery designs modeled in BioSTEAM implement property 
packages created with thermosteam [@Bioindustrial-Park], including a cornstover 
biorefinery for the production of cellulosic ethanol, a lipid-cane biorefinery 
for the co-production of ethanol and biodiesel, a sugarcane biorefinery
for the production of bioethanol, and a wheatstraw biorefinery for the production
of cellulosic ethanol [@BioSTEAM][@Sanchis].

In `thermosteam`, Peng Robinson is the default equation of state 
of all pure components. However, the estimation of pure component chemical 
properties are not limited to solving the equation of state. Several models 
for thermodynamic properties (e.g. density, heat capacity, vapor pressure, 
heat of vaporization, etc.) may rely on fitted coefficients and key chemical 
properties (e.g. critical temperature and pressure). To facilitate the 
calculation of mixture properties, thermosteam's default mixing rule estimates 
mixture properties by assuming a molar weighted average of the pure chemical 
properties.

Thermosteam allows for fast estimation of thermodynamic equilibrium within 
hundreds of microseconds through the smart use of cache and Numba just-in-time 
(JIT) compiled functions [@numba]. The main vapor-liquid equilibrium (VLE) 
algorithm solves the modified Raoult’s law equation with activity coefficients
estimated through UNIQUAC Functional-group Activity Coefficients (UNIFAC) 
interaction parameters [@Gmehling]. Modified Raoult’s law is suitable to 
estimate VLE of nonideal mixtures under low to moderate pressures. At high to 
near-critical pressures, gaseous nonidealities become more significant. In a 
near future, thermosteam may also implement the Predictive Soave–Redlich–Kwong
(PSRK) functional group model for estimating phase equilibrium of critical
mixtures.

All of thermosteam's application program interface (API) is documented with 
examples. These examples also serve as preliminary tests that must pass before
accepting any changes to the software via continuous integration on Github. Additionally, the online documentation includes a full tutorial that concludes with the creation of a property package. Thermosteam’s powerful features and extensive documentation encourage its users to become a part of its community-driven platform 
and help it become more industrially and academically relevant. 

# Acknowledgements

I would like to thank Caleb Bell for developing the open-source `chemicals` library
in Python, which has served as both groundwork and inspiration for developing `thermosteam`. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE—1746047. Any opinions, findings, and conclusions or recommendations expressed in this publication are those of the authors and do not necessarily reflect the views of the National Science Foundation. This work was funded by the DOE Center for Advanced Bioenergy and Bioproducts Innovation (U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research under Award Number DE-SC0018420). Any opinions, findings, and conclusions or recommendations expressed in this publication are those of the author and do not necessarily reflect the views of the U.S. Department of Energy

# References