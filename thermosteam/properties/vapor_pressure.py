# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the vapor_pressure module from the chemicals's library:
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
"""
All data and methods related to the vapor pressure of a chemical.

References
----------
.. [1] Antoine, C. 1888. Tensions des Vapeurs: Nouvelle Relation Entre les 
       Tensions et les Tempé. Compt.Rend. 107:681-684.
.. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
.. [3] Yaws, Carl L. The Yaws Handbook of Vapor Pressure: Antoine 
       Coefficients. 1 edition. Houston, Tex: Gulf Publishing Company, 2007.
.. [4] McGarry, Jack. "Correlation and Prediction of the Vapor Pressures of
       Pure Liquids over Large Pressure Ranges." Industrial & Engineering
       Chemistry Process Design and Development 22, no. 2 (April 1, 1983):
       313-22. doi:10.1021/i200021a023.
.. [5] Wagner, W. "New Vapour Pressure Measurements for Argon and Nitrogen and
       a New Method for Establishing Rational Vapour Pressure Equations."
       Cryogenics 13, no. 8 (August 1973): 470-82. doi:10.1016/0011-2275(73)90003-9
.. [6] Reid, Robert C..; Prausnitz, John M.;; Poling, Bruce E.
       The Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
.. [7] Lee, Byung Ik, and Michael G. Kesler. "A Generalized Thermodynamic
       Correlation Based on Three-Parameter Corresponding States." AIChE Journal
       21, no. 3 (1975): 510-527. doi:10.1002/aic.690210313.
.. [8] Ambrose, D., and J. Walton. "Vapour Pressures up to Their Critical
       Temperatures of Normal Alkanes and 1-Alkanols." Pure and Applied
       Chemistry 61, no. 8 (1989): 1395-1403. doi:10.1351/pac198961081395.
.. [9] Sanjari, Ehsan, Mehrdad Honarmand, Hamidreza Badihi, and Ali
       Ghaheri. "An Accurate Generalized Model for Predict Vapor Pressure of
       Refrigerants." International Journal of Refrigeration 36, no. 4
       (June 2013): 1327-32. doi:10.1016/j.ijrefrig.2013.01.007.
.. [10] Edalat, M., R. B. Bozar-Jomehri, and G. A. Mansoori. "Generalized 
        Equation Predicts Vapor Pressure of Hydrocarbons." Oil and Gas Journal; 
        91:5 (February 1, 1993).

"""
import sys
from chemicals import vapor_pressure as vp
module = sys.modules[__name__]
sys.modules[__name__] = vp
del sys
from ..base import functor, TDependentHandleBuilder
from .dippr import DIPPR_EQ101
import numpy as np
from .data import (Psat_data_WagnerMcGarry,
                   Psat_data_Wagner,
                   Psat_data_Antoine,
                   Psat_data_AntoineExtended,
                   Psat_data_Perrys2_8,
                   Psat_data_VDI_PPDS_3,
)

# %% Vapor pressure

vp.__all__.extend([
    'vapor_pressure_handle',
])

Antoine = functor(vp.Antoine, var='Psat')
TRC_Antoine_extended = functor(vp.TRC_Antoine_extended, var='Psat')
Wagner_original = functor(vp.Wagner_original, var='Psat')
Wagner = functor(vp.Wagner, var='Psat')
boiling_critical_relation = functor(vp.boiling_critical_relation , var='Psat')
Lee_Kesler = functor(vp.Lee_Kesler, var='Psat')
Ambrose_Walton = functor(vp.Ambrose_Walton, var='Psat')
Sanjari = functor(vp.Sanjari, var='Psat')
Edalat = functor(vp.Edalat, var='Psat')

@TDependentHandleBuilder('Psat')
def vapor_pressure_handle(handle, CAS, Tb, Tc, Pc, omega):
    add_model = handle.add_model
    if CAS in Psat_data_WagnerMcGarry:
        _, a, b, c, d, Pc, Tc, Tmin = Psat_data_WagnerMcGarry[CAS]
        Tmax = Tc
        data = (Tc, Pc, a, b, c, d)
        add_model(Wagner_original.functor.from_args(data), Tmin, Tmax)
    elif CAS in Psat_data_Wagner:
        _, a, b, c, d, Tc, Pc, Tmin, Tmax = Psat_data_Wagner[CAS]
        # Some Tmin values are missing; Arbitrary choice of 0.1 lower limit
        if np.isnan(Tmin): Tmin = Tmax * 0.1
        data = (Tc, Pc, a, b, c, d)
        add_model(Wagner_original.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_AntoineExtended:
        _, a, b, c, Tc, to, n, e, f, Tmin, Tmax = Psat_data_AntoineExtended[CAS]
        data = (Tc, to, a, b, c, n, e, f)
        add_model(TRC_Antoine_extended.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_Antoine:
        _, a, b, c, Tmin, Tmax = Psat_data_Antoine[CAS]
        data = (a, b, c)
        add_model(Antoine.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_Perrys2_8:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = Psat_data_Perrys2_8[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(DIPPR_EQ101.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_VDI_PPDS_3:
        _, Tm, Tc, Pc, a, b, c, d = Psat_data_VDI_PPDS_3[CAS]
        data = (Tc, Pc, a, b, c, d)
        add_model(Wagner.functor.from_args(data), 0., Tc,)
    data = (Tb, Tc, Pc)
    if all(data):
        add_model(boiling_critical_relation.functor.from_args(data), 0., Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for f in (Lee_Kesler, Ambrose_Walton, Sanjari, Edalat):
            add_model(f.functor.from_args(data), 0., Tc)
vp.vapor_pressure_handle = vapor_pressure_handle