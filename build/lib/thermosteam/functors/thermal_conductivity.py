# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp2d
from ..base import InterpolatedTDependentModel, TDependentModel, TPDependentHandleBuilder, PhaseTPPropertyBuilder, kappa
from .._constants import R, N_A, k
from math import log, exp
from .utils import CASDataReader
from ..functional import horner_polynomial
from .miscdata import _VDISaturationDict, VDI_tabular_data
from .dippr import DIPPR_EQ100, DIPPR_EQ102


__all__ = ('ThermalConductivity',
           'Sheffy_Johnson', 'Sato_Riedel', 'Lakshmi_Prasad', 'Gharagheizi_liquid',
           'Nicola_original', 'Nicola', 'Bahadori_liquid', 'Mersmann_Kind_thermal_conductivity_liquid',
           'DIPPR9G', 'Missenard', 'Eucken', 'Eucken_modified', 'DIPPR9B_linear',
           'DIPPR9B_monoatomic', 'DIPPR9B_nonlinear', 'Chung', 'Eli_Hanley',
           'Gharagheizi_gas', 'Bahadori_gas', 'Stiel_Thodos_dense',
           'Eli_Hanley_dense', )

read = CASDataReader(__file__, 'Thermal Conductivity')

_Perrys2_314 = read('Table 2-314 Vapor Thermal Conductivity of Inorganic and Organic Substances.tsv')
_Perrys2_315 = read('Table 2-315 Thermal Conductivity of Inorganic and Organic Liquids.tsv')
_VDI_PPDS_9 = read('VDI PPDS Thermal conductivity of saturated liquids.tsv')
_VDI_PPDS_10 = read('VDI PPDS Thermal conductivity of gases.tsv')
### Purely CSP Methods - Liquids

@kappa.l
def Sheffy_Johnson(T, M, Tm):
    return 1.951*(1 - 0.00126*(T - Tm))/(Tm**0.216*M**0.3)

@kappa.l
def Sato_Riedel(T, MW, Tb, Tc):
    Tr = T/Tc
    Tbr = Tb/Tc
    return 1.1053*(3. + 20.*(1 - Tr)**(2/3.))*MW**-0.5/(3. + 20.*(1 - Tbr)**(2/3.))

@kappa.l
def Lakshmi_Prasad(T, MW):
    return 0.0655 - 0.0005*T + (1.3855 - 0.00197*T)*MW**-0.5

@kappa.l
def Gharagheizi_liquid(T, MW, Tb, Pc, omega):
    Pc = Pc/1E5
    B = 16.0407*MW + 2.*Tb - 27.9074
    A = 3.8588*MW**8*(1.0045*B + 6.5152*MW - 8.9756)
    return 1E-4*(10.*omega + 2.*Pc - 2.*T + 4. + 1.908*(Tb + 1.009*B*B/(MW*MW))
        + 3.9287*MW**4*B**-4 + A*B**-8)

@kappa.l
def Nicola_original(T, MW, Tc, omega, Hfus):
    Tr = T/Tc
    Hfus = Hfus*1000
    return -0.5694 - 0.1436*Tr + 5.4893E-10*Hfus + 0.0508*omega + (1./MW)**0.0622

@kappa.l
def Nicola(T, MW, Tc, Pc, omega):
    Tr = T/Tc
    Pc = Pc/1E5
    return 0.5147*(-0.2537*Tr + 0.0017*Pc + 0.1501*omega + (1./MW)**0.2999)

@kappa.l
def Bahadori_liquid(T, MW):
    A = [-6.48326E-2, 2.715015E-3, -1.08580E-5, 9.853917E-9]
    B = [1.565612E-2, -1.55833E-4, 5.051114E-7, -4.68030E-10]
    C = [-1.80304E-4, 1.758693E-6, -5.55224E-9, 5.201365E-12]
    D = [5.880443E-7, -5.65898E-9, 1.764384E-11, -1.65944E-14]
    X = MW
    Y = T
    a = A[0] + B[0]*X + C[0]*X**2 + D[0]*X**3
    b = A[1] + B[1]*X + C[1]*X**2 + D[1]*X**3
    c = A[2] + B[2]*X + C[2]*X**2 + D[2]*X**3
    d = A[3] + B[3]*X + C[3]*X**2 + D[3]*X**3
    return a + b*Y + c*Y**2 + d*Y**3

@kappa.l
def Mersmann_Kind_thermal_conductivity_liquid(T, MW, Tc, Vc, atoms):
    na = sum(atoms.values())
    lambda_star = 2/3.*(na + 40.*(1. - T/Tc)**0.5)
    Vc = Vc*1000 # m^3/mol to m^3/kmol
    N_A2 = N_A*1000 # Their avogadro's constant is per kmol
    kl = lambda_star*(k*Tc)**1.5*N_A2**(7/6.)*Vc**(-2/3.)/Tc*MW**-0.5
    return kl

@TPDependentHandleBuilder
def ThermalConductivityLiquid(handle, CAS, MW, Tm, Tb, Tc, Pc, omega, Hfus):
    add_model = handle.add_model
    if CAS in _Perrys2_315:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = _Perrys2_315[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(DIPPR_EQ100.from_args(data), Tmin, Tmax)
    if CAS in _VDI_PPDS_9:
        _,  A, B, C, D, E = _VDI_PPDS_9[CAS]
        add_model(horner_polynomial.from_kwargs({'coeffs':(E, D, C, B, A)}))
    if CAS in _VDISaturationDict:
        Ts, Ys = VDI_tabular_data(CAS, 'K (l)')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        add_model(InterpolatedTDependentModel(Ts, Ys, Tmin=Tmin, Tmax=Tmax))
    data = (MW, Tm)
    if all(data):
        # Works down to 0, has a nice limit at T = Tm+793.65 from Sympy
        add_model(Sheffy_Johnson.from_args(data), 0, 793.65)
    data = (MW, Tb, Tc)
    if all(data):
        add_model(Sato_Riedel.from_args(data))
    data = (MW, Tb, Pc, omega)
    if all(data):
        add_model(Gharagheizi_liquid.from_args(data), Tb, Tc)
    data = (MW, Tc, Pc, omega)
    if all(data):
        add_model(Nicola.from_args(data))
    data = (MW, Tc, omega, Hfus)
    if all(data):
        add_model(Nicola_original.from_args(data))
    if all((Tc, Pc)):
        data = (Tc, Pc, handle.models)
        add_model(DIPPR9G.from_args(data))
        add_model(Missenard.from_args(data))
    data = (MW,)
    if MW:
        add_model(Lakshmi_Prasad.from_args(data))
        add_model(Bahadori_liquid.from_args(data))


### Thermal Conductivity of Dense Liquids

@kappa.l
def DIPPR9G(T, P, Tc, Pc, kl_models):
    Tr = T/Tc
    Pr = P/Pc
    for kl in kl_models:
        if isinstance(kl, TDependentModel): break
    return kl.evaluate(T)*(0.98 + 0.0079*Pr*Tr**1.4 + 0.63*Tr**1.2*(Pr/(30. + Pr)))


Trs_Missenard = [0.8, 0.7, 0.6, 0.5]
Prs_Missenard = [1, 5, 10, 50, 100, 200]
Qs_Missenard = np.array([[0.036, 0.038, 0.038, 0.038, 0.038, 0.038],
                         [0.018, 0.025, 0.027, 0.031, 0.032, 0.032],
                         [0.015, 0.020, 0.022, 0.024, 0.025, 0.025],
                         [0.012, 0.0165, 0.017, 0.019, 0.020, 0.020]])
Qfunc_Missenard = interp2d(Prs_Missenard, Trs_Missenard, Qs_Missenard)

@kappa.l
def Missenard(T, P, Tc, Pc, kl_models):
    Tr = T/Tc
    Pr = P/Pc
    Q = float(Qfunc_Missenard(Pr, Tr))
    for kl in kl_models:
        if isinstance(kl, TDependentModel): break
    return kl.evaluate(T)*(1. + Q*Pr**0.7)


### Thermal Conductivity of Gases

@kappa.g
def Eucken(T, MW, Cp, mu):
    if callable(Cp):
        Cv = Cp(T) - R
    else:
        Cv = Cp - R
    if callable(mu):
        mu = mu(T)
    MW = MW/1000.
    return (1. + 9/4./(Cv/R))*mu*Cv/MW

@kappa.g
def Eucken_modified(T, MW, Cp, mu):
    if callable(Cp):
        Cv = Cp(T) - R
    else:
        Cv = Cp - R
    if callable(mu):
        mu = mu(T)
    MW = MW/1000.
    return (1.32 + 1.77/(Cv/R))*mu*Cv/MW

@kappa.g
def DIPPR9B_linear(T, MW, Cp, mu, Tc):
    if callable(Cp):
        Cv = (Cp(T) - R) * 1000. # J/mol/K to J/kmol/K
    else:
        Cv = (Cp - R) * 1000.  
    if callable(mu):
        mu = mu(T)
    Tr = T/Tc
    return mu/MW*(1.30*Cv + 14644 - 2928.80/Tr)

@kappa.g    
def DIPPR9B_monoatomic(T, MW, Cp, mu):
    if callable(Cp):
        Cv = (Cp(T) - R) * 1000. # J/mol/K to J/kmol/K
    else:
        Cv = (Cp - R) * 1000.  
    if callable(mu):
        mu = mu(T)
    return 2.5*mu*Cv/MW

@kappa.g
def DIPPR9B_nonlinear(T, MW, Cp, mu):
    if callable(Cp):
        Cv = (Cp(T) - R) * 1000. # J/mol/K to J/kmol/K
    else:
        Cv = (Cp - R) * 1000.  
    return mu/MW*(1.15*Cv + 16903.36)

@kappa.g
def Chung(T, MW, Tc, omega, Cp, mu):
    if callable(Cp):
        Cv = Cp(T) - R # J/mol/K to J/kmol/K
    else:
        Cv = Cp - R 
    if callable(mu):
        mu = mu(T)
    MW = MW/1000.
    alpha = Cv/R - 1.5
    beta = 0.7862 - 0.7109*omega + 1.3168*omega**2
    Z = 2 + 10.5*(T/Tc)**2
    psi = 1 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)
                      /(0.6366 + beta*Z + 1.061*alpha*beta))
    return 3.75*psi/(Cv/R)/MW*mu*Cv

@kappa.g
def Eli_Hanley(T, MW, Tc, Vc, Zc, omega, Cp):
    Cs = [2.907741307E6, -3.312874033E6, 1.608101838E6, -4.331904871E5, 
          7.062481330E4, -7.116620750E3, 4.325174400E2, -1.445911210E1, 2.037119479E-1]
    if callable(Cp):
        Cv = Cp(T) - R # J/mol/K to J/kmol/K
    else:
        Cv = Cp - R 
    Tr = T/Tc
    if Tr > 2: Tr = 2
    theta = 1 + (omega - 0.011)*(0.56553 - 0.86276*log(Tr) - 0.69852/Tr)
    psi = (1 + (omega-0.011)*(0.38560 - 1.1617*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    eta0 = 1E-7*sum([Ci*T0**((i+1. - 4.)/3.) for i, Ci in enumerate(Cs)])
    k0 = 1944*eta0

    H = (16.04/MW)**0.5*f**0.5*h**(-2/3.)
    etas = eta0*H*MW/16.04
    ks = k0*H
    return ks + etas/(MW/1000.)*1.32*(Cv - 1.5*R)

@kappa.g
def Gharagheizi_gas(T, MW, Tb, Pc, omega):
    Pc = Pc/1E4
    B = T + (2.*omega + 2.*T - 2.*T*(2.*omega + 3.2825)/Tb + 3.2825)/(2*omega + T - T*(2*omega+3.2825)/Tb + 3.2825) - T*(2*omega+3.2825)/Tb
    A = (2*omega + T - T*(2*omega + 3.2825)/Tb + 3.2825)/(0.1*MW*Pc*T) * (3.9752*omega + 0.1*Pc + 1.9876*B + 6.5243)**2
    return 7.9505E-4 + 3.989E-5*T - 5.419E-5*MW + 3.989E-5*A

@kappa.g
def Bahadori_gas(T, MW):
    A = [4.3931323468E-1, -3.88001122207E-2, 9.28616040136E-4, -6.57828995724E-6]
    B = [-2.9624238519E-3, 2.67956145820E-4, -6.40171884139E-6, 4.48579040207E-8]
    C = [7.54249790107E-6, -6.46636219509E-7, 1.5124510261E-8, -1.0376480449E-10]
    D = [-6.0988433456E-9, 5.20752132076E-10, -1.19425545729E-11, 8.0136464085E-14]
    X, Y = T, MW
    a = A[0] + B[0]*X + C[0]*X**2 + D[0]*X**3
    b = A[1] + B[1]*X + C[1]*X**2 + D[1]*X**3
    c = A[2] + B[2]*X + C[2]*X**2 + D[2]*X**3
    d = A[3] + B[3]*X + C[3]*X**2 + D[3]*X**3
    return a + b*Y + c*Y**2 + d*Y**3


### Thermal Conductivity of dense gases

@kappa.g
def Stiel_Thodos_dense(T,P, MW, Tc, Pc, Vc, Zc, Vg, kg_models):
    Vm = Vg(T, P)
    for i in kg_models:
        if isinstance(i, TDependentModel):
            kg = i.evaluate(T)
            break
    gamma = 210*(Tc*MW**3./(Pc/1E5)**4)**(1/6.)
    rhor = Vc/Vm
    if rhor < 0.5:
        term = 1.22E-2*(exp(0.535*rhor) - 1.)
    elif rhor < 2:
        term = 1.14E-2*(exp(0.67*rhor) - 1.069)
    else:
        # Technically only up to 2.8
        term = 2.60E-3*(exp(1.155*rhor) + 2.016)
    diff = term/Zc**5/gamma
    kg = kg + diff
    return kg

@kappa.g
def Eli_Hanley_dense(T, P, MW, Tc, Vc, Zc, omega, Cp, Vg):
    Cs = [2.907741307E6, -3.312874033E6, 1.608101838E6, -4.331904871E5,
          7.062481330E4, -7.116620750E3, 4.325174400E2, -1.445911210E1,
          2.037119479E-1]
    Tr = T/Tc
    if Tr > 2:
        Tr = 2
    Vm = Vg(T, P)
    if callable(Cp):
        Cvm = Cp(T) - R # J/mol/K to J/kmol/K
    else:
        Cvm = Cp - R 
    Vr = Vm/Vc
    if Vr > 2:
        Vr = 2
    theta = 1 + (omega - 0.011)*(0.09057 - 0.86276*log(Tr) + (0.31664 - 0.46568/Tr)*(Vr-0.5))
    psi = (1 + (omega-0.011)*(0.39490*(Vr-1.02355) - 0.93281*(Vr-0.75464)*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    rho0 = 16.04/(Vm*1E6)*h  # Vm must be in cm^3/mol here.
    eta0 = 1E-7*sum([Cs[i]*T0**((i+1-4)/3.) for i in range(len(Cs))])
    k1 = 1944*eta0
    b1 = -0.25276920E0
    b2 = 0.334328590E0
    b3 = 1.12
    b4 = 0.1680E3
    k2 = (b1 + b2*(b3 - log(T0/b4))**2)/1000.*rho0

    a1 = -7.19771
    a2 = 85.67822
    a3 = 12.47183
    a4 = -984.6252
    a5 = 0.3594685
    a6 = 69.79841
    a7 = -872.8833

    k3 = exp(a1 + a2/T0)*(exp((a3 + a4/T0**1.5)*rho0**0.1 + (rho0/0.1617 - 1)*rho0**0.5*(a5 + a6/T0 + a7/T0**2)) - 1)/1000.

    if T/Tc > 2:
        dtheta = 0
    else:
        dtheta = (omega - 0.011)*(-0.86276/T + (Vr-0.5)*0.46568*Tc/T**2)
    dfdT = Tc/190.4*dtheta
    X = ((1 - T/f*dfdT)*0.288/Zc)**1.5

    H = (16.04/MW)**0.5*f**0.5/h**(2/3.)
    ks = (k1*X + k2 + k3)*H

    ### Uses calculations similar to those for pure species here
    theta = 1 + (omega - 0.011)*(0.56553 - 0.86276*log(Tr) - 0.69852/Tr)
    psi = (1 + (omega-0.011)*(0.38560 - 1.1617*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    eta0 = 1E-7*sum([Cs[i]*T0**((i+1-4)/3.) for i in range(len(Cs))])
    H = (16.04/MW)**0.5*f**0.5/h**(2/3.)
    etas = eta0*H*MW/16.04
    k = ks + etas/(MW/1000.)*1.32*(Cvm-3*R/2.)
    return k

@kappa.g
def Chung_dense(T, P, MW, Tc, Vc, omega, Cp, Vg, mug, dipole, association=0):
    if callable(Cp):
        Cvm = Cp(T) - R # J/mol/K to J/kmol/K
    else:
        Cvm = Cp - R 
    mu = mug(T, P)
    Vm = Vg(T, P)
    ais = [2.4166E+0, -5.0924E-1, 6.6107E+0, 1.4543E+1, 7.9274E-1, -5.8634E+0, 9.1089E+1]
    bis = [7.4824E-1, -1.5094E+0, 5.6207E+0, -8.9139E+0, 8.2019E-1, 1.2801E+1, 1.2811E+2]
    cis = [-9.1858E-1, -4.9991E+1, 6.4760E+1, -5.6379E+0, -6.9369E-1, 9.5893E+0, -5.4217E+1]
    dis = [1.2172E+2, 6.9983E+1, 2.7039E+1, 7.4344E+1, 6.3173E+0, 6.5529E+1, 5.2381E+2]
    Tr = T/Tc
    mur = 131.3*dipole/(Vc*1E6*Tc)**0.5

    # From Chung Method
    alpha = Cvm/R - 1.5
    beta = 0.7862 - 0.7109*omega + 1.3168*omega**2
    Z = 2 + 10.5*(T/Tc)**2
    psi = 1 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)/(0.6366 + beta*Z + 1.061*alpha*beta))

    y = Vc/(6*Vm)
    B1, B2, B3, B4, B5, B6, B7 = [ais[i] + bis[i]*omega + cis[i]*mur**4 + dis[i]*association for i in range(7)]
    G1 = (1 - 0.5*y)/(1. - y)**3
    G2 = (B1/y*(1 - exp(-B4*y)) + B2*G1*exp(B5*y) + B3*G1)/(B1*B4 + B2 + B3)
    q = 3.586E-3*(Tc/(MW/1000.))**0.5/(Vc*1E6)**(2/3.)
    return 31.2*mu*psi/(MW/1000.)*(G2**-1 + B6*y) + q*B7*y**2*Tr**0.5*G2


@TPDependentHandleBuilder
def ThermalConductivityGas(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, Vg, Cp, mug):
    data = (MW, Tb, Pc, omega)
    add_model = handle.add_model
    if all(data):
        add_model(Gharagheizi_gas.from_args(data))
    data = (MW, Cp, mug, Tc)
    if CAS in _VDISaturationDict:
        Ts, Ys = VDI_tabular_data(CAS, 'K (g)')
        add_model(InterpolatedTDependentModel(Ts, Ys))
    if CAS in _VDI_PPDS_10:
        _,  *data = _VDI_PPDS_10[CAS].tolist()
        data.reverse()
        add_model(horner_polynomial.from_kwargs({'coeffs': data}))
    if all(data):
        add_model(DIPPR9B_linear.from_args(data))
    data = (MW, Tc, omega, Cp, mug)
    if all(data):
        add_model(Chung.from_args(data))
    data = (MW, Tc, Vc, Zc, omega, Cp)
    if all(data):
        add_model(Eli_Hanley.from_args(data))
    data = (MW, Cp, mug)
    if all(data):
        add_model(Eucken_modified.from_args(data))
        add_model(Eucken.from_args(data))
    data = (MW, Tc, Vc, Zc, omega, Cp, Vg)
    if all((MW, Tc, Vc, Zc, omega, Cp, Vg)):
        add_model(Eli_Hanley_dense.from_args(data))
    data = (MW, Tc, Vc, omega, Cp, Vg, mug, dipole)
    if all(data):
        add_model(Chung_dense.from_args(data))
    data = (MW, Tc, Pc, Vc, Zc, Vg, handle.models)
    if all(data):
        add_model(Stiel_Thodos_dense.from_args(data))
    # TODO: Fix propblem with values
    # if CAS in _Perrys2_314:
    #     _, *data, Tmin, Tmax = _Perrys2_314[CAS]
    #     add_model(DIPPR9B_linear(data), Tmin, Tmax)

ThermalConductivity = PhaseTPPropertyBuilder(None, ThermalConductivityLiquid, ThermalConductivityGas, 'kappa')

