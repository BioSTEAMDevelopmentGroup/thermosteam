# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:37:35 2019

@author: yoelr
"""
from .base import PhaseZTProperty, PhaseZTPProperty, display_asfunctor
import numpy as np

__all__ = ('Mixture',
           'new_ideal_mixture',
           'IdealZTProperty',
           'IdealZTPProperty')

# %% Mixture methods

mixture_phaseT_methods = ('Cn',)
mixture_hidden_T_methods = ('_H',)
mixture_phaseTP_methods = ('mu', 'V', 'kappa')
mixture_hidden_phaseTP_methods = ('_H_excess', '_S_excess', '_S')
mixture_T_methods  = ('Hvap', 'sigma', 'epsilon')
mixture_methods = (*mixture_phaseT_methods,
                   *mixture_phaseTP_methods,
                   *mixture_hidden_T_methods,
                   *mixture_hidden_phaseTP_methods,
                   *mixture_T_methods)

# %% Cache

class ThermalCache:
    __slots__ = ('condition', 'nonzero', 'data')
    
    def __init__(self, condition, nonzero, data):
        self.condition = condition
        self.nonzero = nonzero
        self.data = data
        
    def __repr__(self):
        return f"{type(self).__name__}(condition={self.condition}, nonzero={self.nonzero}, data={self.data})"


# %% Utilities

def set_properties_at_T(data, properties, nonzero, T):
    data[nonzero] = [i(T) for i, j in zip(properties, nonzero) if j]

def set_properties_at_TP(data, properties, nonzero, TP):
    data[nonzero] = [i(*TP) for i, j in zip(properties, nonzero) if j]

# %% Mixture properties

class IdealZTPProperty:
    __slots__ = ('var', 'properties', '_cache')

    def __init__(self, properties, var):
        self.properties = tuple(properties)
        self.var = var
        self._cache = {}

    @classmethod
    def from_chemicals(cls, chemicals, var):
        getfield = getattr
        return cls([getfield(i, var) for i in chemicals], var)

    def at_TP(self, z, TP):
        cache = self._cache
        nonzero = z!=0
        TP = TP.tuple
        properties = self.properties
        if TP in cache:
            thermal_cache = cache[TP]
            data = thermal_cache.data
            if TP == thermal_cache.condition:
                cache_nonzero = thermal_cache.nonzero
                new_nonzero = (cache_nonzero != nonzero) & nonzero
                set_properties_at_TP(data, properties, new_nonzero, TP)
                cache_nonzero |= nonzero
            else:
                data[:] = 0
                set_properties_at_TP(data, properties, nonzero, TP)
                thermal_cache.nonzero = nonzero
                thermal_cache.condition = TP
        else:
            data = np.zeros_like(z)
            set_properties_at_TP(data, properties, nonzero, TP)
            cache[TP] = ThermalCache(TP, nonzero, data)
        return (z * data).sum()

    def __call__(self, z, T, P):
        z = np.asarray(z)
        data = np.zeros_like(z)
        set_properties_at_T(data, self.properties, z!=0, (T, P))
        return (z * data).sum()
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


class IdealZTProperty:
    __slots__ = IdealZTPProperty.__slots__
    __repr__ = IdealZTPProperty.__repr__

    __init__ = IdealZTPProperty.__init__
    from_chemicals = IdealZTPProperty.from_chemicals

    def at_TP(self, z, TP):
        cache = self._cache
        nonzero = z!=0
        T = TP.T
        properties = self.properties
        if TP in cache:
            thermal_cache = cache[TP]
            data = thermal_cache.data
            if T == thermal_cache.condition:
                cache_nonzero = thermal_cache.nonzero
                new_nonzero = (cache_nonzero != nonzero) & nonzero
                set_properties_at_T(data, properties, new_nonzero, T)
                cache_nonzero |= nonzero
            else:
                data[:] = 0
                set_properties_at_T(data, properties, nonzero, T)
                thermal_cache.nonzero = nonzero
                thermal_cache.condition = T
        else:
            data = np.zeros_like(z)
            cache[TP] = ThermalCache(T, nonzero, data)
            set_properties_at_T(data, properties, nonzero, T)
        return (z * data).sum()

    def __call__(self, z, T):
        z = np.asarray(z)
        data = np.zeros_like(z)
        set_properties_at_T(data, self.properties, z!=0, T)
        return (z * data).sum()


# %% Ideal mixture phase property
        
def group_properties_by_phase(phase_properties):
    hasfield = hasattr
    getfield = getattr
    iscallable = callable
    properties_by_phase = {'s': [],
                           'l': [],
                           'g': []}
    for phase, properties in properties_by_phase.items():
        for phase_property in phase_properties:
            if iscallable(phase_property) and hasfield(phase_property, phase):
                prop = getfield(phase_property, phase)
            else:
                prop = phase_property
            properties.append(prop)
    return properties_by_phase
    
def build_ideal_PhaseZTProperty(chemicals, var):
    setfield = setattr
    getfield = getattr
    phase_properties = [getfield(i, var) for i in chemicals]
    new = PhaseZTProperty.__new__(PhaseZTProperty)
    for phase, properties in group_properties_by_phase(phase_properties).items():
        setfield(new, phase, IdealZTProperty(properties, var))
    new.var = var
    return new


def build_ideal_PhaseZTPProperty(chemicals, var):
    setfield = setattr
    getfield = getattr
    phase_properties = [getfield(i, var) for i in chemicals]
    new = PhaseZTPProperty.__new__(PhaseZTPProperty)
    for phase, properties in group_properties_by_phase(phase_properties).items():
        setfield(new, phase, IdealZTPProperty(properties, var))
    new.var = var
    return new


# %% Ideal mixture


class Mixture:
    """
    Create an Mixture object for estimating mixture properties.
    
    Parameters
    ----------
    description : str
        Description of mixing rules used.
    Cn : PhaseZTProperty
        Molar heat capacity functor [J/mol/K].
    H : PhaseZTProperty
        Enthalpy functor [J/mol].
    S : PhaseZTPProperty
        Entropy functor [J/mol].
    H_excess : PhaseZTPProperty
        Excess enthalpy functor [J/mol].
    S_excess : PhaseZTPProperty
        Excess entropy functor [J/mol].
    mu : PhaseZTPProperty
        Dynamic viscosity functor [Pa*s].
    V : PhaseZTPProperty
        Molar volume functor [m^3/mol].
    kappa : PhaseZTPProperty
        Thermal conductivity functor [W/m/K].
    Hvap : ZTProperty
        Heat of vaporization functor [J/mol]
    sigma : ZTProperty
        Surface tension functor [N/m].
    epsilon : ZTProperty
        Relative permitivity functor [-]
    rigorous_energy_balance=True : bool
        Whether to rigorously solve for temperature
        in energy balance or simply approximate.
    include_excess_energies=False : bool
        Whether to include excess energies
        in enthalpy and entropy calculations.
    
    Attributes
    ----------
    description : str
        Description of mixing rules used.
    rigorous_energy_balance : bool
        Whether to rigorously solve for temperature
        in energy balance or simply approximate.
    include_excess_energies : bool
        Whether to include excess energies
        in enthalpy and entropy calculations.
    Cn(phase, z, T) : PhaseZTProperty
        Molar heat capacity functor [J/mol/K].
    mu(phase, T, P) : PhaseZTPProperty
        Dynamic viscosity functor [Pa*s].
    V(phase, T, P) : PhaseZTPProperty
        Molar volume functor [m^3/mol].
    kappa(phase, T, P) : PhaseZTPProperty
        Thermal conductivity functor [W/m/K].
    Hvap(phase, T, P) : ZTProperty
        Heat of vaporization functor [J/mol]
    sigma(phase, T, P) : ZTProperty
        Surface tension functor [N/m].
    epsilon(phase, T, P) : ZTProperty
        Relative permitivity [-]
    
    
    """
    __slots__ = ('description',
                 'rigorous_energy_balance',
                 'include_excess_energies',
                 *mixture_methods)
    
    def __init__(self, description, Cn, H, S, H_excess, S_excess,
                 mu, V, kappa, Hvap, sigma, epsilon,
                 rigorous_energy_balance=True, include_excess_energies=False):
        self.description = description
        self.rigorous_energy_balance = rigorous_energy_balance
        self.include_excess_energies = include_excess_energies
        self.Cn = Cn
        self.mu = mu
        self.V = V
        self.kappa = kappa
        self.Hvap = Hvap
        self.sigma = sigma
        self.epsilon = epsilon
        self._H = H
        self._S = S
        self._H_excess = H_excess
        self._S_excess = S_excess
    
    @property
    def Cn_at_TP(self):
        return self.Cn.at_TP
    @property
    def kappa_at_TP(self):
        return self.kappa.at_TP
    @property
    def mu_at_TP(self):
        return self.mu.at_TP
    @property
    def V_at_TP(self):
        return self.V.at_TP
    @property
    def sigma_at_TP(self):
        return self.sigma.at_TP
    @property
    def epsilon_at_TP(self):
        return self.epsilon.at_TP
    @property
    def Hvap_at_TP(self):
        return self.Hvap.at_TP
    
    def H(self, phase, z, T, P):
        """Return enthalpy in J/mol"""
        if self.include_excess_energies:
            return self._H(phase, z, T) + self._H_excess(phase, z, T, P)
        else:
            return self._H(phase, z, T)
    
    def S(self, phase, z, T, P):
        """Return entropy in J/mol"""
        if self.include_excess_energies:
            return self._S(phase, z, T, P) + self._S_excess(phase, z, T, P)
        else:
            return self._S(phase, z, T, P)
    
    def H_at_TP(self, phase, z, TP):
        H = self._H.at_TP(phase, z, TP)
        if self.include_excess_energies:
            H += self._H_excess.at_TP(phase, z, TP)
        return H
    
    def S_at_TP(self, phase, z, TP):
        S = self._S.at_TP(phase, z, TP)
        if self.include_excess_energies:
            S += self._S_excess.at_TP(phase, z, TP)
        return S
    
    def solve_T(self, phase, z, H, T_guess, P):
        """Solve for temperature in Kelvin"""
        # First approximation
        H_guess = self.H(phase, z, T_guess, P)
        if (H - H_guess) < 1e-3: return T_guess
        Cn = self.Cn(phase, z, T_guess)
        T = T_guess + (H - H_guess) / Cn
        if self.rigorous_energy_balance:
            # Solve enthalpy by iteration
            it = 3
            it2 = 0
            while abs(T - T_guess) > 0.05:
                T_guess = T
                if it == 3:
                    it = 0
                    it2 += 1
                    if it2 > 5: break # Its good enough, no need to find exact solution
                    Cn = self.Cn(phase, z, T)
                else:
                    it += 1
                T += (H - self.H(phase, z, T, P))/Cn
        return T
                
    def xsolve_T(self, phase_data, H, T_guess, P):
        """Solve for temperature in Kelvin"""
        # First approximation
        H_guess = self.xH(phase_data, T_guess, P)
        if (H - H_guess) < 1e-3: return T_guess
        Cn = self.xCn(phase_data, T_guess)
        T = T_guess + (H - H_guess)/Cn
        if self.rigorous_energy_balance:
            # Solve enthalpy by iteration
            it = 3
            it2 = 0
            while abs(T - T_guess) > 0.05:
                T_guess = T
                if it == 3:
                    it = 0
                    it2 += 1
                    if it2 > 5: break # Its good enough, no need to find exact solution
                    Cn = self.xCn(phase_data, T_guess)
                else:
                    it += 1
                T += (H - self.xH(phase_data, T, P))/Cn
        return T
    
    def xCn(self, phase_data, T):
        Cn = self.Cn
        return sum([Cn(phase, z, T) for phase, z in phase_data])
    
    def xH(self, phase_data, T, P):
        H = self._H
        H_total = sum([H(phase, z, T) for phase, z in phase_data])
        if self.include_excess_energies:
            H = self._H_excess
            H_total += sum([H(phase, z, T, P) for phase, z in phase_data])
        return H_total
    
    def xS(self, phase_data, T, P):
        S = self._S
        S_total = sum([S(phase, z, T, P) for phase, z in phase_data])
        if self.include_excess_energies:
            S = self._S_excess
            S_total += sum([S(phase, z, T, P) for phase, z in phase_data])
        return S_total
    
    def xV(self, phase_data, T, P):
        V = self.V
        return sum([V(phase, z, T, P) for phase, z in phase_data])
    
    def xmu(self, phase_data, T, P):
        mu = self.mu
        return sum([mu(phase, z, T, P) for phase, z in phase_data])
    
    def xkappa(self, phase_data, T, P):
        kappa = self.kappa
        return sum([kappa(phase, z, T, P) for phase, z in phase_data])
    
    def xCn_at_TP(self, phase_data, TP):
        Cn = self.Cn.at_TP
        return sum([Cn(phase, z, TP) for phase, z in phase_data])
    
    def xH_at_TP(self, phase_data, TP):
        H = self._H.at_TP
        H_total = sum([H(phase, z, TP) for phase, z in phase_data])
        if self.include_excess_energies:
            H_excess = self._H_excess.at_TP
            H_total += sum([H_excess(phase, z, TP) for phase, z in phase_data])
        return H_total
    
    def xS_at_TP(self, phase_data, TP):
        S = self.S.at_TP
        S_total = sum([S(phase, z, TP) for phase, z in phase_data])
        if self.include_excess_energies:
            S_excess = self._S_excess.at_TP
            S_total += sum([S_excess(phase, z, TP) for phase, z in phase_data])
        return S_total
    
    def xV_at_TP(self, phase_data, TP):
        V = self.V.at_TP
        return sum([V(phase, z, TP) for phase, z in phase_data])
    
    def xmu_at_TP(self, phase_data, TP):
        mu = self.mu.at_TP
        return sum([mu(phase, z, TP) for phase, z in phase_data])
    
    def xkappa_at_TP(self, phase_data, TP):
        k = self.k.at_TP
        return sum([k(phase, z, TP) for phase, z in phase_data])
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.description)}, ..., rigorous_energy_balance={self.rigorous_energy_balance}, include_excess_energies={self.include_excess_energies})"
    
    
def new_ideal_mixture(chemicals,
                      rigorous_energy_balance=True,
                      include_excess_energies=False):
    """
    Create a Mixture object that computes mixture properties using ideal mixing rules.
    
    Parameters
    ----------
    chemicals : Chemicals
        For retrieving pure component chemical data.
    rigorous_energy_balance=True : bool
        Whether to rigorously solve for temperature in energy balance or simply approximate.
    include_excess_energies=False : bool
        Whether to include excess energies in enthalpy and entropy calculations.

    """
    chemicals = tuple(chemicals)
    Cn =  build_ideal_PhaseZTProperty(chemicals, 'Cn')
    H =  build_ideal_PhaseZTProperty(chemicals, 'H')
    S = build_ideal_PhaseZTPProperty(chemicals, 'S')
    H_excess = build_ideal_PhaseZTPProperty(chemicals, 'H_excess')
    S_excess = build_ideal_PhaseZTPProperty(chemicals, 'S_excess')
    mu = build_ideal_PhaseZTPProperty(chemicals, 'mu')
    V = build_ideal_PhaseZTPProperty(chemicals, 'V')
    kappa = build_ideal_PhaseZTPProperty(chemicals, 'kappa')
    Hvap = IdealZTProperty.from_chemicals(chemicals, 'Hvap')
    sigma = IdealZTProperty.from_chemicals(chemicals, 'sigma')
    epsilon = IdealZTProperty.from_chemicals(chemicals, 'epsilon')
    return Mixture('ideal mixing rules', Cn, H, S, H_excess, S_excess,
                   mu, V, kappa, Hvap, sigma, epsilon,
                   rigorous_energy_balance, include_excess_energies)