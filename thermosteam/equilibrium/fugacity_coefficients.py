# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .ideal import ideal
import thermosteam as tmo
from thermo.interaction_parameters import IPDB
from thermo import eos_mix
from fluids.constants import R_inv
import numpy as np

__all__ = ('FugacityCoefficients', 
           'IdealFugacityCoefficients')


class FugacityCoefficients:
    """
    Abstract class for the estimation of fugacity coefficients. Non-abstract subclasses should implement the following methods:
        
    __init__(self, chemicals: Iterable[:class:`~thermosteam.Chemical`]):
        Should use pure component data from chemicals to setup future calculations of fugacity coefficients.
    
    __call__(self, y: 1d array, T: float, P:float):
        Should accept an array of vapor molar compositions `y`, temperature `T` (in Kelvin), and pressure `P` (in Pascal), and return an array of fugacity coefficients. Note that the molar compositions must be in the same order as the chemicals defined when creating the FugacityCoefficients object.
        
    """
    __slots__ = ()
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"<{type(self).__name__}([{chemicals}])>"


@ideal
class IdealFugacityCoefficients(FugacityCoefficients):
    """
    Create an IdealFugacityCoefficients object that estimates all fugacity coefficients to be 1 when called with composition, temperature (K), and pressure (Pa).
    
    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ('_chemicals', '_phase')
    
    def __init__(self, chemicals, phase):
        self._chemicals = tuple(chemicals)
        if phase not in 'lg': 
            raise ValueError(f"phase must be either 'g' or 'l', not {phase:r!}")
        self._phase = phase
    
    @property
    def chemicals(self):
        """tuple[Chemical] All chemicals involved in the calculation of fugacity coefficients."""
        return self._chemicals

    @property
    def phase(self):
        """[str] Phase of fugacity coefficients."""
        return self._phase

    def __call__(self, y, T, P):
        return 1.


class GCEOSFugacityCoefficients(FugacityCoefficients):
    """
    Abstract class for estimating fugacity coefficients using a generic cubic equation of state 
    when called with composition, temperature (K), and pressure (Pa).
    
    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ('_chemicals', '_eos', '_phase')
    EOS = None # type[GCEOSMIX] Subclasses must implement this attribute.
    cache = None # [dict] Subclasses must implement this attribute.
    chemsep_db = None # Optional[str] Name of chemsep data base for interaction parameters.
    
    def __new__(cls, chemicals, phase):
        chemicals = tuple(chemicals)
        cache = cls.cache
        if chemicals in cache:
            return cache[chemicals, phase]
        elif phase not in 'lg': 
            raise ValueError(f"phase must be either 'g' or 'l', not {phase:r!}")
        else:
            self = super().__new__(cls)
            self._chemicals = chemicals
            self._phase = phase
            cache[chemicals, phase] = self
            return self
    
    @classmethod
    def subclass(cls, EOS, name=None):
        if name is None: name = EOS.__name__.replace('MIX', '') + 'FugacityCoefficients'
        return type(name, (cls,), dict(EOS=EOS, cache={}))
    
    @property
    def chemicals(self):
        """tuple[Chemical] All chemicals involved in the calculation of fugacity coefficients."""
        return self._chemicals
    
    @property
    def phase(self):
        """[str] Phase of fugacity coefficients."""
        return self._phase
    
    def eos(self, zs, T, P):
        if zs.__class__ is np.ndarray: zs = [float(i) for i in zs]
        phase = self._phase
        if phase == 'g':
            only_g = True
            only_l = False
        else:
            only_g = False
            only_l = True
        try:
            self._eos = eos = self._eos.to_TP_zs_fast(
                T=T, P=P, zs=zs, 
                only_g=only_g, only_l=only_l, 
                full_alphas=False
            )
        except:
            data = tmo.ChemicalData(self.chemicals)
            if self.chemsep_db is None:
                kijs = None
            else:
                try:
                    kijs = IPDB.get_ip_asymmetric_matrix(self.chemsep_db, data.CASs, 'kij')
                except:
                    kijs = None
            self._eos = eos = self.EOS(
                T=T, P=P, zs=zs, 
                Tcs=data.Tcs, Pcs=data.Pcs, omegas=data.omegas, kijs=kijs,
                only_g=only_g, only_l=only_l, 
            )
        return eos
    
    def __call__(self, x, T, P):
        eos = self.eos(x, T, P)
        try:
            if self._phase == 'g':
                log_phis = np.array(eos.dlnphi_dns(eos.Z_g)) + eos.G_dep_g * R_inv / T
            else:
                log_phis = np.array(eos.dlnphi_dns(eos.Z_l)) + eos.G_dep_l * R_inv / T
            return np.exp(log_phis)
        except:
            return 1.
    f = __call__
    args = ()

dct = globals()    
clsnames = []
fugacity_coefficient_classes = []
for name in ('PRMIX', 'SRKMIX', 'PR78MIX', 'VDWMIX', 'PRSVMIX',
             'PRSV2MIX', 'TWUPRMIX', 'TWUSRKMIX', 'APISRKMIX', 'IGMIX', 'RKMIX',
             'PRMIXTranslatedConsistent', 'PRMIXTranslatedPPJP', 'PRMIXTranslated',
             'SRKMIXTranslatedConsistent', 'PSRK', 'MSRKMIXTranslated',
             'SRKMIXTranslated'):
    cls = GCEOSFugacityCoefficients.subclass(getattr(eos_mix, name))
    clsname = cls.__name__
    clsnames.append(clsname)
    fugacity_coefficient_classes.append(cls)
    dct[clsname] = cls

dct['PRFugacityCoefficients'].chemsep_db = 'ChemSep PR'
dct['PR78FugacityCoefficients'].chemsep_db = 'ChemSep PR'
__all__ = (*__all__, *clsnames, 'fugacity_coefficient_classes')
del dct, clsnames
