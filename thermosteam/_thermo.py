# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from . import equilibrium as eq
from ._chemical import Chemical
from ._chemicals import Chemicals
from .mixture import Mixture, ideal_mixture
from .utils import read_only, cucumber
from ._settings import settings

__all__ = ('Thermo',)


@cucumber # Just means you can pickle it
@read_only
class Thermo:
    """
    Create a Thermo object that defines a thermodynamic property package
    
    Parameters
    ----------
    chemicals : Chemicals or Iterable[str]
        Pure component chemical data.
    mixture : Mixture, optional
        Calculates mixture properties.
    Gamma : ActivityCoefficients subclass, optional
        Class for computing activity coefficients.
    Phi : FugacityCoefficients subclass, optional
        Class for computing fugacity coefficients.
    PCF : PoyntingCorrectionFactor subclass, optional
        Class for computing poynting correction factors.
    
    Examples
    --------
    >>> from thermosteam import Thermo
    >>> Thermo(['Ethanol', 'Water'])
    Thermo(chemicals=CompiledChemicals([Ethanol, Water]), mixture=Mixture(rule='ideal mixing', ..., include_excess_energies=False), Gamma=DortmundActivityCoefficients, Phi=IdealFugacityCoefficients, PCF=IdealPoyintingCorrectionFactors)
    
    Attributes
    ----------
    chemicals : Chemicals or Iterable[str]
        Pure component chemical data.
    mixture : Mixture, optional
        Calculates mixture properties.
    Gamma : ActivityCoefficients subclass, optional
        Class for computing activity coefficients.
    Phi : FugacityCoefficients subclass, optional
        Class for computing fugacity coefficients.
    PCF : PoyntingCorrectionFactor subclass, optional
        Class for computing poynting correction factors.
    
    """
    __slots__ = ('chemicals', 'mixture', 'Gamma', 'Phi', 'PCF',
                 'ideal_equilibrium_thermo') 
    
    def __init__(self, chemicals, mixture=None,
                 Gamma=eq.DortmundActivityCoefficients,
                 Phi=eq.IdealFugacityCoefficients,
                 PCF=eq.IdealPoyintingCorrectionFactors):
        if not isinstance(chemicals, Chemicals): chemicals = Chemicals(chemicals)
        if mixture:
            assert isinstance(mixture, Mixture), (
                f"mixture must be a '{Mixture.__name__}' object")
        else:
            mixture = ideal_mixture(chemicals)
        chemicals.compile()
        if settings._debug:
            issubtype = issubclass
            assert issubtype(Gamma, eq.ActivityCoefficients), (
                f"Gamma must be a '{eq.ActivityCoefficients.__name__}' subclass")
            assert issubtype(Phi, eq.FugacityCoefficients), (
                f"Phi must be a '{eq.FugacityCoefficients.__name__}' subclass")
            assert issubtype(PCF, eq.PoyintingCorrectionFactors), (
                f"PCF must be a '{eq.PoyintingCorrectionFactors.__name__}' subclass")
        
        setattr = object.__setattr__
        if (Gamma is eq.IdealActivityCoefficients
            and Phi is eq.IdealFugacityCoefficients
            and PCF is eq.IdealPoyintingCorrectionFactors):
            ideal = self
        else:
            cls = self.__class__
            ideal = cls.__new__(cls)
            setattr(ideal, 'chemicals', chemicals)
            setattr(ideal, 'mixture', mixture)
            setattr(ideal, 'Gamma', eq.IdealActivityCoefficients)
            setattr(ideal, 'Phi', eq.IdealFugacityCoefficients)
            setattr(ideal, 'PCF', eq.IdealPoyintingCorrectionFactors)
            setattr(ideal, 'ideal_equilibrium_thermo', ideal)
        setattr(self, 'chemicals', chemicals)
        setattr(self, 'mixture', mixture)
        setattr(self, 'Gamma', Gamma)
        setattr(self, 'Phi', Phi)
        setattr(self, 'PCF', PCF)
        setattr(self, 'ideal_equilibrium_thermo', ideal)
    
    @property
    def equilibrium_model(self):
        if self.ideal_equilibrium_thermo is self:
            return "Raoult's law"
        elif self.Gamma in (eq.UNIFACActivityCoefficients, eq.DortmundActivityCoefficients):
            return "modified Raoult's law"
        else:
            return "unknown"
    
    def as_chemical(self, chemical):
        isa = isinstance
        if isa(chemical, str):
            try: 
                chemical = self.chemicals.retrieve(chemical)
            except:
                chemical = Chemical(chemical)
        elif not isa(chemical, Chemical):
            raise ValueError('can only convert string to chemical')
        return chemical
    
    def subgroup(self, IDs):
        chemicals = self.chemicals.subgroup(IDs)
        return type(self)(chemicals, None, self.Gamma, self.Phi, self.PCF)
    
    def __repr__(self):
        return f"{type(self).__name__}(chemicals={self.chemicals}, mixture={self.mixture}, Gamma={self.Gamma.__name__}, Phi={self.Phi.__name__}, PCF={self.PCF.__name__})"
    
    def show(self):
        try:
            mixture_info = self.mixture._info().replace('\n', '\n    ')
        except:
            mixture_info = str(self.mixture)
        print(f"{type(self).__name__}(\n"
              f"    chemicals={self.chemicals},\n"
              f"    mixture={mixture_info},\n"
              f"    Gamma={self.Gamma.__name__},\n"
              f"    Phi={self.Phi.__name__},\n"
              f"    PCF={self.PCF.__name__}\n"
               ")")
    _ipython_display_ = show