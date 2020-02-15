# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:18:27 2019

@author: yoelr
"""
from . import equilibrium as eq
from ._chemicals import Chemicals
from .mixture import Mixture, new_ideal_mixture
from .utils import read_only, cucumber
from ._settings import settings

__all__ = ('Thermo',)


@cucumber # Just means you can pickle it
@read_only
class Thermo:
    """Create a Thermo object that defines a thermodynamic property package
    
    Parameters
    ----------
    chemicals : Chemicals or Iterable[str]
        Pure component chemical data.
    mixture : Mixture, optional
        Calculates mixture properties.
    Gamma : ActivityCoefficients subclass, optional
        Class for computing activity coefficiente.
    Phi : FugacityCoefficients subclass, optional
        Class for computing fugacity coefficiente.
    PCF : PoyntingCorrectionFactor subclass, optional
        Class for computing poynting correction factors.
    
    Examples
    --------
    >>> from thermosteam import Thermo
    >>> Thermo(['Ethanol', 'Water'])
    Thermo(chemicals=CompiledChemicals([Ethanol, Water]), mixture=IdealMixture(...), Gamma=DortmundActivityCoefficients, Phi=IdealFugacityCoefficients, PCF=IdealPoyintingCorrectionFactors)
    
    Note how chemicals are compiled when it becomes part of a Thermo object.
    
    Attributes
    ----------
    chemicals : Chemicals or Iterable[str]
        Pure component chemical data.
    mixture : Mixture, optional
        Calculates mixture properties.
    Gamma : ActivityCoefficients subclass, optional
        Class for computing activity coefficiente.
    Phi : FugacityCoefficients subclass, optional
        Class for computing fugacity coefficiente.
    PCF : PoyntingCorrectionFactor subclass, optional
        Class for computing poynting correction factors.
    
    """
    __slots__ = ('chemicals', 'mixture', 'Gamma', 'Phi', 'PCF') 
    
    def __init__(self, chemicals, mixture=None,
                 Gamma=eq.DortmundActivityCoefficients,
                 Phi=eq.IdealFugacityCoefficients,
                 PCF=eq.IdealPoyintingCorrectionFactors):
        if not isinstance(chemicals, Chemicals): chemicals = Chemicals(chemicals)
        if mixture:
            assert isinstance(mixture, Mixture), (
                f"mixture must be a '{Mixture.__name__}' object")
        else:
            mixture = new_ideal_mixture(chemicals)
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
        setattr(self, 'chemicals', chemicals)
        setattr(self, 'mixture', mixture)
        setattr(self, 'Gamma', Gamma)
        setattr(self, 'Phi', Phi)
        setattr(self, 'PCF', PCF)    
    
    def __repr__(self):
        return f"{type(self).__name__}(chemicals={self.chemicals}, mixture={self.mixture}, Gamma={self.Gamma.__name__}, Phi={self.Phi.__name__}, PCF={self.PCF.__name__})"
    

    
    