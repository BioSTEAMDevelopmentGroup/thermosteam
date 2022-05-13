# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from .units_of_measure import AbsoluteUnitsOfMeasure

__all__ = ('settings',)

def raise_no_thermo_error():
    raise RuntimeError("no available 'Thermo' object; "
                       "use settings.set_thermo")

def raise_no_flashpkg_error():
    raise RuntimeError("no available 'FlashConstructor' object; "
                       "use settings.set_flashpkg")
    
class Settings:
    __slots__ = (
        'impact_indicators',
        '_thermo',
        '_phase_names',
        '_debug',
        '_flashpkg',
    )
    
    def __init__(self):
        self.impact_indicators = {}
        self._thermo = None
        self._debug = False
        self._phase_names = {'s': 'Solid',
                             'l': 'Liquid',
                             'g': 'Gas',
                             'S': 'SOLID',
                             'L': 'LIQUID'}
        self._flashpkg = None
    
    def define_impact_indicator(self, key, units):
        self.impact_indicators[key] = AbsoluteUnitsOfMeasure(units)
    
    def get_impact_indicator_units(self, key):
        try:
            return tmo.settings.impact_indicators[key]
        except KeyError:
            raise ValueError(
                f"impact indicator key '{key}' has no defined units; "
                 "units can be defined through `thermosteam.settings.define_impact_indicator`"
            )
    
    @property
    def debug(self):
        """[bool] If True, additional checks may raise errors at runtime."""
        return self._debug
    @debug.setter
    def debug(self, debug):
        self._debug = bool(debug)
    
    @property
    def phase_names(self):
        """[dict] All phase definitions."""
        return self._phase_names
    
    def get_default_thermo(self, thermo):
        """
        Return default Thermo object if `thermo` is None. Otherwise, return 
        the same object.
        
        """
        return thermo if thermo else self.get_thermo()
    
    def get_default_chemicals(self, chemicals):
        """
        Return a default CompiledChemicals object.
        If `chemicals` is a Chemicals object, return the same object.
        
        """
        if isinstance(chemicals, tmo.Chemicals):
            chemicals.compile()
        elif not chemicals:
            thermo = settings._thermo
            if not thermo: raise_no_thermo_error()
            chemicals = thermo.chemicals
        else:
            raise ValueError("chemicals must be a 'Chemicals' object")
        return chemicals
    
    def get_default_mixture(self, mixture):
        """
        Return a default Mixture object.
        If `mixture` is a Mixture object, return the same object.
        
        """
        if not mixture:
            thermo = settings.thermo
            if not thermo: raise_no_thermo_error()
            mixture = thermo.mixture
        return mixture
    
    def get_default_flashpkg(self, flashpkg):
        """
        Return a default FlashConstructor object.
        If `flash` is a FlashConstructor object, return the same object.
        
        """
        if not flashpkg: flashpkg = self.get_flashpkg()
        return flashpkg
    
    def get_flashpkg(self):
        flashpkg = self._flashpkg
        if not flashpkg: self._flashpkg = flashpkg = tmo.equilibrium.FlashPackage()
        return flashpkg
    
    def get_default_flasher(self, flasher):
        if not flasher:
            flashpkg = self.get_flashpkg()
            return flashpkg.flahser()
        return flasher
    
    def flasher(self, IDs=None, N_liquid=None, N_solid=None):
        return self.get_flashpkg().flasher(IDs, N_liquid, N_solid)
    
    def get_thermo(self):
        """Return a default Thermo object."""
        thermo = self._thermo
        if not thermo: raise_no_thermo_error()
        return thermo
    
    def set_thermo(self, thermo, cache=None, skip_checks=False, ideal=False):
        """
        Set the default Thermo object. If `thermo` is not a Thermo object,
        an attempt is made to convert it to one.
        
        Parameters
        ----------
        thermo : Thermo or Iterable[Chemical or str]
            A Thermo object or iterable of chemicals or chemical IDs.
        cache : bool, optional
            Wether or not to use cached chemicals.
        skip_checks : bool, optional
            Whether to skip checks for missing or invalid properties.
            
        """
        if not isinstance(thermo, (tmo.Thermo, tmo.IdealThermo)):
            thermo = tmo.Thermo(thermo, cache=cache, skip_checks=skip_checks)
        if ideal: thermo = thermo.ideal()
        self._thermo = thermo
    
    def set_flashpkg(self, flashpkg):
        """
        Set the default FlashPackage object. 
        
        Parameters
        ----------
        flashpkg : FlashPackage
            A FlashPackage object that predefines algorithms for equilibrium 
            calculations.

        """
        if isinstance(flashpkg, tmo.equilibrium.FlashPackage):
            self._flashpkg = flashpkg
        else:
            raise ValueError(f"flashpkg must be a FlashPackage object, not a '{type(flashpkg).__name__}'")
    
    def get_chemicals(self):
        """Return a default Chemicals object."""
        thermo = self.get_thermo()
        return thermo.chemicals
    
    def __repr__(self): # pragma: no cover
        return "<Settings>"
    
settings = Settings()