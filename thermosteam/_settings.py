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

__all__ = ('settings',)

def raise_no_thermo_error():
    raise RuntimeError("no available 'Thermo' object; "
                       "use settings.set_thermo")

class Settings:
    __slots__ = ('_thermo',
                 '_phase_names',
                 '_debug',
    )
    
    def __init__(self):
        self._thermo = None
        self._debug = False
        self._phase_names = {'s': 'Solid',
                             'l': 'Liquid',
                             'g': 'Gas',
                             'S': 'SOLID',
                             'L': 'LIQUID',
                             'G': 'GAS'}
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
        Return a default Thermo object.
        If `thermo` is a Thermo object, return the same object.
        
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
    
    def get_thermo(self):
        """Return a default Thermo object."""
        thermo = self._thermo
        if not thermo: raise_no_thermo_error()
        return thermo
    
    def set_thermo(self, thermo, cache=None, skip_checks=False):
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
        self._thermo = thermo
    
    def get_chemicals(self):
        """Return a default Chemicals object."""
        thermo = self.get_thermo()
        return thermo.chemicals
    
    def __repr__(self): # pragma: no cover
        return "<Settings>"
    
settings = Settings()