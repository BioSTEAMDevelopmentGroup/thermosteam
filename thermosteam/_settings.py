# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from __future__ import annotations
import thermosteam as tmo
from typing import Optional, Iterable
from .units_of_measure import UnitsOfMeasure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .equilibrium import (
        ActivityCoefficients,
        FugacityCoefficients,
        PoyintingCorrectionFactors,
    )
    from .mixture import Mixture

__all__ = ('settings', 'ProcessSettings')

# %%

def raise_no_thermo_error():
    raise RuntimeError("no available 'Thermo' object; "
                       "use settings.set_thermo to set the default thermo object")

def raise_no_flashpkg_error():
    raise RuntimeError("no available 'FlashPackage' object; "
                       "use settings.set_flashpkg to set the default flash package object")
    
# %%
    
class ProcessSettings:
    """
    A compilation of all settings that may affect BioSTEAM results, including
    thermodynamic property packages, utility agents, and characterization factors.

    Examples
    --------
    Access or change the Chemical Engineering Plant Cost Index (CEPCI):
        
    >>> from biosteam import settings, Stream
    >>> settings.CEPCI # Defaults to average for year 2017
    567.5

    Access or change cooling agents:
        
    >>> settings.cooling_agents
    [<UtilityAgent: cooling_water>,
     <UtilityAgent: chilled_water>,
     <UtilityAgent: chilled_brine>,
     <UtilityAgent: propane>,
     <UtilityAgent: propylene>,
     <UtilityAgent: ethylene>]

    Access or change heating agents:
        
    >>> settings.heating_agents
    [<UtilityAgent: low_pressure_steam>,
     <UtilityAgent: medium_pressure_steam>,
     <UtilityAgent: high_pressure_steam>,
     <UtilityAgent: natural_gas>]

    Access or change the thermodynamic property package:
        
    >>> settings.set_thermo(['Water'], cache=True)
    >>> settings.thermo.show()
    Thermo(
        chemicals=CompiledChemicals([Water]),
        mixture=IdealMixture(...
            include_excess_energies=False
        ),
        Gamma=DortmundActivityCoefficients,
        Phi=IdealFugacityCoefficients,
        PCF=MockPoyintingCorrectionFactors
    )

    Access defined chemicals:
        
    >>> settings.chemicals
    CompiledChemicals([Water])

    Access defined mixture property algorithm:
        
    >>> settings.mixture.show()
    IdealMixture(...
        include_excess_energies=False
    )

    Create stream with default property package:
        
    >>> stream = Stream('stream', Water=2)
    >>> stream.thermo is settings.thermo
    True

    """
    __slots__ = (
        '_thermo',
        '_flashpkg',
    )
    
    def __new__(cls):
        return settings
    
    def set_thermo(self, thermo: tmo.Thermo|Iterable[str|tmo.Chemical], 
            mixture: Optional[Mixture]=None,
            Gamma: Optional[type[ActivityCoefficients]]=None,
            Phi: Optional[type[FugacityCoefficients]]=None,
            PCF: Optional[type[PoyintingCorrectionFactors]]=None,
            cache: Optional[bool]=None,
            skip_checks: Optional[bool]=False, 
            ideal: Optional[bool]=False,
            db: Optional[str]='default',
        ):
        """
        Set the default :class:`~thermosteam.Thermo` object. If `thermo` is 
        not a :class:`~thermosteam.Thermo` object, an attempt is made to 
        convert it to one.
        
        Parameters
        ----------
        thermo : 
            Thermodynamic property package.
        Gamma :
            Class for computing activity coefficients.
        Phi : 
            Class for computing fugacity coefficients.
        PCF : 
            Class for computing poynting correction factors.
        cache : 
            Whether or not to use cached chemicals.
        skip_checks : 
            Whether to skip checks for missing or invalid properties.
        ideal :
            Whether to use ideal phase equilibrium and mixture property 
            algorithms.
        db : str, optional
            Database to load any chemicals.
        
        """
        if not isinstance(thermo, (tmo.Thermo, tmo.IdealThermo)):
            thermo = tmo.Thermo(thermo, mixture=mixture, cache=cache, skip_checks=skip_checks,
                                Gamma=Gamma, Phi=Phi, PCF=PCF, db=db)
        if ideal: thermo = thermo.ideal()
        self._thermo = thermo
    
    def get_thermo(self): # For backwards compatibility
        return self.thermo
    def get_chemicals(self): # For backwards compatibility
        return self.chemicals
    
    @property
    def thermo(self) -> tmo.Thermo:
        """Default thermodynamic property package."""
        try:
            return self._thermo
        except AttributeError:
            raise_no_thermo_error()
    
    @property
    def chemicals(self) -> tmo.Chemicals:
        """Default chemicals."""
        return self.thermo.chemicals
    
    @property
    def mixture(self) -> tmo.Mixture:
        """Default mixture package."""
        return self.thermo.mixture
    
    def define_impact_indicator(self, key: str, units: str):
        """
        Define the units of measure for an LCA impact indicator key.
        
        Parameters
        ----------
        key : 
            Name of impact indicator.
        units :
            Units of measure for impact indicator.
            
        Notes
        -----
        This method is useful for setting characterization factors of 
        streams and utilities in different units of measure.
        
        LCA displacement allocation tables also use the impact indicator 
        units of measure defined here.
        
        Examples
        --------
        :doc:`../tutorial/Life_cycle_assessment`
        
        """
        self.impact_indicators[key] = UnitsOfMeasure(units)
    
    def get_impact_indicator_units(self, key):
        try:
            return tmo.settings.impact_indicators[key]
        except KeyError:
            raise ValueError(
                f"impact indicator key '{key}' has no defined units; "
                 "units can be defined through `settings.define_impact_indicator`"
            )
    
    def get_default_thermo(self, thermo):
        """
        Return the default :class:`~thermosteam.Thermo` object if `thermo` is None.
        Otherwise, return the same object. Otherwise, of `thermo` is a 
        :class:`~thermosteam.Thermo` object, return the same object.
        
        """
        if thermo is None:
            thermo = self.thermo
        elif not isinstance(thermo, (tmo.Thermo, tmo.IdealThermo)):
            raise ValueError("thermo must be a 'Thermo' object")
        return thermo
    
    def get_default_chemicals(self, chemicals):
        """
        Return the default :class:`~thermosteam.Chemicals` object is chemicals
        is None. Otherwise, if `chemicals` is a :class:`~thermosteam.Chemicals` 
        object, return the same object.
        
        """
        if isinstance(chemicals, tmo.Chemicals):
            chemicals.compile()
        elif chemicals is None:
            chemicals = self.chemicals
        else:
            raise ValueError("chemicals must be a 'Chemicals' object")
        return chemicals
    
    def get_default_mixture(self, mixture):
        """
        Return a default :class:`~thermosteam.Mixture` object.
        Otherwise, if `mixture` is a :class:`~thermosteam.Mixture` object,
        return the same object.
        
        """
        if mixture is None:
            mixture = self.mixture
        elif not isinstance(mixture, tmo.Mixture):
            raise ValueError("chemicals must be a 'Mixture' object")
        return mixture
    
    def get_default_flashpkg(self, flashpkg):
        """
        Warnings
        --------
        This method is not yet ready for users.
        
        """
        if not flashpkg: flashpkg = self.get_flashpkg()
        return flashpkg
    
    def get_flashpkg(self):
        """
        Warnings
        --------
        This method is not yet ready for users.
        
        """
        try:
            flashpkg = self._flashpkg
        except AttributeError:
            self._flashpkg = flashpkg = tmo.equilibrium.FlashPackage()
        return flashpkg
    
    def get_default_flasher(self, flasher):
        """
        Warnings
        --------
        This method is not yet ready for users.
        
        """
        if not flasher:
            flashpkg = self.get_flashpkg()
            return flashpkg.flahser()
        return flasher
    
    def flasher(self, IDs=None, N_liquid=None, N_solid=None):
        """
        Warnings
        --------
        This method is not yet ready for users.
        
        """
        return self.get_flashpkg().flasher(IDs, N_liquid, N_solid)
    
    def set_flashpkg(self, flashpkg):
        """
        Set the default FlashPackage object. 
        
        Parameters
        ----------
        flashpkg : FlashPackage
            A FlashPackage object that predefines algorithms for equilibrium 
            calculations.

        Warnings
        --------
        This method is not yet ready for users.
        
        """
        if isinstance(flashpkg, tmo.equilibrium.FlashPackage):
            self._flashpkg = flashpkg
        else:
            raise ValueError(f"flashpkg must be a FlashPackage object, not a '{type(flashpkg).__name__}'")

#: 
settings: ProcessSettings = object.__new__(ProcessSettings)
