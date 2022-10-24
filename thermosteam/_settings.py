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
from .units_of_measure import AbsoluteUnitsOfMeasure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .equilibrium import (
        ActivityCoefficients,
        FugacityCoefficients,
        PoyintingCorrectionFactors,
    )
    from .mixture import Mixture
try: import biosteam as bst
except: pass

__all__ = ('settings', 'ProcessSettings')

# %% Excecutables for dynamic programing of new stream utilities

get_unit_utility_flow_executable = '''
def {flowmethname}(self):
    """Return the {docname} flow rate [kg/hr]."""
    flow = 0.
    if {name} in self._inlet_utility_indices:
        flow += self._ins[self._inlet_utility_indices[{name}]].F_mass
    if {name} in self._outlet_utility_indices:
        flow += self._outs[self._outlet_utility_indices[{name}]].F_mass
    return flow

bst.Unit.{flowmethname} = {flowmethname}
'''

get_unit_utility_cost_executable = '''
def {costmethname}(self):
    """Return the {docname} cost [USD/hr]."""
    return bst.stream_utility_prices[name] * self.{flowmethname}()

bst.Unit.{costmethname} = {costmethname}
'''

get_system_utility_flow_executable = '''
def {flowmethname}(self):
    """Return the {docname} flow rate [kg/yr]."""
    return sum([i.get_utility_flow({name}) for i in self.cost_units]) * self.operating_hours

bst.System.{flowmethname} = {flowmethname}
'''

get_system_utility_cost_executable = '''
def {costmethname}(self):
    """Return the {docname} cost [USD/yr]."""
    return bst.stream_utility_prices[name] * self.{flowmethname}()

bst.System.{costmethname} = {costmethname}
'''


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
     <UtilityAgent: propane>]

    Access or change heating agents:
        
    >>> settings.heating_agents
    [<UtilityAgent: low_pressure_steam>,
     <UtilityAgent: medium_pressure_steam>,
     <UtilityAgent: high_pressure_steam>]

    Access or change the thermodynamic property package:
        
    >>> settings.set_thermo(['Water'], cache=True)
    >>> settings.thermo.show()
    Thermo(
        chemicals=CompiledChemicals([Water]),
        mixture=Mixture(
            rule='ideal', ...
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
    Mixture(
        rule='ideal', ...
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
    
    @property
    def CEPCI(self) -> float:
        """Chemical engineering plant cost index (defaults to 567.5 at 2017)."""
        return bst.CE
    @CEPCI.setter
    def CEPCI(self, CEPCI):
        bst.CE = CEPCI
    
    @property
    def utility_characterization_factors(self) ->  dict[tuple[str, str], tuple[float, AbsoluteUnitsOfMeasure]]:
        """Utility characterization factor data (value and units) by agent ID 
        and impact key."""
        return bst.HeatUtility.characterization_factors
    @utility_characterization_factors.setter
    def utility_characterization_factors(self, utility_characterization_factors):
        bst.HeatUtility.characterization_factors = utility_characterization_factors
    
    @property
    def cooling_agents(self) -> list[bst.UtilityAgent]:
        """All cooling utilities available."""
        return bst.HeatUtility.cooling_agents
    @cooling_agents.setter
    def cooling_agents(self, cooling_agents):
        bst.HeatUtility.cooling_agents = cooling_agents
        
    @property
    def heating_agents(self) -> list[bst.UtilityAgent]:
        """All heating utilities available."""
        return bst.HeatUtility.heating_agents
    @heating_agents.setter
    def heating_agents(self, heating_agents):
        bst.HeatUtility.heating_agents = heating_agents
        
    @property
    def stream_utility_prices(self) -> dict[str, float]:
        """Price of stream utilities [USD/kg] which are defined as 
        inlets and outlets to unit operations."""
        return bst.stream_utility_prices
    @stream_utility_prices.setter
    def stream_utility_prices(self, stream_utility_prices):
        bst.stream_utility_prices = stream_utility_prices
    
    @property
    def impact_indicators(self) -> dict[str, str]:
        """User-defined impact indicators and their units of measure."""
        return bst.impact_indicators
    @impact_indicators.setter
    def impact_indicators(self, impact_indicators):
        bst.impact_indicators = impact_indicators
    
    @property
    def electricity_price(self) -> float:
        """Electricity price [USD/kWhr]"""
        return bst.PowerUtility.price
    @electricity_price.setter
    def electricity_price(self, electricity_price):
        """Electricity price [USD/kWhr]"""
        bst.PowerUtility.price = electricity_price
    
    def register_utility(self, name, price):
        """Register new stream utility in BioSTEAM given the name and the price 
        [USD/kg]."""
        if name not in bst.stream_utility_prices:
            docname = name.lower()
            methname = docname.replace(' ', '_')
            flowmethname = f"get_{methname}_flow"
            costmethname = f"get_{methname}_cost"
            repname = repr(name)
            globs = {'bst': bst}
            flow_kwargs = dict(
                flowmethname=flowmethname,
                docname=docname,
                name=repname,
            )
            cost_kwargs = dict(
                costmethname=costmethname,
                flowmethname=flowmethname,
                docname=docname,
                name=repname,
            )
            
            # Unit
            exec(get_unit_utility_flow_executable.format(**flow_kwargs), globs)
            exec(get_unit_utility_cost_executable.format(**cost_kwargs), globs)
            
            # System
            exec(get_system_utility_flow_executable.format(**flow_kwargs), globs)
            exec(get_system_utility_cost_executable.format(**cost_kwargs), globs)
        bst.stream_utility_prices[name] = price
    
    def set_thermo(self, thermo: tmo.Thermo|Iterable[str|tmo.Chemical], 
            mixture: Optional[Mixture]=None,
            Gamma: Optional[type[ActivityCoefficients]]=None,
            Phi: Optional[type[FugacityCoefficients]]=None,
            PCF: Optional[type[PoyintingCorrectionFactors]]=None,
            cache: Optional[bool]=None,
            skip_checks: Optional[bool]=False, 
            ideal: Optional[bool]=False,
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
        
        """
        if not isinstance(thermo, (tmo.Thermo, tmo.IdealThermo)):
            thermo = tmo.Thermo(thermo, mixture=mixture, cache=cache, skip_checks=skip_checks,
                                Gamma=Gamma, Phi=Phi, PCF=PCF)
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
        self.impact_indicators[key] = AbsoluteUnitsOfMeasure(units)
    
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