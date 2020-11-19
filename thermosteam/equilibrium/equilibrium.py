# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..utils import thermo_user
from .._thermal_condition import ThermalCondition
from ..indexer import MolarFlowIndexer

@thermo_user
class Equilibrium:
    """
    Abstract class for equilibrium objects.
    
    Parameters
    ----------
    imol=None : MaterialIndexer, optional
        Molar chemical phase data is stored here.
    thermal_condition=None : ThermalCondition, optional
        The temperature and pressure used in calculations are stored here.
    thermo=None : Thermo, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.

    Notes
    -----
    Subclasses should implement a __call__ method to perform equilibrium 
    on the `phases` given when subclassing. 

    """
    __slots__ = ('_imol', '_thermal_condition', '_thermo')
    
    def __init_subclass__(cls, phases):
        try: 
            cls.__call__
        except:
            raise NotImplementedError("Equilibrium subclasses must implement a "
                                      "'__call__' method.")
        cls._phases = phases
    
    def __init__(self, imol=None, thermal_condition=None, thermo=None):
        self._load_thermo(thermo)
        chemicals = self._thermo.chemicals
        self._thermal_condition = thermal_condition or ThermalCondition(298.15, 101325.)
        if imol:
            if imol.chemicals is not chemicals:
                raise ValueError('imol chemicals must match thermo chemicals')
        else:
            imol = MolarFlowIndexer(phases=self._phases, chemicals=chemicals)
        self._imol = imol   
        
    @property
    def imol(self):
        """[MaterialIndexer] Chemical phase data."""
        return self._imol
    @property
    def thermal_condition(self):
        """[ThermalCondition] Temperature and pressure data."""
        return self._thermal_condition
    
    def show(self):
        print(self)
    _ipython_console_ = show
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs)
        tab = tabs * 4 * " "
        imol = format(self.imol, str(2*tabs))
        if tabs:
            dlim = "\n" + tab
        else:
            dlim = ", "
        return (f"{type(self).__name__}(imol={imol},{dlim}"
                f"thermal_condition={self.thermal_condition})")
    
    def __repr__(self):
        return self.__format__("1")
    