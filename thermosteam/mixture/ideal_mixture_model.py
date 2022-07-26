# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..base import display_asfunctor
from math import log

__all__ = (
    'IdealTMixtureModel', 'IdealTPMixtureModel', 'IdealEntropyModel'
)

class IdealTPMixtureModel:
    """
    Create an IdealTPMixtureModel object that calculates mixture properties
    based on the molar weighted sum of pure chemical properties.
    
    Parameters
    ----------
    models : Iterable[function(T, P)]
        Chemical property functions of temperature and pressure.
    var : str
        Description of thermodynamic variable returned.
    
    Notes
    -----
    :class:`Mixture` objects can contain IdealMixtureModel objects to establish
    as mixture model for thermodynamic properties.
    
    See also
    --------
    :class:`Mixture`
    :func:`~.mixture_builders.ideal_mixture`
    
    Examples
    --------
    >>> from thermosteam.mixture import IdealTPMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> models = [i.V.l for i in chemicals]
    >>> mixture_model = IdealTPMixtureModel(models, 'V')
    >>> mixture_model
    <IdealTPMixtureModel(mol, T, P) -> V [m^3/mol]>
    >>> mixture_model([0.2, 0.8], 350, 101325)
    5.376...e-05
    
    """
    __slots__ = ('var', 'models',)

    def __init__(self, models, var):
        self.models = tuple(models)
        self.var = var

    def __call__(self, mol, T, P):
        return sum([j * i(T, P) for i, j in zip(self.models, mol) if j])
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"

class IdealEntropyModel:
    """
    Create an IdealEntropyModel object that calculates entropy of a mixture 
    based on the molar weighted sum of pure chemical properties.
    
    Parameters
    ----------
    models : Iterable[function(T, P)]
        Chemical property functions of temperature and pressure.
    var : str
        Description of thermodynamic variable returned.
    
    Notes
    -----
    :class:`Mixture` objects can contain IdealMixtureModel objects to establish
    as mixture model for thermodynamic properties.
    
    See also
    --------
    :class:`Mixture`
    :func:`~.mixture_builders.ideal_mixture`
    
    Examples
    --------
    >>> from thermosteam.mixture import IdealEntropyModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> models = [i.S.l for i in chemicals]
    >>> mixture_model = IdealEntropyModel(models, 'S')
    >>> mixture_model
    <SMixtureModel(mol, T, P) -> S [kJ/K/mol]>
    >>> mixture_model([0.2, 0.8], 350, 101325)
    5.376...e-05
    
    """
    __slots__ = IdealTPMixtureModel.__slots__
    __init__ = IdealTPMixtureModel.__init__
    __repr__ = IdealTPMixtureModel.__repr__

    def __call__(self, mol, T, P):
        total_mol = mol.sum()
        return sum([j * i(T, P) + j * log(j / total_mol) for i, j in zip(self.models, mol) if j])

class IdealTMixtureModel:
    """
    Create an IdealTMixtureModel object that calculates mixture properties
    based on the molar weighted sum of pure chemical properties.
    
    Parameters
    ----------
    models : Iterable[function(T, P)]
        Chemical property functions of temperature and pressure.
    var : str
        Description of thermodynamic variable returned.
    
    Notes
    -----
    :class:`Mixture` objects can contain IdealMixtureModel objects to establish
    as mixture model for thermodynamic properties.
    
    See also
    --------
    :class:`Mixture`
    
    Examples
    --------
    >>> from thermosteam.mixture import IdealTMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> models = [i.Psat for i in chemicals]
    >>> mixture_model = IdealTMixtureModel(models, 'Psat')
    >>> mixture_model
    <IdealTMixtureModel(mol, T, P=None) -> Psat [Pa]>
    >>> mixture_model([0.2, 0.8], 350)
    84914.8703877987
    
    """
    __slots__ = IdealTPMixtureModel.__slots__
    __init__ = IdealTPMixtureModel.__init__
    __repr__ = IdealTPMixtureModel.__repr__

    def __call__(self, mol, T, P=None):
        return sum([j * i(T) for i, j in zip(self.models, mol) if j])
