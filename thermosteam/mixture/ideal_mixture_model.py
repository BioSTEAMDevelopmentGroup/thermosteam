# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..base import display_asfunctor, SparseVector
from math import log

__all__ = (
    'IdealTMixtureModel', 'IdealTPMixtureModel', 'IdealEntropyModel',
    'IdealHvapModel', 'SinglePhaseIdealTMixtureModel', 'SinglePhaseIdealTPMixtureModel',
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
    >>> models = [i.V for i in chemicals]
    >>> mixture_model = IdealTPMixtureModel(models, 'V')
    >>> mixture_model
    <IdealTPMixtureModel(phase, mol, T, P) -> V [m^3/mol]>
    >>> mixture_model('l', [0.2, 0.8], 350, 101325)
    5.364...-05
    
    """
    __slots__ = ('var', 'models')

    def __init__(self, models, var):
        self.models = tuple(models)
        self.var = var

    def __call__(self, phase, mol, T, P):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        models = self.models
        return sum([j * models[i](phase, T, P) for i, j in mol.dct.items()])
    
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
    >>> import numpy as np
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> models = [i.S for i in chemicals]
    >>> mixture_model = IdealEntropyModel(models, 'S')
    >>> mixture_model
    <IdealEntropyModel(phase, mol, T, P) -> S [J/K/mol]>
    >>> mixture_model('l', np.array([0.2, 0.8]), 350, 101325)
    160.174
    
    """
    __slots__ = IdealTPMixtureModel.__slots__
    __init__ = IdealTPMixtureModel.__init__
    __repr__ = IdealTPMixtureModel.__repr__

    def __call__(self, phase, mol, T, P):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        total_mol = mol.sum()
        models = self.models
        return sum([j * models[i](phase, T, P) + j * log(j / total_mol) for i, j in mol.dct.items()])
    

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
    >>> models = [i.Cn for i in chemicals]
    >>> mixture_model = IdealTMixtureModel(models, 'Cn')
    >>> mixture_model
    <IdealTMixtureModel(phase, mol, T, P=None) -> Cn [J/mol/K]>
    >>> mixture_model('l', [0.2, 0.8], 350)
    122.529
    
    """
    __slots__ = IdealTPMixtureModel.__slots__
    __init__ = IdealTPMixtureModel.__init__
    __repr__ = IdealTPMixtureModel.__repr__

    def __call__(self, phase, mol, T, P=None):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        models = self.models
        return sum([j * models[i](phase, T) for i, j in mol.dct.items()])


class SinglePhaseIdealTMixtureModel:
    """
    Create an SinglePhaseIdealTMixtureModel object that calculates mixture properties
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
    >>> from thermosteam.mixture import SinglePhaseIdealTMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> models = [i.Psat for i in chemicals]
    >>> mixture_model = SinglePhaseIdealTMixtureModel(models, 'Psat')
    >>> mixture_model
    <SinglePhaseIdealTMixtureModel(mol, T, P=None) -> Psat [Pa]>
    >>> mixture_model([0.2, 0.8], 350)
    84499.12
    
    """
    __slots__ = IdealTPMixtureModel.__slots__
    __init__ = IdealTPMixtureModel.__init__
    __repr__ = IdealTPMixtureModel.__repr__

    def __call__(self, mol, T, P=None):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        models = self.models
        return sum([j * models[i](T) for i, j in mol.dct.items()])

class SinglePhaseIdealTPMixtureModel:
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
    >>> from thermosteam.mixture import SinglePhaseIdealTPMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> models = [i.V.l for i in chemicals]
    >>> mixture_model = SinglePhaseIdealTPMixtureModel(models, 'V')
    >>> mixture_model
    <SinglePhaseIdealTPMixtureModel(mol, T, P) -> V [m^3/mol]>
    >>> mixture_model([0.2, 0.8], 350, 101325)
    5.3640...-05
    
    """
    __slots__ = ('var', 'models')

    def __init__(self, models, var):
        self.models = tuple(models)
        self.var = var

    def __call__(self, mol, T, P):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        models = self.models
        return sum([j * models[i](T, P) for i, j in mol.dct.items()])
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"

class IdealHvapModel:
    __slots__ = ('chemicals',)
    var = 'Hvap'

    def __init__(self, chemicals):
        self.chemicals = chemicals

    def __call__(self, mol, T, P=None):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        chemicals = self.chemicals
        return sum([
            j * k.Hvap(T) for i, j in mol.dct.items()
            if not (k:=chemicals[i]).locked_state
        ])
    
    __repr__ = IdealTPMixtureModel.__repr__
