# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena, <yoelcortes@gmail.com>, Yalin Li, <yalinli2@illinois.edu>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from thermosteam import functional as fn
import flexsolve as flx
from chemicals import elements
from warnings import warn
from free_properties import property_array
from . import (
    _parse as prs,
    _xparse as xprs,
)
from ..utils import chemicals_user
from .._phase import NoPhase, phase_tuple
from ..indexer import ChemicalIndexer, MaterialIndexer
from ..exceptions import InfeasibleRegion
import numpy as np
import pandas as pd

__all__ = ('Reaction', 'ReactionItem', 'ReactionSet', 
           'ParallelReaction', 'SeriesReaction', 'ReactionSystem',
           'Rxn', 'RxnS', 'RxnI', 'PRxn', 'SRxn', 'RxnSys')

def get_stoichiometric_string(stoichiometry, phases, chemicals):
    if phases:
        return xprs.get_stoichiometric_string(stoichiometry,
                                              phases,
                                              chemicals)
    else:
        return prs.get_stoichiometric_string(stoichiometry, 
                                             chemicals)

def check_material_feasibility(material: np.ndarray):
    if fn.infeasible(material):
        raise InfeasibleRegion('not enough reactants; reaction conversion')
    else:
        material[material < 0.] = 0. 

def set_reaction_basis(rxn, basis):
    if basis != rxn._basis:
        if basis == 'wt':
            rxn._stoichiometry *= rxn.MWs
        elif basis == 'mol':
            rxn._stoichiometry /= rxn.MWs
        else:
            raise ValueError("basis must be either by 'wt' or by 'mol'")
        rxn._rescale()
        rxn._basis = basis

def as_material_array(material, basis, phases, chemicals):
    isa = isinstance
    if isa(material, np.ndarray):
        return material, None
    elif isa(material, tmo.Stream):
        if (phases or len(material.phases) != 1) and material.phases != phases:
            raise ValueError("reaction and stream phases do not match")
        if material.chemicals is chemicals:
            config = None
        else:
            config = material.chemicals, material.imol.reset_chemicals(chemicals)
        if basis == 'mol':
            return material.imol.data, config
        elif basis == 'wt':
            return material.imass.data, config
        else:
            raise ValueError("basis must be either 'mol' or 'wt'")
    else:
        raise ValueError('reaction material must be either an array or a stream')


# %%

@chemicals_user
class Reaction:
    """
    Create a Reaction object which defines a stoichiometric reaction and
    conversion. A Reaction object is capable of reacting the material
    flow rates of a :class:`thermosteam.Stream` object.
    
    Parameters
    ----------
    reaction : dict or str
               A dictionary of stoichiometric coefficients or a stoichiometric
               equation written as:
               i1 R1 + ... + in Rn -> j1 P1 + ... + jm Pm
    reactant : str
               ID of reactant compound.
    X : float
        Reactant conversion (fraction).
    chemicals=None : Chemicals, defaults to settings.chemicals.
        Chemicals corresponing to each entry in the stoichiometry array.
    basis='mol': {'mol', 'wt'}
        Basis of reaction.
    
    Other Parameters
    ----------------
    check_mass_balance=False: bool
        Whether to check if mass is not created or destroyed.
    correct_mass_balance=False: bool
        Whether to make sure mass is not created or destroyed by varying the 
        reactant stoichiometric coefficient.
    check_atomic_balance=False: bool
        Whether to check if stoichiometric balance by atoms cancel out.
    correct_atomic_balance=False: bool
        Whether to correct the stoichiometry according to the atomic balance.
    
    Notes
    -----
    A reaction object can react either a stream or an array. When a stream
    is passed, it reacts either the mol or mass flow rate according to
    the basis of the reaction object. When an array is passed, the array
    elements are reacted regardless of what basis they are associated with.
    
    Warning
    -------
    Negative conversions and conversions above 1.0 are fair game (allowed), but
    may lead to odd/infeasible values when reacting a stream.
    
    Examples
    --------
    Electrolysis of water to molecular hydrogen and oxygen:
    
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['H2O', 'H2', 'O2'], cache=True)
    >>> tmo.settings.set_thermo(chemicals)
    >>> reaction = tmo.Reaction('2H2O,l -> 2H2,g + O2,g', reactant='H2O', X=0.7)
    >>> reaction.show() # Note that the default basis is by 'mol'
    Reaction (by mol):
     stoichiometry             reactant    X[%]
     H2O,l -> H2,g + 0.5 O2,g  H2O,l       70.00
    >>> reaction.reactant # The reactant is a tuple of phase and chemical ID
    ('l', 'H2O')
    >>> feed = tmo.Stream('feed', H2O=100)
    >>> feed.phases = ('g', 'l') # Gas and liquid phases must be available
    >>> reaction(feed) # Call to run reaction on molar flow
    >>> feed.show() # Notice how 70% of water was converted to product
    MultiStream: feed
     phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): (g) H2   70
                         O2   35
                     (l) H2O  30
    
    Let's change to a per 'wt' basis:
    
    >>> reaction.basis = 'wt'
    >>> reaction.show()
    Reaction (by wt):
     stoichiometry                     reactant    X[%]
     H2O,l -> 0.112 H2,g + 0.888 O2,g  H2O,l       70.00
    
    Although we changed the basis, the end result is the same if we pass a 
    stream:
    
    >>> feed = tmo.Stream('feed', H2O=100)
    >>> feed.phases = ('g', 'l')
    >>> reaction(feed) # Call to run reaction on mass flow
    >>> feed.show() # Notice how 70% of water was converted to product
    MultiStream: feed
     phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): (g) H2   70
                         O2   35
                     (l) H2O  30
    
    If chemicals phases are not specified, Reaction objects can 
    react a any single phase Stream object (regardless of phase):
    
    >>> reaction = tmo.Reaction('2H2O -> 2H2 + O2', reactant='H2O', X=0.7)
    >>> feed = tmo.Stream('feed', H2O=100, phase='g')
    >>> reaction(feed) 
    >>> feed.show() 
    Stream: feed
     phase: 'g', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): H2O  30
                     H2   70
                     O2   35
    
    Alternatively, it's also possible to react an array (instead of a stream):
    
    >>> import numpy as np
    >>> array = np.array([100., 0. , 0.])
    >>> reaction(array)
    >>> array
    array([30., 70., 35.])
    
    Reaction objects with the same reactant can be added together:
        
    >>> tmo.settings.set_thermo(['Glucose', 'Ethanol', 'H2O', 'O2', 'CO2'])
    >>> fermentation = tmo.Reaction('Glucose + O2 -> Ethanol + CO2', reactant='Glucose', X=0.7)
    >>> combustion = tmo.Reaction('Glucose + O2 -> H2O + CO2', reactant='Glucose', X=0.2)
    >>> mixed_reaction = fermentation + combustion
    >>> mixed_reaction.show()
    Reaction (by mol):
     stoichiometry                                    reactant    X[%]
     Glucose + O2 -> 0.778 Ethanol + 0.222 H2O + CO2  Glucose    90.00
    
    Note how conversions are added and the stoichiometry rescales to a per
    reactant basis. Conversly, reaction objects may be substracted as well:
    
    >>> combustion = mixed_reaction - fermentation
    >>> combustion.show()
    Reaction (by mol):
     stoichiometry              reactant    X[%]
     Glucose + O2 -> H2O + CO2  Glucose    20.00
    
    When a Reaction object is multiplied (or divided), a new Reaction object
    with the conversion multiplied (or divided) is returned:
        
    >>> combustion_multiplied = 2 * combustion
    >>> combustion_multiplied.show()
    Reaction (by mol):
     stoichiometry              reactant    X[%]
     Glucose + O2 -> H2O + CO2  Glucose    40.00
    >>> fermentation_divided = fermentation / 2
    >>> fermentation_divided.show()
    Reaction (by mol):
     stoichiometry                  reactant    X[%]
     Glucose + O2 -> Ethanol + CO2  Glucose    35.00
    
    """
    phases = MaterialIndexer.phases
    __slots__ = ('_basis',
                 '_phases',
                 '_chemicals',
                 '_X_index', 
                 '_stoichiometry', 
                 '_X')
    
    def __init__(self, reaction, reactant, X, 
                 chemicals=None, basis='mol', *,
                 phases=None,
                 check_mass_balance=False,
                 check_atomic_balance=False,
                 correct_atomic_balance=False,
                 correct_mass_balance=False):
        if basis in ('wt', 'mol'):
            self._basis = basis
        else:
            raise ValueError("basis must be either by 'wt' or by 'mol'")
        self.X = X
        chemicals = self._load_chemicals(chemicals)
        if reaction:
            self._phases = phases = phase_tuple(phases) if phases else xprs.get_phases(reaction)
            if phases:
                self._stoichiometry = stoichiometry = xprs.get_stoichiometric_array(reaction, phases, chemicals)
                reactant_index = self._chemicals.index(reactant)
                for phase_index, x in enumerate(stoichiometry[:, reactant_index]):
                    if x: break
                self._X_index = (phase_index, reactant_index)
            else:
                self._stoichiometry = prs.get_stoichiometric_array(reaction, chemicals)
                self._X_index = self._chemicals.index(reactant)
            self._rescale()
            if correct_atomic_balance:
                self.correct_atomic_balance()
            else:
                if correct_mass_balance:
                    self.correct_mass_balance()
                elif check_mass_balance:
                    self.check_mass_balance()
                if check_atomic_balance:
                    self.check_atomic_balance()
        else:
            self._stoichiometry = np.zeros(chemicals.size)
            self._X_index = self._chemicals.index(reactant)
    
    @property
    def reaction_chemicals(self):
        """Return all chemicals involved in the reaction."""
        return [i for i,j in zip(self._chemicals, self._stoichiometry) if j]
    
    def reset_chemicals(self, chemicals):
        phases = self.phases
        stoichiometry = self._stoichiometry
        reactant = self.reactant
        if phases:
            M, N = stoichiometry.shape
            new_stoichiometry = np.zeros([M, chemicals.size])
            IDs = self._chemicals.IDs
            for i in range(M):
                for j in range(N):
                    value = stoichiometry[i, j]
                    if value: new_stoichiometry[i, chemicals.index(IDs[j])] = value
            phase, reactant = reactant
            X_index = phases.index(phase), chemicals.index(reactant)
        else:
            new_stoichiometry = np.zeros(chemicals.size)
            IDs = self._chemicals.IDs
            for i in range(len(IDs)):
                value = stoichiometry[i]
                if value: new_stoichiometry[chemicals.index(IDs[i])] = value
            X_index = chemicals.index(reactant)
        self._chemicals = chemicals
        self._stoichiometry = new_stoichiometry
        self._X_index = X_index
    
    def __eq__(self, other):
        try:
            return all([
                self._basis == other._basis,
                self._phases == other._phases,
                self._chemicals is other._chemicals,
                self._X_index == other._X_index,
                (self._stoichiometry == other._stoichiometry).all(),
                self._X == other._X,
            ])
        except:
            return False
    
    def copy(self, basis=None):
        """Return copy of Reaction object."""
        copy = self.__new__(self.__class__)
        copy._basis = self._basis
        copy._phases = self._phases
        copy._stoichiometry = self._stoichiometry.copy()
        copy._X_index = self._X_index
        copy._chemicals = self._chemicals
        copy._X = self._X
        if basis: set_reaction_basis(copy, basis)
        return copy
    
    def has_reaction(self):
        return bool(self.X and self.stoichiometry.any())
    
    def _math_compatible_reaction(self, rxn, copy=True):
        basis = self.basis
        if copy or basis != rxn._basis: rxn = rxn.copy(basis)
        if self._chemicals is not rxn._chemicals:
            raise ValueError('chemicals must be the same to add/substract reactions')
        if self._phases != rxn._phases:
            raise ValueError('phases must be the same to add/substract reactions')
        if self._X_index != rxn._X_index:
            raise ValueError('reactants must be the same to add/substract reactions')
        return rxn
    
    def __radd__(self, rxn):
        return self + rxn
    
    def __add__(self, rxn):
        if rxn == 0 or not rxn.has_reaction(): return self.copy()
        rxn = self._math_compatible_reaction(rxn)
        stoichiometry = self._stoichiometry*self.X + rxn._stoichiometry*rxn.X
        rxn._stoichiometry = stoichiometry/-(stoichiometry[rxn._X_index])
        rxn.X = self.X + rxn.X
        return rxn
    
    def __iadd__(self, rxn):
        if not rxn.has_reaction(): return self
        rxn = self._math_compatible_reaction(rxn, copy=False)
        stoichiometry = self._stoichiometry*self.X + rxn._stoichiometry*rxn.X
        self._stoichiometry = stoichiometry/-(stoichiometry[self._X_index])
        self.X = self.X + rxn.X
        return self
    
    def __mul__(self, num):
        new = self.copy()
        new.X *= float(num)
        return new
    
    def __rmul__(self, num):
        return self.__mul__(num)
    
    def __imul__(self, num):
        self.X *= num
        return self
    
    def __truediv__(self, num):
        return self.__mul__(1./num)  
    
    def __itruediv__(self, num):
        return self.__imul__(1./num) 
    
    def __neg__(self):
        new = self.copy()
        new.X *= -1.
        return new
    
    def __sub__(self, rxn):
        if not rxn.has_reaction(): return self
        rxn = self._math_compatible_reaction(rxn)
        stoichiometry = self._stoichiometry*self.X - rxn._stoichiometry*rxn.X
        rxn._stoichiometry = stoichiometry/-(stoichiometry[rxn._X_index])
        rxn.X = self.X - rxn.X
        return rxn
    
    def __isub__(self, rxn):
        if not rxn.has_reaction(): return self
        rxn = self._math_compatible_reaction(rxn, copy=False)
        stoichiometry = self._stoichiometry*self.X + rxn._stoichiometry*rxn.X
        self._stoichiometry = stoichiometry/-(stoichiometry[self._X_index])
        self.X = self.X - rxn.X
        return self
    
    def __call__(self, material):
        material_array, config = as_material_array(
            material, self._basis, self._phases, self._chemicals
        )
        isproperty = isinstance(material_array, property_array)
        values = material_array.value if isproperty else material_array
        self._reaction(values)
        if tmo.reaction.CHECK_FEASIBILITY:
            check_material_feasibility(values)
        else:
            fn.remove_negligible_negative_values(values)
        if isproperty: material_array[:] = values
        if config: material._imol.reset_chemicals(*config)
        
    def force_reaction(self, material):
        """React material ignoring feasibility checks."""
        material_array, config = as_material_array(material,
                                           self._basis,
                                           self._phases,
                                           self._chemicals)
        isproperty = isinstance(material_array, property_array)
        values = material_array.value if isproperty else material_array
        self._reaction(values)
        fn.remove_negligible_negative_values(values)
        if isproperty: material_array[:] = values
        if config: material._imol.reset_chemicals(*config)
    
    def product_yield(self, product, basis=None):
        """Return yield of product per reactant."""
        product_index = self._chemicals.index(product)
        product_coefficient = self._stoichiometry[product_index]
        product_yield = product_coefficient * self.X
        if basis and self.basis != basis:
            chemicals_tuple = self._chemicals.tuple
            reactant_index = self._X_index
            MW_reactant = chemicals_tuple[reactant_index].MW
            MW_product = chemicals_tuple[product_index].MW
            if basis == 'wt':
                product_yield *= MW_reactant / MW_product
            elif basis == 'mol':
                product_yield *= MW_product / MW_reactant
            else:
                raise ValueError("basis must be either 'wt' or 'mol'; "
                                f"not {repr(basis)}")
        return product_yield
    
    def adiabatic_reaction(self, stream):
        """
        React stream material adiabatically, accounting for the change in enthalpy
        due to the heat of reaction.
        
        Examples
        --------
        Note how the stream temperature changes after each reaction due to the
        heat of reaction. Adiabatic combustion of hydrogen:
        
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['H2', 'O2', 'H2O'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> reaction = tmo.Reaction('2H2 + O2 -> 2H2O', reactant='H2', X=0.7)
        >>> s1 = tmo.Stream('s1', H2=10, O2=20, H2O=1000, T=373.15, phase='g')
        >>> s2 = tmo.Stream('s2')
        >>> s2.copy_like(s1) # s1 and s2 are the same
        >>> s1.show() # Before reaction
        Stream: s1
         phase: 'g', T: 373.15 K, P: 101325 Pa
         flow (kmol/hr): H2   10
                         O2   20
                         H2O  1e+03
        
        >>> reaction.show()
        Reaction (by mol):
         stoichiometry       reactant    X[%]
         H2 + 0.5 O2 -> H2O  H2         70.00
        
        >>> reaction(s1) 
        >>> s1.show() # After isothermal reaction
        Stream: s1
         phase: 'g', T: 373.15 K, P: 101325 Pa
         flow (kmol/hr): H2   3
                         O2   16.5
                         H2O  1.01e+03
        
        >>> reaction.adiabatic_reaction(s2)
        >>> s2.show() # After adiabatic reaction
        Stream: s2
         phase: 'g', T: 421.6 K, P: 101325 Pa
         flow (kmol/hr): H2   3
                         O2   16.5
                         H2O  1.01e+03
        
        Adiabatic combustion of H2 and CH4:
        
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['H2', 'CH4', 'O2', 'CO2', 'H2O'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> reaction = tmo.ParallelReaction([
        ...    #            Reaction definition          Reactant    Conversion
        ...    tmo.Reaction('2H2 + O2 -> 2H2O',        reactant='H2',  X=0.7),
        ...    tmo.Reaction('CH4 + O2 -> CO2 + 2H2O',  reactant='CH4', X=0.1)
        ... ])
        >>> s1 = tmo.Stream('s1', H2=10, CH4=5, O2=100, H2O=100, T=373.15, phase='g')
        >>> s2 = tmo.Stream('s2')
        >>> s1.show() # Before reaction
        Stream: s1
         phase: 'g', T: 373.15 K, P: 101325 Pa
         flow (kmol/hr): H2   10
                         CH4  5
                         O2   100
                         H2O  100
        
        >>> reaction.show()
        ParallelReaction (by mol):
        index  stoichiometry            reactant    X[%]
        [0]    H2 + 0.5 O2 -> H2O       H2         70.00
        [1]    CH4 + O2 -> CO2 + 2 H2O  CH4        10.00
        
        >>> reaction.adiabatic_reaction(s1)
        >>> s1.show() # After adiabatic reaction
        Stream: s1
         phase: 'g', T: 666.21 K, P: 101325 Pa
         flow (kmol/hr): H2   3
                         CH4  4.5
                         O2   96
                         CO2  0.5
                         H2O  108    
        
        Sequential combustion of CH4 and CO:
        
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['CH4', 'CO','O2', 'CO2', 'H2O'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> reaction = tmo.SeriesReaction([
        ...     #            Reaction definition                 Reactant       Conversion
        ...     tmo.Reaction('2CH4 + 3O2 -> 2CO + 4H2O',       reactant='CH4',    X=0.7),
        ...     tmo.Reaction('2CO + O2 -> 2CO2',               reactant='CO',     X=0.1)
        ...     ])
        >>> s1 = tmo.Stream('s1', CH4=5, O2=100, H2O=100, T=373.15, phase='g')
        >>> s1.show() # Before reaction
        Stream: s1
         phase: 'g', T: 373.15 K, P: 101325 Pa
         flow (kmol/hr): CH4  5
                         O2   100
                         H2O  100
        
        >>> reaction.show()
        SeriesReaction (by mol):
        index  stoichiometry               reactant    X[%]
        [0]    CH4 + 1.5 O2 -> CO + 2 H2O  CH4        70.00
        [1]    CO + 0.5 O2 -> CO2          CO         10.00
        
        >>> reaction.adiabatic_reaction(s1)
        >>> s1.show() # After adiabatic reaction
        Stream: s1
         phase: 'g', T: 649.84 K, P: 101325 Pa
         flow (kmol/hr): CH4  1.5
                         CO   3.15
                         O2   94.6
                         CO2  0.35
                         H2O  107
        
        """
        if not isinstance(stream, tmo.Stream):
            raise ValueError(f"stream must be a Stream object, not a '{type(stream).__name__}' object")
        Hnet = stream.Hnet
        self(stream)
        stream.H = Hnet - stream.Hf
    
    def _reaction(self, material_array):
        material_array += material_array[self._X_index] * self.X * self._stoichiometry
    
    @property
    def dH(self):
        """
        Heat of reaction at given conversion. Units are in either
        J/mol-reactant or J/g-reactant; depending on basis.
        
        Warning
        -------
        Latents heats of vaporization are not accounted for; only heats of 
        formation are included in this term. Note that heats of vaporization
        are temperature dependent and cannot be calculated using a Reaction
        object.
        
        """
        if self._basis == 'mol':
            Hfs = self._chemicals.Hf
        else:
            Hfs = self._chemicals.Hf / self.MWs
        return self.X * (Hfs * self._stoichiometry).sum()
    
    @property
    def X(self):
        """[float] Reaction converion as a fraction."""
        return self._X
    @X.setter
    def X(self, X):
        self._X = float(X)
    
    @property
    def stoichiometry(self):
        """[array] Stoichiometry coefficients."""
        return self._stoichiometry
    @property
    def istoichiometry(self):
        """[ChemicalIndexer] Stoichiometry coefficients."""
        stoichiometry = self._stoichiometry
        if stoichiometry.ndim == 1:
            return tmo.indexer.ChemicalIndexer.from_data(self._stoichiometry,
                                                         chemicals=self._chemicals,
                                                         check_data=False)
        else:
            return tmo.indexer.MaterialIndexer.from_data(self._stoichiometry,
                                                         phases=self._phases,
                                                         chemicals=self._chemicals,
                                                         check_data=False)
    
    @property
    def reactant(self):
        """[str] Reactant associated to conversion."""
        if self._phases:
            phase_index, chemical_index = self._X_index
            return self._phases[phase_index], self._chemicals.IDs[chemical_index]
        else:
            return self._chemicals.IDs[self._X_index] 

    @property
    def MWs(self):
        """[1d array] Molecular weights of all chemicals [g/mol]."""
        return self._chemicals.MW

    @property
    def basis(self):
        """{'mol', 'wt'} Basis of reaction"""
        return self._basis
    @basis.setter
    def basis(self, basis):
        set_reaction_basis(self, basis)

    def _get_stoichiometry_by_wt(self):
        """Return stoichiometry by weight."""
        if self._basis == 'mol':
            stoichiometry_by_wt = self._stoichiometry * self.MWs
        else:
            stoichiometry_by_wt = self._stoichiometry
        return stoichiometry_by_wt
        
    def _get_stoichiometry_by_mol(self):
        """Return stoichiometry on a molar basis."""
        if self._basis == 'wt':
            stoichiometry_by_mol = self._stoichiometry / self.MWs
        else:
            stoichiometry_by_mol = self._stoichiometry
        return stoichiometry_by_mol
    
    def mass_balance_error(self):
        """Return error in stoichiometric mass balance. If positive,
        mass is being created. If negative, mass is being destroyed."""
        stoichiometry_by_wt = self._get_stoichiometry_by_wt()
        return stoichiometry_by_wt.sum()
    
    def atomic_balance_error(self):
        """Return a dictionary of errors in stoichiometric atomic balances. 
        If value is positive, the atom is being created. If negative, the atom 
        is being destroyed."""
        stoichiometry_by_mol = self._get_stoichiometry_by_mol()
        formula_array = self.chemicals.formula_array
        unbalanced_array = formula_array @ stoichiometry_by_mol
        return elements.array_to_atoms(unbalanced_array)
    
    def check_mass_balance(self, tol=1e-3):
        """Check that stoichiometric mass balance is correct."""
        error = self.mass_balance_error()
        if abs(error) > tol:
            raise RuntimeError("material stoichiometry is unbalanced by "
                              f"{error} g / mol-reactant")
    
    def check_atomic_balance(self, tol=1e-3):
        """Check that stoichiometric atomic balance is correct."""
        atoms = self.atomic_balance_error()
        if abs(sum(atoms.values())) > tol: 
            raise RuntimeError("atomic stoichiometry is unbalanced by the "
                               "following molar stoichiometric coefficients:\n "
            + "\n ".join([f"{symbol}: {value}" for symbol, value in atoms.items()])
        )
    
    def correct_mass_balance(self, variable=None):
        """
        Make sure mass is not created or destroyed by varying the 
        reactant stoichiometric coefficient.
        """
        if variable:
            index = self.chemicals.get_index(variable)
        else:
            index = self._X_index
        stoichiometry_by_wt = self._get_stoichiometry_by_wt()
        if self.phases: 
            stoichiometry_by_wt = stoichiometry_by_wt.sum(0)
        def f(x):
            stoichiometry_by_wt[index] = x
            return stoichiometry_by_wt.sum()
        
        x = flx.aitken_secant(f, 1)
        if self._basis == 'mol': 
            x /= self.MWs[index]
            if self.phases:
                row = np.where(self._stoichiometry[:, index])
                self._stoichiometry[row, index] = x
            else:
                self._stoichiometry[index] = x 
        self._rescale()
    
    def correct_atomic_balance(self, constants=None):
        """
        Correct stoichiometry coffecients to satisfy atomic balance.
        
        Parameters
        ----------
        constants : str, optional
            IDs of chemicals for which stoichiometric coefficients are held constant.
        
        Examples
        --------
        Balance glucose fermentation to ethanol:
        
        >>> import thermosteam as tmo
        >>> from biorefineries import lipidcane as lc
        >>> tmo.settings.set_thermo(lc.chemicals)
        >>> fermentation = tmo.Reaction('Glucose + O2 -> Ethanol + CO2',
        ...                             reactant='Glucose',  X=0.9)
        >>> fermentation.correct_atomic_balance()
        >>> fermentation.show()
        Reaction (by mol):
         stoichiometry                 reactant    X[%]
         Glucose -> 2 Ethanol + 2 CO2  Glucose    90.00
        
        Balance methane combustion:
            
        >>> combustion = tmo.Reaction('CH4 + O2 -> Water + CO2',
        ...                           reactant='CH4', X=1)
        >>> combustion.correct_atomic_balance()
        >>> combustion.show()
        Reaction (by mol):
         stoichiometry                reactant    X[%]
         2 O2 + CH4 -> 2 Water + CO2  CH4       100.00
         
        Balance electrolysis of water (with chemical phases specified):
            
        >>> electrolysis = tmo.Reaction('H2O,l -> H2,g + O2,g',
        ...                             chemicals=tmo.Chemicals(['H2O', 'H2', 'O2']),
        ...                             reactant='H2O', X=1)
        >>> electrolysis.correct_atomic_balance()
        >>> electrolysis.show()
        Reaction (by mol):
         stoichiometry             reactant    X[%]
         H2O,l -> H2,g + 0.5 O2,g  H2O,l     100.00
        
        Note that if the reaction is underspecified, there are infinite
        ways to balance the reaction and a runtime error is raised:
        
        >>> rxn_underspecified = tmo.Reaction('CH4 + Glucose + O2 -> Water + CO2',
        ...                                   reactant='CH4', X=1)
        >>> rxn_underspecified.correct_atomic_balance()
        Traceback (most recent call last):
        RuntimeError: reaction stoichiometry is underspecified; pass the 
        `constants` argument to the `<Reaction>.correct_atomic_balance` method 
        to specify which stoichiometric coefficients to hold constant
        
        Chemical coefficients can be held constant to prevent this error:
        
        >>> rxn_underspecified = tmo.Reaction('CH4 + Glucose + O2 -> Water + CO2',
        ...                                   reactant='CH4', X=1)
        >>> rxn_underspecified.correct_atomic_balance(['Glucose', 'CH4'])
        >>> rxn_underspecified.show()
        Reaction (by mol):
         stoichiometry                            reactant    X[%]
         Glucose + 8 O2 + CH4 -> 8 Water + 7 CO2  CH4       100.00
        
        """
        stoichiometry_by_mol = self._get_stoichiometry_by_mol()
        phases = self.phases
        if phases:
            stoichiometry_by_mol = stoichiometry_by_mol.sum(0)
        chemicals = self.chemicals
        if constants:
            if isinstance(constants, str): constants = [constants]
            constants = set(constants)
            constant_index = chemicals.indices(constants)
        else:
            constant_index = [self._X_index[1] if phases else self._X_index]
        chemical_index, = np.where(stoichiometry_by_mol)
        chemical_index = np.setdiff1d(chemical_index, constant_index)
        formula_array = chemicals.formula_array
        b = - (formula_array[:, constant_index]
               * stoichiometry_by_mol[constant_index]).sum(1, keepdims=True)
        atomic_bool_index = np.any(formula_array * stoichiometry_by_mol, axis=1)
        atomic_index, = np.where(atomic_bool_index)
        b = b[atomic_index, :]
        A = formula_array[atomic_index, :][:, chemical_index]
        M_atoms, N_chemicals = A.shape
        if M_atoms != N_chemicals:
            x, _, rank, *_ = np.linalg.lstsq(A, b, rcond=None)
            if N_chemicals > rank:
                raise RuntimeError(
                     "reaction stoichiometry is underspecified (i.e. there are "
                     "infinite ways to balance the reaction); pass the "
                     "`constants` argument to the `<Reaction>.correct_atomic_balance` "
                     "method to specify which stoichiometric coefficients to hold constant"
                )
            residual_mass = ((A @ x - b) * self.MWs).sum()
            if residual_mass > 1e-6:
                warn(f'atomic balance was solved with a residual mass error of {residual_mass} g / mol of reactant')
        else:
            x = np.linalg.solve(A, b)
        
        stoichiometry_by_mol[chemical_index] = x.flatten()
        by_wt = self._basis == 'wt'
        stoichiometry = stoichiometry_by_mol * self.MWs if by_wt else stoichiometry_by_mol
        if phases: 
            self._stoichiometry[:] = (self._stoichiometry != 0.) * stoichiometry
        elif by_wt: 
            self._stoichiometry[:] = stoichiometry
        self._rescale()
    
    def _rescale(self):
        """Scale stoichiometry to a per reactant basis."""
        new_scale = -self._stoichiometry[self._X_index]
        if new_scale == 0.:
            raise RuntimeError(f"reactant '{self.reactant}' does not participate in stoichiometric reaction")
        self._stoichiometry /= new_scale
    
    def to_df(self, index=None):
        columns = [f'Stoichiometry (by {self.basis})', 'Reactant', 'Conversion [%]']
        stoichiometry = get_stoichiometric_string(self.stoichiometry, self.phases, self.chemicals)
        reactant = self.reactant
        conversion = 100. * self.X
        df = pd.DataFrame(data=[[stoichiometry, reactant, conversion]], columns=columns, index=[index] if index else None)
        df.index.name = 'Reaction'
        return df
    
    def __repr__(self):
        reaction = get_stoichiometric_string(self.stoichiometry, self.phases, self.chemicals)
        return f"{type(self).__name__}('{reaction}', reactant='{self.reactant}', X={self.X:.3g}, basis={repr(self.basis)})"
    
    def _info(self):
        info = f"{type(self).__name__} (by {self.basis}):"
        rxn = get_stoichiometric_string(self.stoichiometry, self.phases, self.chemicals)
        if self.phases:
            phase, ID = self.reactant
            cmp = ID + ',' + phase
        else:
            cmp = self.reactant
        lrxn = len(rxn)
        lcmp = len(cmp)
        maxrxnlen = max([13, lrxn]) + 2
        maxcmplen = max([8, lcmp]) + 2
        X = self.X
        info += "\nstoichiometry" + " "*(maxrxnlen - 13) + "reactant" + " "*(maxcmplen - 8) + '  X[%]'
        rxn_spaces = " "*(maxrxnlen - lrxn)
        cmp_spaces = " "*(maxcmplen - lcmp)
        info += f"\n{rxn}{rxn_spaces}{cmp}{cmp_spaces}{X*100: >6.2f}"
        return info
    
    def show(self):
        print(self._info())
    _ipython_display_ = show


class ReactionItem(Reaction):
    """
    Create a ReactionItem object from the a ReactionSet and reaction index.
    
    Parameters
    ----------
    rxnset : ReactionSet
    index : int
        Index of reaction.
        
    """
    __slots__ = ('_index')
    phases = MaterialIndexer.phases
    
    def __init__(self, rxnset, index):
        self._stoichiometry = rxnset._stoichiometry[index]
        self._phases = rxnset._phases
        self._basis = rxnset._basis
        self._X = rxnset._X
        self._chemicals = rxnset._chemicals
        self._X_index = rxnset._X_index[index]
        self._index = index
    
    @property
    def basis(self):
        """{'mol', 'wt'} Basis of reaction"""
        return self._basis
    @basis.setter
    def basis(self, basis):
        raise TypeError('cannot change basis of reaction item')
    
    def copy(self, basis=None):
        """Return copy of Reaction object."""
        copy = Reaction.__new__(Reaction)
        copy._basis = self._basis
        copy._phases = self._phases
        copy._stoichiometry = self._stoichiometry.copy()
        copy._X_index = self._X_index
        copy._chemicals = self._chemicals
        copy._X = self.X
        if basis: set_reaction_basis(copy, basis)
        return copy
    
    @property
    def X(self):
        """[float] Reaction converion as a fraction."""
        return self._X[self._index]
    @X.setter
    def X(self, X):
        self._X[self._index] = X
        

class ReactionSet:
    """
    Create a ReactionSet that contains all reactions and conversions as an array.
    
    Parameters
    ----------
    reactions : Iterable[Reaction]
    
    """
    __slots__ = Reaction.__slots__
    __eq__ = Reaction.__eq__
    copy = Reaction.copy
    phases = MaterialIndexer.phases
    _get_stoichiometry_by_mol = Reaction._get_stoichiometry_by_mol
    _get_stoichiometry_by_wt = Reaction._get_stoichiometry_by_wt
    force_reaction = Reaction.force_reaction
    adiabatic_reaction = Reaction.adiabatic_reaction
    __call__ = Reaction.__call__
    
    def __init__(self, reactions):
        if not reactions: raise ValueError('no reactions passed')
        phases_set = set([i.phases for i in reactions])
        if len(phases_set) > 1:
            raise ValueError('all reactions must implement the same phases')
        self._phases, = phases_set
        chemicals = {i.chemicals for i in reactions}
        try: self._chemicals, = chemicals
        except: raise ValueError('all reactions must have the same chemicals')
        basis = {i.basis for i in reactions}
        try: self._basis, = basis
        except: raise ValueError('all reactions must have the same basis')
        self._stoichiometry = np.array([i._stoichiometry for i in reactions])
        self._X = np.array([i.X for i in reactions])
        X_index = [i._X_index for i in reactions]
        self._X_index = tuple(X_index) if self._phases else np.array(X_index)
        
    def __getitem__(self, index):
        stoichiometry = self._stoichiometry[index]
        if (self.phases and stoichiometry.ndim == 2) or stoichiometry.ndim == 1:
            return ReactionItem(self, index)
        else:
            rxnset = self.__new__(self.__class__)
            rxnset._basis = self._basis
            rxnset._phases = self._phases
            rxnset._stoichiometry = stoichiometry
            rxnset._X = self._X[index]
            rxnset._X_index = self._X_index[index]
            rxnset._chemicals = self._chemicals
            return rxnset
    
    def reset_chemicals(self, chemicals):
        phases = self.phases
        stoichiometry = self._stoichiometry
        reactants = self.reactants
        if phases:
            A, B, C = stoichiometry.shape
            new_stoichiometry = np.zeros([A, B, chemicals.size])
            IDs = self._chemicals.IDs
            for i in range(A):
                for j in range(B):
                    for k in range(C):
                        value = stoichiometry[i, j, k]
                        if value: new_stoichiometry[i, j, chemicals.index(IDs[k])] = value
            X_index = tuple([(phases.index(i), chemicals.index(j)) for i, j in reactants])
        else:
            A, B = stoichiometry.shape
            new_stoichiometry = np.zeros([A, chemicals.size])
            IDs = self._chemicals.IDs
            for i in range(A):
                for j in range(B):
                    value = stoichiometry[i, j]
                    if value: new_stoichiometry[i, chemicals.index(IDs[j])] = value
            X_index = tuple([chemicals.index(i) for i in reactants])
        self._chemicals = chemicals
        self._stoichiometry = new_stoichiometry
        self._X_index = X_index
    
    @property
    def reaction_chemicals(self):
        """Return all chemicals involved in the reaction."""
        return [i for i,j in zip(self._chemicals, self._stoichiometry.any(axis=0)) if j]
    
    @property
    def basis(self):
        """{'mol', 'wt'} Basis of reaction"""
        return self._basis
    @basis.setter
    def basis(self, basis):
        raise TypeError('cannot change basis of reaction set')
    
    @property
    def X(self):
        """[1d array] Reaction converions."""
        return self._X
    @X.setter
    def X(self, X):
        """[1d array] Reaction converions."""
        if X is not self._X: self._X[:] = X
    
    @property
    def chemicals(self):
        """[Chemicals] Chemicals corresponing to each entry in the stoichiometry array."""
        return self._chemicals
    @property
    def stoichiometry(self):
        """[2d array] Stoichiometry coefficients."""
        return self._stoichiometry
    
    @property
    def reactants(self):
        """tuple[str] Reactants associated to conversion."""
        IDs = self._chemicals.IDs
        phases = self._phases
        X_index = self._X_index
        if phases:
            return tuple([(phases[i], IDs[j]) for i,j in X_index])
        else:
            return tuple([IDs[i] for i in X_index])
    
    @property
    def MWs(self):
        """[2d array] Molecular weights of all chemicals."""
        return self._chemicals.MW[np.newaxis, :]
    
    def to_df(self, index=None):
        columns = [f'Stoichiometry (by {self.basis})', 'Reactant', 'Conversion [%]']
        chemicals = self._chemicals
        phases = self._phases
        rxns = [get_stoichiometric_string(i, phases, chemicals) for i in self._stoichiometry]
        cmps = [ID + ',' + phase for phase, ID in self.reactants] if phases else self.reactants
        Xs = self.X
        data = list(zip(rxns, cmps, Xs))
        df = pd.DataFrame(data, columns=columns, index=index if index else None)
        if isinstance(self, ParallelReaction):
            df.index.name = 'Parallel reaction'
        elif isinstance(self, SeriesReaction):
            df.index.name = 'Reaction in series'
        return df
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join([repr(i).replace('ReactionItem', 'Reaction') for i in self])}])"
    
    def _info(self, index_name='index'):
        info = f"{type(self).__name__} (by {self.basis}):"
        chemicals = self._chemicals
        phases = self._phases
        length = len
        string = str
        rxns = [get_stoichiometric_string(i, phases, chemicals) for i in self._stoichiometry]
        maxrxnlen = max([13, *[length(i) for i in rxns]]) + 2
        cmps = [ID + ',' + phase for phase, ID in self.reactants] if phases else self.reactants
        maxcmplen = max([8, *[length(i) for i in cmps]]) + 2
        Xs = self.X
        N = len(Xs)
        maxnumspace = max(length(string(N)) + 1, len(index_name))
        info += f"\n{index_name}" + " "*(max(2, length(string(N)))) + "stoichiometry" + " "*(maxrxnlen - 13) + "reactant" + " "*(maxcmplen - 8) + '  X[%]'
        for N, rxn, cmp, X in zip(range(N), rxns, cmps, Xs):
            rxn_spaces = " "*(maxrxnlen - length(rxn))
            cmp_spaces = " "*(maxcmplen - length(cmp))
            num = string(N)
            numspace = (maxnumspace - length(num)) * " "
            info += f"\n[{N}]{numspace}{rxn}{rxn_spaces}{cmp}{cmp_spaces}{X*100: >6.2f}"
        return info
    _ipython_display_ = show = Reaction.show
        

class ParallelReaction(ReactionSet):
    """
    Create a ParallelReaction object from Reaction objects. When called, 
    it returns the change in material due to all parallel reactions.
    
    Parameters
    ----------
    reactions : Iterable[Reaction]
    
    Examples
    --------
    Run two reactions in parallel:
    
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['H2', 'Ethanol', 'CH4', 'O2', 'CO2', 'H2O'], cache=True)
    >>> tmo.settings.set_thermo(chemicals)
    >>> kwargs = dict(phases='lg', correct_atomic_balance=True)
    >>> reaction = tmo.ParallelReaction([
    ...    #            Reaction definition                    Reactant             Conversion
    ...    tmo.Reaction('H2,g + O2,g -> 2H2O,g',               reactant='H2',       X=0.7, **kwargs),
    ...    tmo.Reaction('Ethanol,l + O2,g -> CO2,g + 2H2O,g',  reactant='Ethanol',  X=0.1, **kwargs)
    ... ])
    >>> reaction.reactants # Note that reactants are tuples of phase and ID pairs.
    (('g', 'H2'), ('l', 'Ethanol'))
    
    >>> reaction.show()
    ParallelReaction (by mol):
    index  stoichiometry                            reactant     X[%]
    [0]    H2,g + 0.5 O2,g -> H2O,g                 H2,g        70.00
    [1]    3 O2,g + Ethanol,l -> 2 CO2,g + 3 H2O,g  Ethanol,l   10.00
    
    >>> s1 = tmo.MultiStream('s1', T=373.15, 
    ...                      l=[('Ethanol', 10)],
    ...                      g=[('H2', 10), ('CH4', 5), ('O2', 100), ('H2O', 10)])
    >>> s1.show() # Before reaction
    MultiStream: s1
     phases: ('g', 'l'), T: 373.15 K, P: 101325 Pa
     flow (kmol/hr): (g) H2       10
                         CH4      5
                         O2       100
                         H2O      10
                     (l) Ethanol  10
    
    >>> reaction(s1)
    >>> s1.show() # After isothermal reaction
    MultiStream: s1
     phases: ('g', 'l'), T: 373.15 K, P: 101325 Pa
     flow (kmol/hr): (g) H2       3
                         CH4      5
                         O2       93.5
                         CO2      2
                         H2O      20
                     (l) Ethanol  9
    
    Reaction items are accessible:
    
    >>> reaction[0].show()
    ReactionItem (by mol):
     stoichiometry             reactant    X[%]
     H2,g + 0.5 O2,g -> H2O,g  H2,g       70.00
    
    Note that changing the conversion of a reaction item changes the 
    conversion of its parent reaction set:
        
    >>> reaction[0].X = 0.5
    >>> reaction.show()
    ParallelReaction (by mol):
    index  stoichiometry                            reactant     X[%]
    [0]    H2,g + 0.5 O2,g -> H2O,g                 H2,g        50.00
    [1]    3 O2,g + Ethanol,l -> 2 CO2,g + 3 H2O,g  Ethanol,l   10.00
    
    Reactions subsets can be made as well:
        
    >>> reaction[:1].show()
    ParallelReaction (by mol):
    index  stoichiometry             reactant    X[%]
    [0]    H2,g + 0.5 O2,g -> H2O,g  H2,g       50.00
    
    Get net reaction conversion of reactants as a material indexer:
        
    >>> mi = reaction.X_net(indexer=True)
    >>> mi.show()
    MaterialIndexer:
     (g) H2        0.5
     (l) Ethanol   0.1
    >>> mi['g', 'H2']
    0.5
    
    If no phases are specified for a reaction set, the `X_net` property returns
    a ChemicalIndexer:
    
    >>> kwargs = dict(correct_atomic_balance=True)
    >>> reaction = tmo.ParallelReaction([
    ...    #            Reaction definition            Reactant             Conversion
    ...    tmo.Reaction('H2 + O2 -> 2H2O',             reactant='H2',       X=0.7, **kwargs),
    ...    tmo.Reaction('Ethanol + O2 -> CO2 + 2H2O',  reactant='Ethanol',  X=0.1, **kwargs)
    ... ])
    >>> ci = reaction.X_net(indexer=True)
    >>> ci.show()
    ChemicalIndexer:
     H2       0.7
     Ethanol  0.1
    >>> ci['H2']
    0.7
    
    """
    __slots__ = ()
    
    def _reaction(self, material_array):
        reacted = self._X * np.array([material_array[i] for i in self._X_index], float)
        if self._phases:
            material_array += (reacted[:, np.newaxis, np.newaxis] * self._stoichiometry).sum(0)
        else:
            material_array += reacted @ self._stoichiometry

    def reduce(self):
        """
        Return a new Parallel reaction object that combines reaction 
        with the same reactant together, reducing the number of reactions.
        """
        rxn_dict = {i: [] for i in set(self._X_index)}
        for i in self: rxn_dict[i._X_index].append(i)
        for key, rxns in rxn_dict.items():
            rxn, *rxns = rxns
            rxn = rxn.copy()
            for i in rxns: rxn += i
            rxn_dict[key] = rxn 
        return self.__class__(rxn_dict.values())
            
    def X_net(self, indexer=False):
        """Return net reaction conversion of reactants as a dictionary or
        a ChemicalIndexer if indexer is True."""
        X_net = {}
        for i, j in zip(self.reactants, self.X):
            if i in X_net:
                X_net[i] += j
            else:
                X_net[i] = j
        if indexer:
            chemicals = self.chemicals
            phases = self.phases
            if phases:
                phases = [i[0] for i in X_net]
                mi = MaterialIndexer(phases=phases, chemicals=chemicals)
                for i,j in X_net.items(): mi[i] = j
                return mi                
            else:
                data = chemicals.kwarray(X_net)
                return ChemicalIndexer.from_data(data, NoPhase, chemicals, False)
        else:
            return X_net


class SeriesReaction(ReactionSet):
    """
    Create a ParallelReaction object from Reaction objects. When called, 
    it returns the change in material due to all reactions in series.
    
    Parameters
    ----------
    reactions : Iterable[Reaction]
    
    
    """
    __slots__ = ()

    def reduce(self):
        raise TypeError('cannot reduce a SeriesReation object, only '
                        'ParallelReaction objects are reducible')

    def _reaction(self, material_array):
        for i, j, k in zip(self._X_index, self.X, self._stoichiometry):
            material_array += material_array[i] * j * k

    def X_net(self, indexer=False):
        """Return net reaction conversion of reactants as a dictionary or
        a ChemicalIndexer if indexer is True."""
        X_net = {}
        for i, j in zip(self.reactants, self.X):
            if i in X_net:
                X_net[i] += (1 - X_net[i]) * j
            else:
                X_net[i] = j
        if indexer:
            chemicals = self.chemicals
            data = chemicals.kwarray(X_net)
            return ChemicalIndexer.from_data(data, NoPhase, chemicals, False)
        else:
            return X_net

class ReactionSystem:
    """
    Create a ReactionSystem object that can react a stream across a series of
    reactions.
    
    Parameters
    ----------
    *reactions : Reaction, ParallelReaction, or SeriesReaction
        All reactions within the reaction system.
    
    Examples
    --------
    Create a reaction system for cellulosic fermentation of biomass:
    
    >>> from thermosteam import Rxn, RxnSys, PRxn, SRxn, settings, Chemical, Stream
    >>> cal2joule = 4.184
    >>> Glucan = Chemical('Glucan', search_db=False, formula='C6H10O5', Hf=-233200*cal2joule, phase='s', default=True)
    >>> Glucose = Chemical('Glucose', phase='s')
    >>> CO2 = Chemical('CO2', phase='g')
    >>> HMF = Chemical('HMF', search_ID='Hydroxymethylfurfural', phase='l', default=True)
    >>> Biomass = Glucose.copy(ID='Biomass')
    >>> settings.set_thermo(['Water', 'Ethanol', 'LacticAcid', HMF, Glucose, Glucan, CO2, Biomass])
    
    >>> saccharification = PRxn([
    ...     Rxn('Glucan + H2O -> Glucose', reactant='Glucan', X=0.9),
    ...     Rxn('Glucan -> HMF + 2H2O', reactant='Glucan', X=0.025)
    ... ])
    >>> fermentation = SRxn([
    ...     Rxn('Glucose -> 2LacticAcid', reactant='Glucose', X=0.03),
    ...     Rxn('Glucose -> 2Ethanol + 2CO2', reactant='Glucose', X=0.95),
    ... ])
    >>> cell_growth = Rxn('Glucose -> Biomass', reactant='Glucose', X=1.0)
    >>> cellulosic_rxnsys = RxnSys(saccharification, fermentation, cell_growth)
    >>> cellulosic_rxnsys.show()
    ReactionSystem:
    index  reaction
    [0]    ParallelReaction (by mol):
           subindex  stoichiometry              reactant    X[%]
           [0]       Water + Glucan -> Glucose  Glucan     90.00
           [1]       Glucan -> 2 Water + HMF    Glucan      2.50
    [1]    SeriesReaction (by mol):
           subindex  stoichiometry                 reactant    X[%]
           [0]       Glucose -> 2 LacticAcid       Glucose     3.00
           [1]       Glucose -> 2 Ethanol + 2 CO2  Glucose    95.00
    [2]    Reaction (by mol):
           stoichiometry       reactant    X[%]
           Glucose -> Biomass  Glucose   100.00
    
    Compute the flux of glucan through saccharification reactions:
    
    >>> feed = Stream('feed', Glucan=1.0, Water=5.0)
    >>> cellulosic_rxnsys.reactant_flux(feed, index=0)
    0.925
    
    Compute the flux of glucan through glucan to glucose saccharification:
    
    >>> cellulosic_rxnsys.reactant_flux(feed, index=0, subindex=0)
    0.9
    
    Compute the flux of glucose through cell growth:
        
    >>> cellulosic_rxnsys.reactant_flux(feed, index=2)
    0.04365
    
    Compute the flux of glucose through ethanol production:
    
    >>> cellulosic_rxnsys.reactant_flux(feed, index=1, subindex=1)
    0.8293
    
    Notice how reacting the stream leads to ethanol production equivalent
    of 2x the glucose flux to that reaction:
    
    >>> cellulosic_rxnsys(feed)
    >>> feed.show()
    Stream: feed
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): Water       4.15
                     Ethanol     1.66
                     LacticAcid  0.054
                     HMF         0.025
                     Glucan      0.075
                     CO2         1.66
                     Biomass     0.0437
    
    """
    __slots__ = ('_reactions',
                 '_basis',
                 '_chemicals',
                 '_phases')
    
    def __init__(self, *reactions, basis=None):
        if not reactions: raise ValueError('Reactions cannot be empty')
        self._reactions = reactions
        try: self._phases, = set([i._phases for i in reactions])
        except: raise ValueError('all reactions must have the same phases')
        try: self._chemicals, = set([i._chemicals for i in reactions])
        except: raise ValueError('all reactions must have the same chemicals')
        try: self._basis, = set([i._basis for i in reactions])
        except: raise ValueError('all reactions must have the same basis')
        
    force_reaction = Reaction.force_reaction
    adiabatic_reaction = Reaction.adiabatic_reaction
    __call__ = Reaction.__call__
    show = Reaction.show
    
    @property
    def X(self):
        return [i.X for i in self._reactions]
    @X.setter
    def X(self, X):
        for i, j in zip(self._reactions, X): i.X = j
    
    @property
    def reactions(self):
        return self._reactions
    
    def _reaction(self, material):
        basis = self._basis
        for i in self._reactions: 
            if i._basis != basis: raise RuntimeError('not all reactions have the same basis')
            i._reaction(material)
        
    def X_net(self, indexer=False):
        """Return net reaction conversion of reactants as a dictionary or
        a ChemicalIndexer if indexer is True."""
        X_net = {}
        for rxn in self._reactions:
            for i, j in rxn.X_net().items():
                if i in X_net:
                    X_net[i] += (1 - X_net[i]) * j
                else:
                    X_net[i] = j
        if indexer:
            chemicals = self._chemicals
            data = chemicals.kwarray(X_net)
            return ChemicalIndexer.from_data(data, NoPhase, chemicals, False)
        else:
            return X_net
        
    def reactant_flux(self, material, index, subindex=None):
        """
        Return the amount of reactant being reacted in a reaction.
        
        Parameters
        ----------
        material : Stream or array
            Material entering reaction system.
        index : int
            Index of reaction to calculate reactant flux.
        subindex : int, optional
            Subindex of reaction to calculate reactant flux.    
        
        """
        material_array, config = as_material_array(
            material.copy(), self._basis, self._phases, self._chemicals
        )
        isproperty = isinstance(material_array, property_array)
        preconverted_material = material_array.value if isproperty else material_array
        reactions = self.reactions
        for i, rxn in enumerate(reactions):
            if i == index: break
            rxn(preconverted_material)
        reaction = reactions[index]
        if subindex is not None:
            if isinstance(reaction, SeriesReaction):
                reactions = reaction
                for i, rxn in enumerate(reactions):
                    if i == subindex: break
                    rxn(preconverted_material)
            reaction = reaction[subindex]
            X_index = reaction._X_index
            return reaction.X * preconverted_material[X_index]
        elif isinstance(reaction, SeriesReaction):
            raise ValueError('must pass subindex if the index refers to a SeriesReaction object')
        elif isinstance(reaction, ParallelReaction):
            X_index = reaction._X_index
            return (reaction.X * preconverted_material[X_index]).sum()
        else:
            X_index = reaction._X_index
            return reaction.X * preconverted_material[X_index]
        
        
    def __repr__(self):
        return f"{type(self).__name__}({', '.join([repr(i) for i in self.reactions])})"
    
    def _info(self):
        indexed_info = f"{type(self).__name__}:\n"
        isa = isinstance
        infos = [(i._info('subindex') if isa(i, ReactionSet) else i._info()) 
                 for i in self._reactions]
        N = len(infos)
        index_name = 'index'
        maxnumspace = max(len(str(N)) + 1, len(index_name))
        header = f"{index_name}" + " "*(max(2, maxnumspace-3))
        indexed_info += header + "reaction"
        info_dim = '\n' + len(header) * " "
        for index, info in enumerate(infos):
            num = str(index) 
            numspace = (maxnumspace - len(num)) * " "
            info = info.replace('\n', info_dim)
            indexed_info += f"\n[{num}]{numspace}{info}"
        return indexed_info    
        
    _ipython_display_ = show = Reaction.show

# Short-hand conventions
Rxn = Reaction
RxnI = ReactionItem
RxnS = ReactionSet
PRxn = ParallelReaction
SRxn = SeriesReaction
RxnSys = ReactionSystem