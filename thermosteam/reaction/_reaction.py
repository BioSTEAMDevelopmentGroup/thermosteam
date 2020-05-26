# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:23:52 2019

@author: yoelr
"""
import thermosteam as tmo
from . import _parse as prs
from ..utils import chemicals_user
from .._phase import NoPhase
from ..indexer import ChemicalIndexer
from ..exceptions import InfeasibleRegion
import numpy as np

__all__ = ('Reaction', 'ParallelReaction', 'SeriesReaction')

def check_material_feasibility(material: np.ndarray):
    if (material < 0.).any(): raise InfeasibleRegion('not enough reactants; reaction conversion')

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

def as_material_array(material, basis, chemicals):
    isa = isinstance
    if isa(material, np.ndarray):
        return material
    elif isa(material, tmo.MultiStream):
        raise ValueError('reacting a stream with multiple phases is underdetermined')
    elif isa(material, tmo.Stream):
        assert material.chemicals is chemicals, "reaction and stream chemicals do not match"
        if basis == 'mol':
            return material.mol
        elif basis == 'wt':
            return material.mass
        else:
            raise ValueError("basis must be either 'mol' or 'wt'")
    else:
        raise ValueError('reaction material must be either an array or a stream')

@chemicals_user
class Reaction:
    """
    Create a Reaction object which defines a stoichiometric reaction and
    conversion. When called, it elements of a material array due to the
    reaction.
    
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
    check_material_balance=False: bool
        Whether to assert that mass is not created or destroyed.
    check_atomic_balance=False: bool
        Whether to assert that stoichiometric balance by atoms cancel out.
    correct_atomic_balance=False: bool
        Whether to correct the stoichiometry according to the atomic balance.
    
    Notes
    -----
    A reaction object can react either a stream or an array. When a stream
    is passed, it reacts either the mol or mass flow rate according to
    the basis of the reaction object. When an array is passed, the array
    elements are reacted regardless of what basis they are associated with.
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> import thermosteam.reaction as rxn
    >>> chemicals = tmo.Chemicals(['H2O', 'H2', 'O2'])
    >>> tmo.settings.set_thermo(chemicals)
    >>> reaction = rxn.Reaction('2H2O -> 2H2 + O2', reactant='H2O', X=0.7)
    >>> reaction.show() # Note that the default basis is by 'mol'
    Reaction (by mol):
     stoichiometry       reactant    X[%]
     H2O -> H2 + 0.5 O2  H2O        70.00
    >>> feed = tmo.Stream('feed', H2O=200)
    >>> reaction(feed) # Call to run reaction on molar flow
    >>> feed.show() # Notice how 70% of water was converted to product
    Stream: feed
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): H2O  60
                     H2   140
                     O2   70
    
    Alternatively, its also possible to react an array (instead of a stream):
    
    >>> import numpy as np
    >>> array = np.array([200., 0. , 0.])
    >>> reaction(array)
    >>> array
    array([ 60., 140.,  70.])
    
    Let's change to a per 'wt' basis:
        
    >>> reaction.basis = 'wt'
    >>> reaction.show()
    Reaction (by wt):
     stoichiometry               reactant    X[%]
     H2O -> 0.112 H2 + 0.888 O2  H2O        70.00
    
    Although we changed the basis, the end result is the same if we pass a 
    stream:
    
    >>> feed = tmo.Stream('feed', H2O=200)
    >>> reaction(feed) # Call to run reaction on mass flow
    >>> feed.show() # Notice how 70% of water was converted to product
    Stream: feed
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): H2O  60
                     H2   140
                     O2   70
    
    If we pass an array, however, the Reaction object assumes the
    array data is by weight:
        
    >>> array = np.array([200., 0. , 0.])
    >>> reaction(array)
    >>> array
    array([ 60.   ,  15.666, 124.334])
    
    """
    __slots__ = ('_basis',
                 '_chemicals',
                 '_X_index', 
                 '_stoichiometry', 
                 '_X')
    
    def __init__(self, reaction, reactant, X, 
                 chemicals=None, basis='mol', *,
                 check_material_balance=False,
                 check_atomic_balance=False,
                 correct_atomic_balance=False):
        if basis in ('wt', 'mol'):
            self._basis = basis
        else:
            raise ValueError("basis must be either by 'wt' or by 'mol'")
        self._X = X #: [float] Reactant conversion
        chemicals = self._load_chemicals(chemicals)
        if reaction:            
            self._stoichiometry = prs.get_stoichiometric_array(reaction, chemicals)
            self._X_index = self._chemicals.index(reactant)
            self._rescale()
            if correct_atomic_balance:
                self.correct_atomic_balance()
            else:
                if check_atomic_balance:
                    self.check_atomic_balance()
                elif check_material_balance:
                    self.check_material_balance()
        else:
            self._stoichiometry = np.zeros(chemicals.size)
            self._X_index = self._chemicals.index(reactant)
    
    def copy(self, basis=None):
        """Return copy of Reaction object."""
        copy = self.__new__(self.__class__)
        copy._basis = self._basis
        copy._stoichiometry = self._stoichiometry.copy()
        copy._X_index = self._X_index
        copy._chemicals = self._chemicals
        copy._X = self._X
        if basis: set_reaction_basis(copy, basis)
        return copy
    
    def has_reaction(self):
        return bool(self.X and self.stoichiometry.any())
    
    def __radd__(self, rxn):
        return self + rxn
    
    def __add__(self, rxn):
        if rxn == 0 or not rxn.has_reaction(): return self.copy()
        assert self._basis == rxn._basis, 'basis must be the same to add reactions'
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to add reactions'
        assert self._X_index == rxn._X_index, 'reactants must be the same to add reactions'
        new = self.copy()
        stoichiometry = self._stoichiometry*self.X + rxn._stoichiometry*rxn.X
        new._stoichiometry = stoichiometry/-(stoichiometry[new._X_index])
        new.X = self.X + rxn.X
        return new
    
    def __iadd__(self, rxn):
        if rxn == 0 or not rxn.has_reaction(): return self
        basis = self.basis
        if basis != rxn._basis: rxn = rxn.copy(basis)
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to add reactions'
        assert self._X_index == rxn._X_index, 'reactants must be the same to add reactions'
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
    
    def __div__(self, num):
        self.__mul__(self, 1/num)
    
    def __rdiv__(self, num):
        self.__mul__(self, 1/num)    
    
    def __idiv__(self, num):
        return self.__imul__(self, 1/num) 
    
    def __neg__(self):
        new = self.copy()
        new.X *= -1.
        return new
    
    def __sub__(self, rxn):
        assert self._basis == rxn._basis, 'basis must be the same to substract reactions'
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to substract reactions'
        assert self._X_index == rxn._X_index, 'reactants must be the same to substract reactions'
        new = self.copy()
        stoichiometry = self._stoichiometry*self.X - rxn._stoichiometry*rxn.X
        new._stoichiometry = stoichiometry/-(stoichiometry[new._X_index])
        new.X = self.X - rxn.X
        return new
    
    def __isub__(self, rxn):
        basis = self.basis
        if basis != rxn._basis: rxn = rxn.to_basis(basis)
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to substract reactions'
        assert self._X_index == rxn._X_index, 'reactants must be the same to substract reactions'
        stoichiometry = self._stoichiometry*self.X + rxn._stoichiometry*rxn.X
        self._stoichiometry = stoichiometry/-(stoichiometry[self._X_index])
        self.X = self.X - rxn.X
        return 
    
    def __call__(self, material):
        material_array = as_material_array(material,
                                           self._basis,
                                           self._chemicals)
        self._reaction(material_array)
        if tmo.reaction.CHECK_FEASIBILITY:
            check_material_feasibility(material_array)
        
    def force_reaction(self, material):
        """React material ignoring feasibility checks."""
        material_array = as_material_array(material,
                                           self._basis,
                                           self._chemicals)
        self._reaction(material_array)

    def adiabatic_reaction(self, material):
        """Adjust product enthalpy based on calculated H_rxn for adiabatic reactions."""
        if isinstance(material, np.ndarray):
            raise ValueError("Please use adiabatic_reaction(stream), not adiabatic_reaction(stream.mol) \
                or adiabatic_reaction(stream.wt)")
        Hnet_0 = material.Hf + material.H
        material_array = as_material_array(material,
                                           self._basis,
                                           self._chemicals)
        self._reaction(material_array)
        material.H = Hnet_0 - material.Hf
    
    def _reaction(self, material_array):
        material_array += material_array[self._X_index] * self.X * self._stoichiometry
    
    @property
    def dH(self):
        """
        Heat of reaction at given conversion. Units are in either
        J/mol-reactant or J/g-reactant; depending on basis.
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
        self._X = X
    
    @property
    def stoichiometry(self):
        """[array] Stoichiometry coefficients."""
        return self._stoichiometry
    @property
    def istoichiometry(self):
        """[ChemicalIndexer] Stoichiometry coefficients."""
        return tmo.indexer.ChemicalIndexer.from_data(self._stoichiometry,
                                                     chemicals=self._chemicals,
                                                     check_data=False)
    @property
    def reactant(self):
        """[str] Reactant associated to conversion."""
        return self._chemicals.IDs[self._X_index]

    @property
    def MWs(self):
        """[1d array] Molecular weights of all chemicals [mol/g]."""
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
    
    def check_material_balance(self, tol=1e-3):
        """Assert that stoichiometric mass balance is correct."""
        stoichiometry_by_wt = self._get_stoichiometry_by_wt()
        error = abs(stoichiometry_by_wt.sum())
        assert error <= tol, (
            f"material stoichiometry is unbalanced by {error} in weight"
        )
    
    def check_atomic_balance(self, tol=1e-3):
        """Assert that stoichiometric mass balance is correct."""
        stoichiometry_by_mol = self._get_stoichiometry_by_mol()
        formula_array = self.chemicals.formula_array
        unbalanced_array = formula_array @ stoichiometry_by_mol
        atoms = tmo.properties.elements.array_to_atoms(unbalanced_array)
        assert abs(sum(atoms.values())) < tol, (
            f"atomic stoichiometry is unbalanced by the following molar stoichiometric coefficients:\n "
            + "\n ".join([f"{symbol}: {value}" for symbol, value in atoms.items()])
        )
    
    def correct_atomic_balance(self, constants=None):
        """Correct stoichiometry coffecients to satisfy atomic balance."""
        stoichiometry_by_mol = self._get_stoichiometry_by_mol()
        chemicals = self.chemicals
        if constants:
            constant_index = chemicals.indices(constants)
        else:
            constant_index = [self._X_index]
        chemical_index, = np.where(stoichiometry_by_mol != 0.)
        chemical_index = np.setdiff1d(chemical_index, constant_index)
        formula_array = chemicals.formula_array
        b = - (formula_array[:, constant_index]
               * stoichiometry_by_mol[constant_index]).sum(1, keepdims=True)
        atomic_index, _ = np.where(b != 0)
        b = b[atomic_index, :]
        A = formula_array[atomic_index, :][:, chemical_index]
        M_atoms, N_chemicals = A.shape
        if M_atoms != N_chemicals:
            raise RuntimeError(
                 'to solve atomic balance, number of atoms '
                f'({M_atoms} available) must be equal to the number of '
                 'chemicals, not including the reactant '
                f'({N_chemicals} available)'
            )
        x = np.linalg.solve(A, b)
        stoichiometry_by_mol[chemical_index] = x.flatten()
        if self._basis == 'wt': 
            self._stoichiometry[:] = stoichiometry_by_mol * self.MWs
        self._rescale()
    
    def _rescale(self):
        """Scale stoichiometry to a per reactant basis."""
        new_scale = -self._stoichiometry[self._X_index]
        self._stoichiometry /= new_scale
    
    def __repr__(self):
        stoichiometry = prs.get_stoichiometric_string(self._stoichiometry, self._chemicals)
        return f"{type(self).__name__}('{stoichiometry}', reactant='{self.reactant}', X={self.X:.3g}, basis={self.basis})"
    
    def show(self):
        info = f"{type(self).__name__} (by {self.basis}):"
        rxn = prs.get_stoichiometric_string(self._stoichiometry, self._chemicals)
        cmp = self.reactant
        lrxn = len(rxn)
        lcmp = len(cmp)
        maxrxnlen = max([13, lrxn]) + 2
        maxcmplen = max([8, lcmp]) + 2
        X = self.X
        info += "\n stoichiometry" + " "*(maxrxnlen - 13) + "reactant" + " "*(maxcmplen - 8) + '  X[%]'
        rxn_spaces = " "*(maxrxnlen - lrxn)
        cmp_spaces = " "*(maxcmplen - lcmp)
        info += f"\n {rxn}{rxn_spaces}{cmp}{cmp_spaces}{X*100: >6.2f}"
        print(info)
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
    def __init__(self, rxnset, index):
        self._stoichiometry = rxnset._stoichiometry[index]
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
    __slots__ = ('_basis',
                 '_stoichiometry', 
                 '_X', '_X_index', 
                 '_chemicals')

    copy = Reaction.copy
    _get_stoichiometry_by_mol = Reaction._get_stoichiometry_by_mol
    _get_stoichiometry_by_wt = Reaction._get_stoichiometry_by_wt
    
    def __init__(self, reactions):
        assert reactions, 'no reactions passed'
        chemicals = {i.chemicals for i in reactions}
        try: self._chemicals, = chemicals
        except: raise ValueError('all reactions must have the same chemicals')
        basis = {i.basis for i in reactions}
        try: self._basis, = basis
        except: raise ValueError('all reactions must have the same basis')
        self._stoichiometry = np.array([i._stoichiometry for i in reactions])
        self._X = np.array([i.X for i in reactions])
        self._X_index = np.array([i._X_index for i in reactions])
    
    def __getitem__(self, index):
        stoichiometry = self._stoichiometry[index]
        if len(stoichiometry.shape) == 1:
            return ReactionItem(self, index)
        else:
            rxnset = self.__new__(self.__class__)
            rxnset._basis = self._basis
            rxnset._stoichiometry = stoichiometry
            rxnset._X = self._X[index]
            rxnset._X_index = self._X_index[index]
            rxnset._chemicals = self._chemicals
            return rxnset
    
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
        return tuple([IDs[i] for i in self._X_index])
    
    @property
    def MWs(self):
        """[2d array] Molecular weights of all chemicals."""
        return self._chemicals.MW[np.newaxis, :]
    
    def _rescale(self):
        """Scale stoichiometry to a per reactant basis."""
        X_index = self._X_index
        new_scale = -self._stoichiometry[np.arange(X_index.size), X_index, np.newaxis]
        self._stoichiometry /= new_scale
    
    def __repr__(self):
        return f"{type(self).__name__}([', '.join([repr(i) for i in self])])"
    
    def show(self):
        info = f"{type(self).__name__} (by {self.basis}):"
        chemicals = self._chemicals
        rxns = [prs.get_stoichiometric_string(i, chemicals) for i in self._stoichiometry]
        maxrxnlen = max([13, *[len(i) for i in rxns]]) + 2
        cmps = self.reactants
        maxcmplen = max([8, *[len(i) for i in cmps]]) + 2
        Xs = self.X
        N = len(Xs)
        maxnumspace = max(len(str(N)) + 1, 5)
        info += f"\nindex" + " "*(max(2, maxnumspace-3)) + "stoichiometry" + " "*(maxrxnlen - 13) + "reactant" + " "*(maxcmplen - 8) + '  X[%]'
        for N, rxn, cmp, X in zip(range(N), rxns, cmps, Xs):
            rxn_spaces = " "*(maxrxnlen - len(rxn))
            cmp_spaces = " "*(maxcmplen - len(cmp))
            num = str(N)
            numspace = (maxnumspace - len(num)) * " "
            info += f"\n[{N}]{numspace}{rxn}{rxn_spaces}{cmp}{cmp_spaces}{X*100: >6.2f}"
        print(info)
    _ipython_display_ = show
        

class ParallelReaction(ReactionSet):
    """
    Create a ParallelReaction object from Reaction objects. When called, 
    it returns the change in material due to all parallel reactions.
    
    Parameters
    ----------
    reactions : Iterable[Reaction]
    
    
    """
    __slots__ = ()
    
    def __call__(self, material):
        material_array = as_material_array(material,
                                           self._basis, 
                                           self._chemicals)
        self._reaction(material_array)
        if tmo.reaction.CHECK_FEASIBILITY: 
            check_material_feasibility(material_array)
        
    def force_reaction(self, material):
        """React material ignoring feasibility checks."""
        material_array = as_material_array(material, self._basis, self._chemicals)
        self._reaction(material_array)

    def adiabatic_reaction(self, material):
        """Adjust product enthalpy based on calculated H_rxn for adiabatic reactions."""
        if isinstance(material, np.ndarray):
            raise ValueError("Please use adiabatic_reaction(stream), not adiabatic_reaction(stream.mol) \
                or adiabatic_reaction(stream.wt)")
        Hnet_0 = material.Hf + material.H
        material_array = as_material_array(material,
                                           self._basis,
                                           self._chemicals)
        self._reaction(material_array)
        material.H = Hnet_0 - material.Hf

    def _reaction(self, material_array):
        material_array += material_array[self._X_index] * self.X @ self._stoichiometry

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
            
    @property
    def X_net(self):
        """[ChemicalIndexer] Net reaction conversion of reactants."""
        X_net = {}
        for i, j in zip(self.reactants, self.X):
            if i in X_net:
                X_net[i] += j
            else:
                X_net[i] = j
        chemicals = self.chemicals
        data = chemicals.kwarray(X_net)
        return ChemicalIndexer.from_data(data, NoPhase, chemicals, False)


class SeriesReaction(ReactionSet):
    """
    Create a ParallelReaction object from Reaction objects. When called, 
    it returns the change in material due to all reactions in series.
    
    Parameters
    ----------
    reactions : Iterable[Reaction]
    
    
    """
    __slots__ = ()
    
    def __call__(self, material):
        material_array = as_material_array(material,
                                           self._basis, 
                                           self._chemicals)
        self._reaction(material_array)
        if tmo.reaction.CHECK_FEASIBILITY:
            check_material_feasibility(material_array)

    def force_reaction(self, material):
        """React material ignoring feasibility checks."""
        array = as_material_array(material, self._basis, self._chemicals)
        self._reaction(array)

    def adiabatic_reaction(self, material):
        """Adjust product enthalpy based on calculated H_rxn for adiabatic reactions."""
        if isinstance(material, np.ndarray):
            raise ValueError("Please use adiabatic_reaction(stream), not adiabatic_reaction(stream.mol) \
                or adiabatic_reaction(stream.wt)")
        Hnet_0 = material.Hf + material.H
        material_array = as_material_array(material,
                                           self._basis,
                                           self._chemicals)
        self._reaction(material_array)
        material.H = Hnet_0 - material.Hf

    def reduce(self):
        raise TypeError('cannot reduce a SeriesReation object, only '
                        'ParallelReaction objects are reducible')

    def _reaction(self, material_array):
        for i, j, k in zip(self._X_index, self.X, self._stoichiometry):
            material_array += material_array[i] * j * k

    @property
    def X_net(self):
        """[ChemicalIndexer] Net reaction conversion of reactants."""
        X_net = {}
        for i, j in zip(self.reactants, self.X):
            if i in X_net:
                X_net[i] += (1 - X_net[i]) * j
            else:
                X_net[i] = j
        chemicals = self.chemicals
        data = chemicals.kwarray(X_net)
        return ChemicalIndexer.from_data(data, NoPhase, chemicals, False)

# Short-hand conventions
# Rxn = Reaction
# RxnI = ReactionItem
# RxnS = ReactionSet
# PRxn = ParallelReaction
# SRxn = SeriesReaction
