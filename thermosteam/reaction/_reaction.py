# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:23:52 2019

@author: yoelr
"""
from . import _parse as prs
from ..utils import chemicals_user
import numpy as np

__all__ = ('Reaction', 'ParallelReaction', 'SeriesReaction')

def stoi2str(stoi, chemicals):
    """Parse a stoichiometric array and chemicals to a reaction definition."""
    return f"{prs.arr2str(stoi, chemicals)}"

@chemicals_user
class Reaction:
    """Create a Reaction object which defines a stoichiometric reaction and conversion. When called, it returns the change in material due to the reaction.
    
    Parameters
    ----------
    reaction : str
               Stoichiometric equation written as:
               'i1 R1 + ... + in Rn -> j1 P1 + ... + jm Pm'

    reactant : str
               ID of reactant compound.
        
    X : float
        Reactant conversion (fraction).
        
    chemicals=None : Chemicals, defaults to settings.chemicals.
        Chemicals corresponing to each entry in the stoichiometry array.
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> tmo.settings.thermo = tmo.Thermo(['H2O', 'H2', 'O2'])
    >>> rn = tmo.reaction
    >>> srn = rn.Reaction('2H2O -> 2H2 + O2', reactant='H2O', X=0.7)
    >>> srn
    Reaction('H2O -> H2 + 0.5 O2', reactant='H2O', X=0.7)
    >>> feed = tmo.Stream('feed', H2O=200)
    >>> srn(feed.mol) # Call to run reaction on molar flow
    >>> feed # Notice how 70% of water was converted to product
    Stream: feed
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kmol/hr): H2O  60
                     H2   140
                     O2   70
    
    """
    __slots__ = ('_chemicals', '_Xindex', '_stoi', '_X')
    def __init__(self, reaction, reactant, X, chemicals=None):
        chemicals = self._load_chemicals(chemicals)
        self._stoi = prs.str2arr(reaction, chemicals)
        self._Xindex = self._chemicals.index(reactant)
        self._stoi *= 1/-(self._stoi[self._Xindex])
        self._X = X #: [float] Reactant conversion
    
    def copy(self):
        """Return copy of Reaction object."""
        copy = self.__new__(type(self))
        copy._stoi = self._stoi
        copy._Xindex = self._Xindex
        copy._chemicals = self._chemicals
        copy._X = self._X
        return copy
    
    def __add__(self, rxn):
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to add reactions'
        assert self.reactant is rxn.reactant, 'reactants must be the same to add reactions'
        new = self.copy()
        stoi = self._stoi*self.X + rxn._stoi*rxn.X
        new._stoi = stoi/-(stoi[new._Xindex])
        new.X = (self.X + rxn.X)/2
        return new
    
    def __iadd__(self, rxn):
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to add reactions'
        assert self.reactant is rxn.reactant, 'reactants must be the same to add reactions'
        stoi = self._stoi*self.X + rxn._stoi*rxn.X
        self._stoi = stoi/-(stoi[self._Xindex])
        self.X = (self.X + rxn.X)/2
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
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to add reactions'
        assert self.reactant is rxn.reactant, 'reactants must be the same to add reactions'
        new = self.copy()
        stoi = self._stoi*self.X - rxn._stoi*rxn.X
        new._stoi = stoi/-(stoi[new._Xindex])
        new.X = (self.X - rxn.X)/2
        return new
    
    def __isub__(self, rxn):
        assert self._chemicals is rxn._chemicals, 'working chemicals must be the same to add reactions'
        assert self.reactant is rxn.reactant, 'reactants must be the same to add reactions'
        stoi = self._stoi*self.X + rxn._stoi*rxn.X
        self._stoi = stoi/-(stoi[self._Xindex])
        self.X = (self.X - rxn.X)/2
        return 
    
    def __call__(self, material):
        material += material[self._Xindex]*self.X*self._stoi
    
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
        return self._stoi
    @property
    def reactant(self):
        """[str] Reactant associated to conversion."""
        return self._chemicals.IDs[self._Xindex]
    
    def __repr__(self):
        return f"{type(self).__name__}('{stoi2str(self._stoi, self._chemicals)}', reactant='{self.reactant}', X={self.X:.3g})"
    
    def show(self):
        outs = f"{type(self).__name__}:"
        rxn = stoi2str(self._stoi, self._chemicals)
        cmp = self.reactant
        lrxn = len(rxn)
        lcmp = len(cmp)
        maxrxnlen = max([13, lrxn]) + 2
        maxcmplen = max([8, lcmp]) + 2
        X = self.X
        outs += "\n stoichiometry" + " "*(maxrxnlen - 13) + "reactant" + " "*(maxcmplen - 8) + '  X[%]'
        rxn_spaces = " "*(maxrxnlen - lrxn)
        cmp_spaces = " "*(maxcmplen - lcmp)
        outs += f"\n {rxn}{rxn_spaces}{cmp}{cmp_spaces}{X*100: >6.2f}"
        print(outs)
    _ipython_display_ = show


class ReactionItem(Reaction):
    """Create a ReactionItem object from a ReactionSet and index."""
    __slots__ = ('_index')
    def __init__(self, rxnset, index):
        self._stoi = rxnset._stoi[index]
        self._X = rxnset._X
        self._chemicals = rxnset._chemicals
        self._Xindex = rxnset._Xindex[index]
        self._index = index
    
    def copy(self):
        """Return copy of Reaction object."""
        new = super().copy()
        new._index = self._index
        return new
    
    @property
    def X(self):
        """[float] Reaction converion as a fraction."""
        return self._X[self._index]
    @X.setter
    def X(self, X):
        self._X[self._index] = X
        

class ReactionSet:
    """Create a ReactionSet that contains all reactions and conversions as an array."""
    __slots__ = ('_stoi', '_X', '_Xindex', '_chemicals')
    def __init__(self, reactions):
        assert len({i.chemicals for i in reactions})==1, 'all reactions must have the same chemicals'
        self._stoi = np.array([i._stoi for i in reactions])
        self._X = np.array([i.X for i in reactions])
        self._Xindex = np.array([i._Xindex for i in reactions])
        self._chemicals = reactions[0].chemicals
    
    def __getitem__(self, index):
        stoi = self._stoi[index]
        if len(stoi.shape) == 1:
            return ReactionItem(self, index)
        else:
            rxnset = self.__new__(type(self))
            rxnset._stoi = stoi
            rxnset._X = self._X[index]
            rxnset._Xindex = self._Xindex[index]
            rxnset._chemicals = self._chemicals
            return rxnset
    
    @property
    def X(self):
        """[float] Reaction converions."""
        return self._X
    
    @property
    def chemicals(self):
        """[Chemicals] Chemicals corresponing to each entry in the stoichiometry array."""
        return self._chemicals
    @property
    def stoichiometry(self):
        """[array] Stoichiometry coefficients."""
        return self._stoi
    
    @property
    def reactants(self):
        """[str] Reactants associated to conversion."""
        IDs = self._chemicals.IDs
        return tuple(IDs[i] for i in self._Xindex)
    
    def __repr__(self):
        return f"<{type(self).__name__}: {', '.join(set(self.reactants))}>"
    
    def show(self):
        outs = f"{type(self).__name__}:"
        chemicals = self._chemicals
        rxns = [stoi2str(i, chemicals) for i in self._stoi]
        maxrxnlen = max([13, *[len(i) for i in rxns]]) + 2
        cmps = self.reactants
        maxcmplen = max([8, *[len(i) for i in cmps]]) + 2
        Xs = self.X
        N = len(Xs)
        maxnumspace = max(len(str(N)) + 1, 5)
        outs += f"\nindex" + " "*(max(2, maxnumspace-3)) + "stoichiometry" + " "*(maxrxnlen - 13) + "reactant" + " "*(maxcmplen - 8) + '  X[%]'
        for N, rxn, cmp, X in zip(range(N), rxns, cmps, Xs):
            rxn_spaces = " "*(maxrxnlen - len(rxn))
            cmp_spaces = " "*(maxcmplen - len(cmp))
            num = str(N)
            numspace = (maxnumspace - len(num)) * " "
            outs += f"\n[{N}]{numspace}{rxn}{rxn_spaces}{cmp}{cmp_spaces}{X*100: >6.2f}"
        print(outs)
    _ipython_display_ = show
        
class ParallelReaction(ReactionSet):
    """Create a ParallelReaction object from Reaction objects. When called, it returns the change in material due to all parallel reactions."""
    __slots__ = ()
    
    def __call__(self, material):
        material += material[self._Xindex]*self.X @ self._stoi

    @property
    def X_net(self):
        """[dict] Net reaction conversion of reactants."""
        X_net = {}
        for i, j in zip(self.reactants, self.X):
            if i in X_net:
                X_net[i] += j
            else:
                X_net[i] = j
        return X_net

class SeriesReaction(ReactionSet):
    """Create a ParallelReaction object from Reaction objects. When called, it returns the change in material due to all reactions in series."""
    __slots__ = ()
    
    def __call__(self, material):
        for i, j, k in zip(self._Xindex, self.X, self._stoi):
            material += material[i]*j*k

    @property
    def X_net(self):
        """[dict] Net reaction conversion of reactants."""
        X_net = {}
        for i, j in zip(self.reactants, self.X):
            if i in X_net:
                X_net[i] *= j
            else:
                X_net[i] = j
        return X_net

# Short-hand conventions
# Rxn = Reaction
# RxnI = ReactionItem
# RxnS = ReactionSet
# PRxn = ParallelReaction
# SRxn = SeriesReaction

