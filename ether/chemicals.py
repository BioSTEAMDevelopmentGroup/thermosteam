# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:41:02 2019

@author: yoelr
"""
from .utils import read_only
from .exceptions import UndefinedChemical
from .chemical import Chemical
import numpy as np

__all__ = ('Chemicals', 'CompiledChemicals')
setattr = object.__setattr__

# %% Utilities

def must_compile(*args, **kwargs):
    raise TypeError("method valid only for compiled chemicals; "
                    "run <chemicals>.compile() to compile")

# %% Chemicals

class Chemicals:
    """Create Chemicals object that contains Chemical objects as attributes.

    Parameters
    ----------
    chemicals : Iterable[str or Chemical]
           Strings should be one of the following [-]:
              * Name, in IUPAC form or common form or a synonym registered in PubChem
              * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
              * InChI key, prefixed by 'InChIKey='
              * PubChem CID, prefixed by 'PubChem='
              * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
              * CAS number
        
    """
    _cache = {}
    def __new__(cls, chemicals):
        chemicals = tuple(chemicals)
        cache = cls._cache
        if chemicals in cache:
            return cache[chemicals]
        else:
            self = super().__new__(cls)
            isa = isinstance
            for chem in chemicals:
                if isa(chem, Chemical):
                    setattr(self, chem.ID, chem)
                else:
                    setattr(self, chem, Chemical(chem))
            cache[chemicals] = self
            return self
    
    def __setattr__(self, ID, chemical):
        raise TypeError("can't set attribute; use <Chemicals>.append instead")
    
    def __getnewargs__(self):
        return (tuple(self),)
    
    def compile(self):
        setattr(self, '__class__', CompiledChemicals)
        self._compile()
    
    def subgroup(self, IDs):
        return type(self)([getattr(self, i) for i in IDs])
        
    def retrieve(self, IDs):
        dct = self.__dict__
        return [dct[i] for i in IDs]
    
    def append(self, chemical):
        """Append a Chemical."""
        setattr(self, chemical.ID, chemical)
    
    def extend(self, chemicals):
        """Extend with more Chemical objects."""
        for c in chemicals: setattr(self, c.ID, c)
    
    kwarray = array = index = indices = must_compile
        
    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, chemical):
        return chemical in self.__dict__.values()
    
    def __iter__(self):
        yield from self.__dict__.values()
    
    def __str__(self):
        return f"[{', '.join(self.__dict__)}]"
    
    def __repr__(self):
        return f'{type(self).__name__}({self})'


@read_only(methods=('append', 'extend'))
class CompiledChemicals(Chemicals):
    """Create CompiledChemicals object that contains Chemical objects as attributes.

    Parameters
    ----------
    chemicals : Iterable[str or Chemical]
           Strings should be one of the following [-]:
              * Name, in IUPAC form or common form or a synonym registered in PubChem
              * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
              * InChI key, prefixed by 'InChIKey='
              * PubChem CID, prefixed by 'PubChem='
              * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
              * CAS number
        
    """
    
    def __new__(cls, chemicals):
        self = super().__new__(cls, chemicals)
        self._compile()
        return self
    
    def __dir__(self):
        return self.IDs + tuple(dir(type(self)))
    
    def compile(self): pass
    
    def _compile(self):
        dct = self.__dict__
        tup = tuple
        chemicals = tup(dct.values())
        IDs = tup([i.ID for i in chemicals])
        CAS = tup([i.CAS for i in chemicals])
        N = len(IDs)
        index = tup(range(N))
        for i in chemicals: dct[i.CAS] = i
        dct['tuple'] = chemicals
        dct['size'] = N
        dct['IDs'] = IDs
        dct['MW'] = np.array([i.MW for i in chemicals])
        dct['_index'] = dict((*zip(CAS, index), *zip(IDs, index)))
        dct['_isheavy'] = np.array([i.Tb in (np.inf, None) for i in chemicals])
        dct['_islight'] = np.array([i.Tb in (0, -np.inf) for i in chemicals], dtype=bool)
        nonfinite = (np.inf, -np.inf, None)
        # TODO: Fix equilibrium indices according to property package
        dct['_has_equilibrium'] = np.array([(bool(i.Dortmund)
                                             and i.Tb not in nonfinite)
                                            for i in chemicals])
    
    def subgroup(self, IDs):
        chemicals = self.retrieve(IDs)
        new = Chemicals(chemicals)
        new.compile()
        for i in new.IDs:
            for j in self.get_synonyms(i):
                try: new.set_synonym(i, j)
                except: pass
        return new
    
    def get_synonyms(self, ID):
        k = self._index[ID]
        return [i for i, j in self._index.items() if j==k] 

    def set_synonym(self, ID, synonym):
        chemical = getattr(self, ID)
        dct = self.__dict__
        if synonym in dct and dct[synonym] is not chemical:
            raise ValueError(f"synonym '{synonym}' already in use by {repr(dct[synonym])}")
        else:
            self._index[synonym] = self._index[ID]
            dct[synonym] = chemical
    
    def kwarray(self, IDdata):
        """Return an array with entries that correspond to the orded chemical IDs.
        
        Parameters
        ----------
            IDdata : dict
                     ID-data pairs.
            
        Examples
        --------
        >>> from ether import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.kwarray(dict(Water=2))
        array([2., 0.])
        
        """
        return self.array(*zip(*IDdata.items()))
    
    def array(self, IDs, data):
        """Return an array with entries that correspond to the ordered chemical IDs.
        
        Parameters
        ----------
        IDs : iterable
              Compound IDs.
        data : array_like
               Data corresponding to IDs.
            
        Examples
        --------
        >>> from ether import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.array(['Water'], [2])
        array([2., 0.])
        
        """
        array = np.zeros(len(self))
        array[self.indices(IDs)] = data
        return array

    def index(self, ID):
        """Return index of specified chemical.

        Parameters
        ----------
        ID: str
            Compound ID

        Examples
        --------
        Index by ID:
        
        >>> from ether import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.index('Water')
        1

        Indices by CAS number:
        
        >>> chemicals.index('7732-18-5'):
        1

        """
        try: return self._index[ID]
        except KeyError:
            raise UndefinedChemical(ID)

    def indices(self, IDs):
        """Return indices of specified chemicals.

        Parameters
        ----------
        IDs : iterable
              Species IDs or CAS numbers.

        Examples
        --------
        Indices by ID:
        
        >>> from ether import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.indices(['Water', 'Ethanol'])
        [1, 0]

        Indices by CAS number:
        
        >>> chemicals.indices(['7732-18-5', '64-17-5']):
        [1, 0]

        """
        try:
            dct = self._index
            return [dct[i] for i in IDs]
        except:
            for i in IDs:
                if i not in dct: raise UndefinedChemical(i)     
    
    def __len__(self):
        return self.size
    
    def __contains__(self, chemical):
        return chemical in self.tuple
    
    def __iter__(self):
        return iter(self.tuple)
        
    def _equilibrium_indices(self, nonzero):
        """Return indices of species in equilibrium."""
        return np.where(self._has_equilibrium & nonzero)[0]

    def _heavy_indices(self, nonzero):
        """Return indices of heavy species not in equilibrium."""
        return np.where(self._isheavy & nonzero)[0]

    def _light_indices(self, nonzero):
        """Return indices of light species not in equilibrium."""
        return np.where(self._islight & nonzero)[0]
    
    def __str__(self):
        return f"[{', '.join(self.IDs)}]"