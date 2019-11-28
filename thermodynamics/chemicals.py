# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:41:02 2019

@author: yoelr
"""
from .exceptions import UndefinedChemical
from .chemical import Chemical
import numpy as np

__all__ = ('Chemicals', 'CompiledChemicals')

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
    
    def __init__(self, chemicals):
        setfield = setattr
        isa = isinstance
        for chem in chemicals:
            if isa(chem, Chemical):
                setfield(self, chem.ID, chem)
            else:
                setfield(self, chem, Chemical(chem))
    
    @property
    def IDs(self):
        return tuple(self.__dict__)
    
    def compile(self):
        self.__class__ = CompiledChemicals
        self._compile()
    
    def subgroup(self, IDs):
        return self.__class__([getattr(self, i) for i in IDs])
        
    def extend(self, chemicals):
        """Extend with more Chemicals."""
        for c in chemicals: setattr(self, c.ID, c)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, chemical):
        return chemical in self.__dict__.values()
    
    def __iter__(self):
        yield from self.__dict__.values()
    
    def __str__(self):
        IDs = [i for i in self.IDs]
        return f"[{', '.join(IDs)}]"
    
    def __repr__(self):
        return f'{type(self).__name__}({self})'
    
    
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
    
    def __init__(self, chemicals):
        super().__init__(chemicals)
        self._compile()
    
    def extend(self, chemicals):
        raise TypeError("'{type(self).__name__}' object is read-only")
    
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
        dct['_N'] = N
        dct['_index'] = dict((*zip(CAS, index), *zip(IDs, index)))
        dct['_isheavy'] = np.array([i.Tb in (np.inf, None) for i in chemicals])
        dct['_islight'] = np.array([i.Tb in (0, -np.inf) for i in chemicals], dtype=bool)
        nonfinite = (np.inf, -np.inf, None)
        dct['_has_equilibrium'] = np.array([(bool(i.Dortmund)
                                              and i.Tb not in nonfinite)
                                             for i in chemicals])
        dct['_chemicals'] = chemicals
        dct['_IDs'] = IDs
        dct['_MW'] = np.array([i.MW for i in chemicals])
    
    @property
    def IDs(self):
        """[tuple] IDs of Species object."""
        return self._IDs
    
    def subgroup(self, IDs):
        new = object.__new__(self.__class__)
        setfield = object.setattr
        getfield = getattr
        for i in IDs: setfield(new, i, getfield(self, i))
        for i in self.IDs:
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
    
    def kwarray(self, **data):
        """Return an array with entries that correspond to the orded chemical IDs.
        
        Parameters
        ----------
            **data : 
                     ID-value pair.
            
        Examples
        --------
        >>> from thermotree import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.kwarray(Water=2)
        array([2., 0.])
        
        """
        return self.array(*zip(*data.items()))
    
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
        >>> from thermotree import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.array(['Water'], [2])
        array([2., 0.])
        
        """
        array = np.zeros(self._N)
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
        
        >>> from thermotree import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.index('Water')
        1

        Indices by CAS number:
        
        >>> chemicals.index('7732-18-5'):
        1

        """
        try: return self._index[ID]
        except KeyError:
            if ID not in self._indexdct: raise UndefinedChemical(ID)

    def indices(self, IDs):
        """Return indices of specified chemicals.

        Parameters
        ----------
        IDs : iterable
              Species IDs or CAS numbers.

        Examples
        --------
        Indices by ID:
        
        >>> from thermotree import CompiledChemicals
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
        except KeyError:
            for i in IDs:
                if i not in dct: raise UndefinedChemical(i)     
    
    def __delattr__(self, ID):
        raise AttributeError(f"'{type(self).__name__} object is read-only")
    
    def __setattr__(self, ID, name):
        raise AttributeError(f"'{type(self).__name__} object is read-only")
    
    def __len__(self):
        return self._N
    
    def __contains__(self, chemical):
        return chemical in self._chemicals
    
    def __iter__(self):
        yield from self._chemicals
        
    def _equilibrium_indices(self, nonzero):
        """Return indices of species in equilibrium."""
        return np.where(self._has_equilibrium & nonzero)

    def _heavy_indices(self, nonzero):
        """Return indices of heavy species not in equilibrium."""
        return np.where(self._isheavy & nonzero)

    def _light_indices(self, nonzero):
        """Return indices of light species not in equilibrium."""
        return np.where(self._islight & nonzero)
    