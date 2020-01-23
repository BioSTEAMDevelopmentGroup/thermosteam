# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:41:02 2019

@author: yoelr
"""
from .utils import read_only
from .exceptions import UndefinedChemical
from ._chemical import Chemical, _thermo
from .indexer import ChemicalIndexer
import numpy as np

__all__ = ('Chemicals', 'CompiledChemicals')
setattr = object.__setattr__

key_thermo_props = ('V', 'S', 'H', 'Cn')

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
        try:
            return [dct[i] for i in IDs]
        except KeyError:
            for ID in IDs:
                if ID not in dct: raise UndefinedChemical(ID)
    
    def append(self, chemical):
        """Append a Chemical."""
        assert isinstance(chemical, Chemical), ("only 'Chemical' objects are allowed, "
                                               f"not '{type(chemical).__name__}'")
        setattr(self, chemical.ID, chemical)
    
    def extend(self, chemicals):
        """Extend with more Chemical objects."""
        if isinstance(chemicals, Chemicals):
            self.__dict__.update(chemicals.__dict__)
        else:
            for chemical in chemicals:
                assert isinstance(chemical, Chemical), ("only 'Chemical' objects are allowed, "
                                                        f"not '{type(chemical).__name__}'")
                setattr(self, chemical.ID, chemical)
    
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
        for i in chemicals:
            assert not i.get_missing_slots(key_thermo_props), f"{i} is missing key thermodynamic properties"
        IDs = tup([i.ID for i in chemicals])
        CAS = tup([i.CAS for i in chemicals])
        N = len(IDs)
        index = tup(range(N))
        for i in chemicals: dct[i.CAS] = i
        dct['tuple'] = chemicals
        dct['size'] = N
        dct['IDs'] = IDs
        dct['CASs'] = tup([i.CAS for i in chemicals])
        dct['MW'] = np.array([i.MW for i in chemicals])
        dct['Hf'] = np.array([i.Hf for i in chemicals])
        dct['Hc'] = np.array([i.Hc for i in chemicals])
        dct['_index'] = index = dict((*zip(CAS, index),
                                      *zip(IDs, index)))
        dct['_index_cache'] = {}
        dct['equilibrium_chemicals'] = equilibrium_chemicals = []
        dct['heavy_chemicals'] = heavy_chemicals = []
        dct['light_chemicals'] = light_chemicals = []
        for i in chemicals:
            locked_phase = i.locked_state.phase
            if locked_phase:
                if locked_phase in ('s', 'l'):
                    heavy_chemicals.append(i)
                elif locked_phase == 'g':
                    light_chemicals.append(i)
                else:
                    raise Exception('chemical locked state has an invalid phase')
            else:
                equilibrium_chemicals.append(i)
        dct['_equilibrium_indices'] = eq_index = [index[i.ID] for i in equilibrium_chemicals]
        dct['_has_equilibrium'] = has_equilibrium = np.zeros(N, dtype=bool)
        dct['_heavy_indices'] = [index[i.ID] for i in heavy_chemicals]
        dct['_light_indices'] = [index[i.ID] for i in light_chemicals]
        has_equilibrium[eq_index] = True
        
    
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
    
    def kwarray(self, ID_data):
        """Return an array with entries that correspond to the orded chemical IDs.
        
        Parameters
        ----------
        ID_data : dict
                 ID-data pairs.
            
        Examples
        --------
        >>> from ether import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.kwarray(dict(Water=2))
        array([2., 0.])
        
        """
        return self.array(*zip(*ID_data.items()))
    
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
        array[self.get_index(tuple(IDs))] = data
        return array

    def iarray(self, IDs, data):
        array = self.array(IDs, data)
        return ChemicalIndexer.from_data(array, phase=None, chemicals=self)

    def ikwarray(self, ID_data):
        array = self.kwarray(ID_data)
        return ChemicalIndexer.from_data(array, phase=None, chemicals=self)

    def index(self, ID):
        """Return index of specified chemical.

        Parameters
        ----------
        ID: str
            Chemicl ID

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
              Chemical IDs or CAS numbers.

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
    
    def get_index(self, key):
        cache = self._index_cache
        try: 
            index = cache[key]
        except KeyError: 
            cache[key] = index = self._get_index(key)
        except TypeError:
            key = tuple(key)
            cache[key] = index = self._get_index(key)
        return index
    
    def _get_index(self, IDs):
        if isinstance(IDs, str):
            return self.index(IDs)
        elif isinstance(IDs, tuple):
            return self.indices(IDs)
        elif IDs is ...:
            return IDs
        else:
            raise IndexError(f"only strings, tuples, and ellipsis are valid IDs")
    
    def __len__(self):
        return self.size
    
    def __contains__(self, chemical):
        return chemical in self.tuple
    
    def __iter__(self):
        return iter(self.tuple)
        
    def get_equilibrium_indices(self, nonzero):
        """Return indices of species in equilibrium."""
        return np.where(self._has_equilibrium & nonzero)[0]
    
    def __str__(self):
        return f"[{', '.join(self.IDs)}]"