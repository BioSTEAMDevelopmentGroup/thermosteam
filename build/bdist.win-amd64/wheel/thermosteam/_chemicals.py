# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:41:02 2019

@author: yoelr
"""
from .utils import read_only, repr_listed_values
from .exceptions import UndefinedChemical
from ._chemical import Chemical
from .indexer import ChemicalIndexer
import thermosteam as tmo
import numpy as np

__all__ = ('Chemicals', 'CompiledChemicals')
setattr = object.__setattr__

# %% Functions

def must_compile(*args, **kwargs):
    raise TypeError("method valid only for compiled chemicals; "
                    "run <chemicals>.compile() to compile")

def chemical_data_array(chemicals, attr):
    getfield = getattr
    data = np.array([getfield(i, attr) for i in chemicals])
    data.setflags(0)
    return data
    

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
    
    Examples
    --------
    Create a Chemicals object from chemical identifiers:
    
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> chemicals
    Chemicals([Water, Ethanol])
    
    All chemicals are stored as attributes:
        
    >>> chemicals.Water, chemicals.Ethanol
    (Chemical('Water'), Chemical('Ethanol'))
    
        
    """
    def __init__(self, chemicals):
        isa = isinstance
        for chem in chemicals:
            if isa(chem, Chemical):
                setattr(self, chem.ID, chem)
            else:
                setattr(self, chem, Chemical(chem))
    
    def __setattr__(self, ID, chemical):
        raise TypeError("can't set attribute; use <Chemicals>.append instead")
    
    def __getnewargs__(self):
        return (tuple(self),)
    
    def compile(self):
        """
        Cast as a CompiledChemicals object.
        
        Examples
        --------
        >>> from thermosteam import Chemicals
        >>> chemicals = Chemicals(['Water', 'Ethanol'])
        >>> chemicals.compile()
        >>> chemicals
        CompiledChemicals([Water, Ethanol])
        
        """
        CompiledChemicals._compile(self)
        setattr(self, '__class__', CompiledChemicals)
    
    def subgroup(self, IDs):
        """
        Create a new subgroup of chemicals.
        
        Parameters
        ----------
        IDs : Iterable[str]
              Chemical identifiers.
              
        Examples
        --------
        
        >>> chemicals = Chemicals(['Water', 'Ethanol', 'Propane'])
        >>> chemicals.subgroup(['Propane', 'Water'])
        Chemicals([Propane, Water])
        
        """
        return type(self)([getattr(self, i) for i in IDs])
        
    def retrieve(self, IDs):
        """
        Return a list of chemicals.
        
        Parameters
        ----------
        IDs : Iterable[str]
              Chemical identifiers.
              
        Examples
        --------
        
        >>> chemicals = Chemicals(['Water', 'Ethanol', 'Propane'])
        >>> chemicals.retrieve(['Propane', 'Water'])
        [Chemical('Propane'), Chemical('Water')]
        
        """
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
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(self.__dict__)}])"


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
        
    Attributes
    ----------
    tuple : tuple[Chemical]
            All compiled chemicals.
    size : int
           Number of chemicals.
    IDs : tuple[str]
          IDs of all chemicals.
    CASs : tuple[str]
           CASs of all chemicals
    MW : 1d ndarray
         MWs of all chemicals.
    Hf : 1d ndarray
         Heats of formation of all chemicals.
    Hc : 1d ndarray
         Heats of combustion of all chemicals.
    equilibrium_chemicals : tuple[str]
                            IDs of chemicals that may have multiple phases.
    heavy_chemicals : tuple[str]
                      IDs of chemicals that are only present in liquid or solid phases.
    light_chemicals : tuple[str]
                      IDs of chemicals that are only present in gas phases.
        
    Examples
    --------
    
    Create a CompiledChemicals object from chemical identifiers
    
    >>> from thermosteam import CompiledChemicals, Chemical
    >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
    >>> chemicals
    CompiledChemicals([Water, Ethanol])
    
    All chemicals are stored as attributes:
        
    >>> chemicals.Water, chemicals.Ethanol
    (Chemical('Water'), Chemical('Ethanol'))
    
    Note that because they are compiled, the append and extend methods do not work:
        
    >>> # Propane = Chemical('Propane')
    >>> # chemicals.append(Propane)
    >>> # TypeError: 'CompiledChemicals' object is read-only
        
    
    """
    _cache = {}
    def __new__(cls, chemicals):
        isa = isinstance
        chemicals = tuple([chem if isa(chem, Chemical) else Chemical(chem)
                           for chem in chemicals])        
        cache = cls._cache
        if chemicals in cache:
            self = cache[chemicals]
        else:
            self = super().__new__(cls)
            setfield = setattr
            for chem in chemicals:
                setfield(self, chem.ID, chem)
            self._compile()
        return self
    
    def __dir__(self):
        return self.IDs + tuple(dir(type(self)))
    
    def __reduce__(self):
        return CompiledChemicals, (self.tuple,)
    
    def compile(self):
        """Do nothing, CompiledChemicals objects are already compiled.""" 
    
    def refresh_constants(self):
        """
        Refresh constant arrays according to their chemical values,
        including the molecular weight, heats of formation,
        and heats of combustion.
        
        Examples
        --------
        Some chemical constants may not be defined in thermosteam, 
        such as the heat of combustion.
        We can update it and refresh the compiled constants:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Glucose'])
        >>> chemicals.Glucose.HHV = -2291836.024
        >>> chemicals.refresh_constants()
        >>> chemicals.HHV
        array([-2291836.024])
        
        """
        dct = self.__dict__
        chemicals = self.tuple
        dct['MW'] = np.array([i.MW for i in chemicals])
        dct['Hf'] = np.array([i.Hf for i in chemicals])
        dct['LHV'] = np.array([i.LHV for i in chemicals])
        dct['HHV'] = np.array([i.HHV for i in chemicals])

    def get_combustion_reactions(self):
        """Return a ParallelReactions object with all defined combustion reactions."""
        reactions = [i.get_combustion_reaction(self) 
                         for i in self if i.combustion]
        return tmo.reaction.ParallelReaction(reactions)

    def _compile(self):
        dct = self.__dict__
        tuple_ = tuple
        chemicals = tuple_(dct.values())
        for chemical in chemicals:
            key_properties = chemical.get_key_property_names()
            missing_slots = chemical.get_missing_slots(key_properties)
            if not missing_slots: continue
            missing = repr_listed_values(missing_slots)
            raise RuntimeError(
                f"{chemical} is missing key thermodynamic properties ({missing}); "
                "use the `<Chemical>.get_missing_slots()` to check "
                "all missing properties")
        IDs = tuple_([i.ID for i in chemicals])
        CAS = tuple_([i.CAS for i in chemicals])
        N = len(IDs)
        index = tuple_(range(N))
        for i in chemicals: dct[i.CAS] = i
        dct['tuple'] = chemicals
        dct['size'] = N
        dct['IDs'] = IDs
        dct['CASs'] = tuple_([i.CAS for i in chemicals])
        dct['MW'] = chemical_data_array(chemicals, 'MW')
        dct['Hf'] = chemical_data_array(chemicals, 'Hf')
        dct['LHV'] = chemical_data_array(chemicals, 'LHV')
        dct['HHV'] = chemical_data_array(chemicals, 'HHV')
        dct['_index'] = index = dict((*zip(CAS, index),
                                      *zip(IDs, index)))
        dct['_index_cache'] = {}
        equilibrium_chemicals = []
        heavy_chemicals = []
        light_chemicals = []
        for i in chemicals:
            locked_phase = i.locked_state
            if locked_phase:
                if locked_phase in ('s', 'l'):
                    heavy_chemicals.append(i)
                elif locked_phase == 'g':
                    light_chemicals.append(i)
                else:
                    raise Exception('chemical locked state has an invalid phase')
            else:
                equilibrium_chemicals.append(i)
        dct['equilibrium_chemicals'] = tuple_(equilibrium_chemicals)
        dct['heavy_chemicals'] = tuple_(heavy_chemicals)
        dct['light_chemicals'] = tuple_(light_chemicals)
        dct['_equilibrium_indices'] = eq_index = [index[i.ID] for i in equilibrium_chemicals]
        dct['_has_equilibrium'] = has_equilibrium = np.zeros(N, dtype=bool)
        dct['_heavy_indices'] = [index[i.ID] for i in heavy_chemicals]
        dct['_light_indices'] = [index[i.ID] for i in light_chemicals]
        has_equilibrium[eq_index] = True
    
    def subgroup(self, IDs):
        """
        Create a new subgroup of chemicals.
        
        Parameters
        ----------
        IDs : Iterable[str]
              Chemical identifiers.
              
        Examples
        --------
        
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol', 'Propane'])
        >>> chemicals.subgroup(['Propane', 'Water'])
        CompiledChemicals([Propane, Water])
        
        """
        chemicals = self.retrieve(IDs)
        new = Chemicals(chemicals)
        new.compile()
        for i in new.IDs:
            for j in self.get_synonyms(i):
                try: new.set_synonym(i, j)
                except: pass
        return new
    
    def get_synonyms(self, ID):
        """Get all synonyms of a chemical.
        
        Parameters
        ----------
        ID : str
            Chemical identifier.
            
        Examples
        --------
        Get all synonyms of water:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water'])
        >>> chemicals.get_synonyms('Water')
        ['7732-18-5', 'Water']
        
        
        """
        k = self._index[ID]
        return [i for i, j in self._index.items() if j==k] 

    def set_synonym(self, ID, synonym):
        """
        Set a new synonym for a chemical.
        
        Parameters
        ----------
        ID : str
            Chemical identifier.
        synonym : str
            New identifier for chemical.
            
        Examples
        --------
        Set new synonym for water:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water'])
        >>> chemicals.set_synonym('Water', 'H2O')
        >>> chemicals.H2O is chemicals.Water
        True
        
        
        """
        chemical = getattr(self, ID)
        dct = self.__dict__
        if synonym in dct and dct[synonym] is not chemical:
            raise ValueError(f"synonym '{synonym}' already in use by {repr(dct[synonym])}")
        else:
            self._index[synonym] = self._index[ID]
            dct[synonym] = chemical
    
    def kwarray(self, ID_data):
        """
        Return an array with entries that correspond to the orded chemical IDs.
        
        Parameters
        ----------
        ID_data : dict
                 ID-data pairs.
            
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.kwarray(dict(Water=2))
        array([2., 0.])
        
        """
        return self.array(*zip(*ID_data.items()))
    
    def array(self, IDs, data):
        """
        Return an array with entries that correspond to the ordered chemical IDs.
        
        Parameters
        ----------
        IDs : iterable
              Compound IDs.
        data : array_like
               Data corresponding to IDs.
            
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.array(['Water'], [2])
        array([2., 0.])
        
        """
        array = np.zeros(len(self))
        array[self.get_index(tuple(IDs))] = data
        return array

    def iarray(self, IDs, data):
        """
        Return a chemical indexer.
        
        Parameters
        ----------
        IDs : iterable
              Chemical IDs.
        data : array_like
               Data corresponding to IDs.
            
        Examples
        --------
        Create a chemical indexer from chemical IDs and data:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> chemical_indexer = chemicals.iarray(['Water', 'Ethanol'], [2., 1.])
        >>> chemical_indexer.show()
        ChemicalIndexer:
         Water    2
         Ethanol  1
        
        Note that indexers allow for computationally efficient indexing using identifiers:
            
        >>> chemical_indexer['Ethanol', 'Water']
        array([1., 2.])
        >>> chemical_indexer['Ethanol']
        1.0
        
        """
        array = self.array(IDs, data)
        return ChemicalIndexer.from_data(array, phase=None, chemicals=self)

    def ikwarray(self, ID_data):
        """
        Return a chemical indexer.
        
        Parameters
        ----------
        ID_data : Dict[str: float]
              Chemical ID-value pairs.
            
        Examples
        --------
        Create a chemical indexer from chemical IDs and data:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> chemical_indexer = chemicals.ikwarray(dict(Water=2., Ethanol=1.))
        >>> chemical_indexer.show()
        ChemicalIndexer:
         Water    2
         Ethanol  1
        
        Note that indexers allow for computationally efficient indexing using identifiers:
            
        >>> chemical_indexer['Ethanol', 'Water']
        array([1., 2.])
        >>> chemical_indexer['Ethanol']
        1.0
        
        """
        array = self.kwarray(ID_data)
        return ChemicalIndexer.from_data(array, phase=None, chemicals=self)

    def index(self, ID):
        """
        Return index of specified chemical.

        Parameters
        ----------
        ID: str
            Chemical ID

        Examples
        --------
        Index by ID:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.index('Water')
        0

        Indices by CAS number:
        
        >>> chemicals.index('7732-18-5')
        0

        """
        try: return self._index[ID]
        except KeyError:
            raise UndefinedChemical(ID)

    def indices(self, IDs):
        """
        Return indices of specified chemicals.

        Parameters
        ----------
        IDs : iterable
              Chemical IDs or CAS numbers.

        Examples
        --------
        Indices by ID:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.indices(['Water', 'Ethanol'])
        [0, 1]

        Indices by CAS number:
        
        >>> chemicals.indices(['7732-18-5', '64-17-5'])
        [0, 1]

        """
        try:
            dct = self._index
            return [dct[i] for i in IDs]
        except:
            for i in IDs:
                if i not in dct: raise UndefinedChemical(i)     
    
    def get_index(self, key):
        """
        Return index/indices of specified chemicals.

        Parameters
        ----------
        key : iterable[str] or str
              A single chemical identifier or multiple.

        Notes
        -----
        CAS numbers are also supported.

        Examples
        --------
        Get multiple indices:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.get_index(['Water', 'Ethanol'])
        [0, 1]
        
        Get a single index:
        
        >>> chemicals.get_index('Ethanol')
        1

        """
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
        
    def get_equilibrium_indices(self, nonzeros):
        """
        Return indices of species in equilibrium
        given an array dictating whether or not
        the chemicals are present.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> data = chemicals.kwarray(dict(Water=2., Ethanol=1.))
        >>> chemicals.get_equilibrium_indices(data!=0)
        array([0, 2], dtype=int64)
        
        """
        return np.where(self._has_equilibrium & nonzeros)[0]
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(self.IDs)}])"