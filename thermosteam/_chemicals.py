# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
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
                    "run <Chemicals>.compile() to compile")

def chemical_data_array(chemicals, attr):
    getfield = getattr
    data = np.asarray([getfield(i, attr) for i in chemicals], dtype=float)
    data.setflags(0)
    return data
    

# %% Chemicals

class Chemicals:
    """
    Create a Chemicals object that contains Chemical objects as attributes.

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
    
    def __getnewargs__(self):
        return (tuple(self),)
    
    def __setattr__(self, ID, chemical):
        raise TypeError("can't set attribute; use <Chemicals>.append instead")
    
    def __setitem__(self, ID, chemical):
        raise TypeError("can't set item; use <Chemicals>.append instead")
    
    def __getitem__(self, key):
        """
        Return a chemical or a list of chemicals.
        
        Parameters
        ----------
        key : Iterable[str] or str
              Chemical identifiers.
              
        Examples
        --------
        
        >>> chemicals = Chemicals(['Water', 'Ethanol', 'Propane'])
        >>> chemicals['Propane', 'Water']
        [Chemical('Propane'), Chemical('Water')]
        
        """
        dct = self.__dict__
        try:
            if isinstance(key, str):
                return dct[key]
            else:
                return [dct[i] for i in key]
        except KeyError as key_error:
            raise UndefinedChemical(key_error.args[0])
    
    def append(self, chemical):
        """Append a Chemical."""
        assert isinstance(chemical, Chemical), ("only 'Chemical' objects are allowed, "
                                               f"not '{type(chemical).__name__}'")
        ID = chemical.ID
        assert ID not in self.__dict__, f"{ID} already defined in chemicals"
        setattr(self, ID, chemical)
    
    def extend(self, chemicals):
        """Extend with more Chemical objects."""
        if isinstance(chemicals, Chemicals):
            self.__dict__.update(chemicals.__dict__)
        else:
            for chemical in chemicals:
                assert isinstance(chemical, Chemical), ("only 'Chemical' objects are allowed, "
                                                        f"not '{type(chemical).__name__}'")
                setattr(self, chemical.ID, chemical)
    
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
    
    kwarray = array = index = indices = must_compile
        
    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, chemical):
        if isinstance(chemical, str):
            return chemical in self.__dict__
        elif isinstance(chemical, Chemical):
            return chemical in self.__dict__.values()
        else:
            return False
    
    def __iter__(self):
        yield from self.__dict__.values()
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(self.__dict__)}])"


@read_only(methods=('append', 'extend', '__setitem__'))
class CompiledChemicals(Chemicals):
    """
    Create a CompiledChemicals object that contains Chemical objects as attributes.

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
    vle_chemicals : tuple[Chemical]
        Chemicals that may have vapor and liquid phases.
    lle_chemicals : tuple[Chemical]
        Chemicals that may have two liquid phases.
    heavy_chemicals : tuple[Chemical]
        Chemicals that are only present in liquid or solid phases.
    light_chemicals : tuple[Chemical]
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
        return ('append', 'array', 'compile', 'extend', 
                'get_combustion_reactions', 'get_index',
                'get_lle_indices', 'get_synonyms',
                'get_vle_indices', 'iarray', 'ikwarray',
                'index', 'indices', 'kwarray', 'refresh_constants', 
                'set_synonym', 'subgroup') + self.IDs
    
    def __reduce__(self):
        return CompiledChemicals, (self.tuple,)
    
    def compile(self):
        """Do nothing, CompiledChemicals objects are already compiled.""" 
    
    def refresh_constants(self):
        """
        Refresh constant arrays according to their chemical values,
        including the molecular weight, heats of formation,
        and heats of combustion.
        
        """
        dct = self.__dict__
        chemicals = self.tuple
        dct['MW'] = chemical_data_array([i.MW for i in chemicals])
        dct['Hf'] = chemical_data_array([i.Hf for i in chemicals])
        dct['LHV'] = chemical_data_array([i.LHV for i in chemicals])
        dct['HHV'] = chemical_data_array([i.HHV for i in chemicals])

    def get_combustion_reactions(self):
        """Return a ParallelReactions object with all defined combustion reactions."""
        reactions = [i.get_combustion_reaction(self) for i in self]
        return tmo.reaction.ParallelReaction([i for i in reactions if i is not None])

    def _compile(self):
        dct = self.__dict__
        tuple_ = tuple
        chemicals = tuple_(dct.values())
        free_energies = ('H', 'S', 'H_excess', 'S_excess')
        for chemical in chemicals:
            if chemical.get_missing_properties(free_energies):
                chemical.reset_free_energies()
            key_properties = chemical.get_key_property_names()
            missing_properties = chemical.get_missing_properties(key_properties)
            if not missing_properties: continue
            missing = repr_listed_values(missing_properties)
            raise RuntimeError(
                f"{chemical} is missing key thermodynamic properties ({missing}); "
                "use the `<Chemical>.get_missing_properties()` to check "
                "all missing properties")
        IDs = tuple_([i.ID for i in chemicals])
        CAS = tuple_([i.CAS for i in chemicals])
        size = len(IDs)
        index = tuple_(range(size))
        for i in chemicals: dct[i.CAS] = i
        dct['tuple'] = chemicals
        dct['size'] = size
        dct['IDs'] = IDs
        dct['CASs'] = tuple_([i.CAS for i in chemicals])
        dct['MW'] = chemical_data_array(chemicals, 'MW')
        dct['Hf'] = chemical_data_array(chemicals, 'Hf')
        dct['LHV'] = chemical_data_array(chemicals, 'LHV')
        dct['HHV'] = chemical_data_array(chemicals, 'HHV')
        dct['_index'] = index = dict((*zip(CAS, index),
                                      *zip(IDs, index)))
        dct['_index_cache'] = {}
        vle_chemicals = []
        lle_chemicals = []
        heavy_chemicals = []
        light_chemicals = []
        for i in chemicals:
            locked_phase = i.locked_state
            if locked_phase:
                if locked_phase in ('s', 'l'):
                    heavy_chemicals.append(i)
                    if i.Dortmund or i.UNIFAC:
                        lle_chemicals.append(i)
                elif locked_phase == 'g':
                    light_chemicals.append(i)
                else:
                    raise Exception('chemical locked state has an invalid phase')
            else:
                vle_chemicals.append(i)
                lle_chemicals.append(i)
        dct['vle_chemicals'] = tuple_(vle_chemicals)
        dct['lle_chemicals'] = tuple_(lle_chemicals)
        dct['heavy_chemicals'] = tuple_(heavy_chemicals)
        dct['light_chemicals'] = tuple_(light_chemicals)
        dct['_has_vle'] = has_vle = np.zeros(size, dtype=bool)
        dct['_has_lle'] = has_lle = np.zeros(size, dtype=bool)
        dct['_heavy_indices'] = [index[i.ID] for i in heavy_chemicals]
        dct['_light_indices'] = [index[i.ID] for i in light_chemicals]
        vle_index = [index[i.ID] for i in vle_chemicals]
        lle_index = [index[i.ID] for i in lle_chemicals]
        has_vle[vle_index] = True
        has_lle[lle_index] = True
        
    @property
    def formula_array(self):
        """
        An array describing the formulas of all chemicals.
        Each column is a chemical and each row an element.
        Rows are ordered by atomic number.
        
        Examples
        --------
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol', 'Propane'])
        >>> chemicals.formula_array
        array([[2., 6., 8.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 2., 3.],
               [0., 0., 0.],
               [1., 1., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
        try: return self._formula_array
        except: pass
        self.__dict__['_formula_array'] = formula_array = np.zeros((118, self.size))
        atoms_to_array = tmo.chemicals.elements.atoms_to_array
        for i, chemical in enumerate(self):
            formula_array[:, i] = atoms_to_array(chemical.atoms)
        formula_array.setflags(0)
        return formula_array
    
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
        chemicals = self[IDs]
        new = Chemicals(chemicals)
        new.compile()
        for i in new.IDs:
            for j in self.get_synonyms(i):
                try: new.set_synonym(i, j)
                except: pass
        return new
    
    def get_synonyms(self, ID):
        """
        Get all synonyms of a chemical.
        
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
    
    def zeros(self):
        """
        Return an array of zeros with entries that correspond to the orded chemical IDs.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.zeros()
        array([0., 0.])
        
        """
        return np.zeros(self.size) 
    
    def ones(self):
        """
        Return an array of ones with entries that correspond to the orded chemical IDs.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> chemicals.ones()
        array([1., 1.])
        
        """
        return np.ones(self.size) 
    
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
        array = self.zeros()
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
        return ChemicalIndexer.from_data(array, chemicals=self)

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
        return ChemicalIndexer.from_data(array, chemicals=self)

    def isplit(self, split, order=None):
        """
        Create a chemical indexer that represents chemical splits.
    
        Parameters
        ----------   
        split : Should be one of the following
                * [float] Split fraction
                * [array_like] Componentwise split 
                * [dict] ID-split pairs
        order=None : Iterable[str], options
            Chemical order of split. Defaults to biosteam.settings.chemicals.IDs
           
        Examples
        --------
        From a dictionary:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> chemical_indexer = chemicals.isplit(dict(Water=0.5, Ethanol=1.))
        >>> chemical_indexer.show()
        ChemicalIndexer:
         Water    0.5
         Ethanol  1
        
        From iterable given the order:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> chemical_indexer = chemicals.isplit([0.5, 1], ['Water', 'Ethanol'])
        >>> chemical_indexer.show()
        ChemicalIndexer:
         Water    0.5
         Ethanol  1
           
        From a fraction:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> chemical_indexer = chemicals.isplit(0.75)
        >>> chemical_indexer.show()
        ChemicalIndexer:
         Water     0.75
         Methanol  0.75
         Ethanol   0.75
            
        """
        if isinstance(split, dict):
            assert not order, "cannot pass 'order' key word argument when split is a dictionary"
            order, split = zip(*split.items())
        
        if order:
            isplit = self.iarray(order, split)
        elif hasattr(split, '__len__'):
            isplit = ChemicalIndexer.from_data(np.asarray(split),
                                               phase=None,
                                               chemicals=self)
        else:
            split = split * np.ones(self.size)
            isplit = ChemicalIndexer.from_data(split,
                                               phase=None,
                                               chemicals=self)
        return isplit

    def index(self, ID):
        """
        Return index of specified chemical.

        Parameters
        ----------
        ID: str
            Chemical identifier.

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
              Chemical indentifiers.

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
        except KeyError as key_error:
            raise UndefinedChemical(key_error.args[0])
    
    def get_index(self, IDs):
        """
        Return index/indices of specified chemicals.

        Parameters
        ----------
        IDs : iterable[str] or str
              Chemical identifiers.

        Notes
        -----
        CAS numbers are also supported.

        Examples
        --------
        Get multiple indices:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'])
        >>> IDs = ('Water', 'Ethanol')
        >>> chemicals.get_index(IDs)
        [0, 1]
        
        Get a single index:
        
        >>> chemicals.get_index('Ethanol')
        1

        """
        cache = self._index_cache
        try: 
            index = cache[IDs]
        except KeyError: 
            if len(cache) > 1000: cache.clear()
            cache[IDs] = index = self._get_index(IDs)
        except TypeError:
            raise TypeError("only strings, tuples, and ellipsis are valid index keys")
        return index
    
    def _get_index(self, IDs):
        if isinstance(IDs, str):
            return self.index(IDs)
        elif isinstance(IDs, tuple):
            return self.indices(IDs)
        elif IDs is ...:
            return IDs
        else:
            raise TypeError("only strings, tuples, and ellipsis are valid index keys")    
    
    def __len__(self):
        return self.size
    
    def __contains__(self, chemical):
        if isinstance(chemical, str):
            return chemical in self.__dict__
        elif isinstance(chemical, Chemical):
            return chemical in self.tuple
        else:
            return False
    
    def __iter__(self):
        return iter(self.tuple)
    
    def get_vle_indices(self, nonzeros):
        """
        Return indices of species in vapor-liquid equilibrium given an array
        dictating whether or not the chemicals are present.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> data = chemicals.kwarray(dict(Water=2., Ethanol=1.))
        >>> chemicals.get_vle_indices(data!=0)
        array([0, 2], dtype=int64)
        
        """
        return np.where(self._has_vle & nonzeros)[0]
    
    def get_lle_indices(self, nonzeros):
        """
        Return indices of species in liquid-liquid equilibrium given an array
        dictating whether or not the chemicals are present.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> data = chemicals.kwarray(dict(Water=2., Ethanol=1.))
        >>> chemicals.get_lle_indices(data!=0)
        array([0, 2], dtype=int64)
        
        """
        return np.where(self._has_lle & nonzeros)[0]
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(self.IDs)}])"