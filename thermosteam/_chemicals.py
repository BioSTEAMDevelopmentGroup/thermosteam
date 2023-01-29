# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from . import utils
from .exceptions import UndefinedChemicalAlias
from ._chemical import Chemical
from .indexer import ChemicalIndexer, SplitIndexer
from collections.abc import Sequence
import thermosteam as tmo
import numpy as np

__all__ = ('Chemicals', 'CompiledChemicals')
setattr = object.__setattr__
    
# %% Functions

def must_compile(*args, **kwargs): # pragma: no cover
    raise TypeError("method valid only for compiled chemicals; "
                    "run <Chemicals>.compile() to compile")

def chemical_data_array(chemicals, attr, dtype=float):
    getfield = getattr
    data = np.asarray([getfield(i, attr) for i in chemicals], dtype)
    data.setflags(0)
    return data
    

# %% Chemicals

class Chemicals:
    """
    Create a Chemicals object that contains Chemical objects as attributes.

    Parameters
    ----------
    chemicals : Iterable[str or :class:`~thermosteam.Chemical`]
        Strings should be one of the following [-]:
           * Name, in IUPAC form or common form or an alias registered in PubChem
           * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
           * InChI key, prefixed by 'InChIKey='
           * PubChem CID, prefixed by 'PubChem='
           * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
           * CAS number
    cache : bool, optional
        Whether or not to use cached chemicals.
    
    Examples
    --------
    Create a Chemicals object from chemical identifiers:
    
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'], cache=True)
    >>> chemicals
    Chemicals([Water, Ethanol])
    
    All chemicals are stored as attributes:
        
    >>> chemicals.Water, chemicals.Ethanol
    (Chemical('Water'), Chemical('Ethanol'))
    
    Chemicals can also be accessed as items:
    
    >>> chemicals = Chemicals(['Water', 'Ethanol', 'Propane'], cache=True)
    >>> chemicals['Ethanol']
    Chemical('Ethanol')
    >>> chemicals['Propane', 'Water']
    [Chemical('Propane'), Chemical('Water')]
    
    A Chemicals object can be extended with more chemicals:
        
    >>> from thermosteam import Chemical
    >>> Methanol = Chemical('Methanol')
    >>> chemicals.append(Methanol)
    >>> chemicals
    Chemicals([Water, Ethanol, Propane, Methanol])
    >>> new_chemicals = Chemicals(['Hexane', 'Octanol'], cache=True)
    >>> chemicals.extend(new_chemicals)
    >>> chemicals
    Chemicals([Water, Ethanol, Propane, Methanol, Hexane, Octanol])
    
    Chemical objects cannot be repeated:
    
    >>> chemicals.append(chemicals.Water)
    Traceback (most recent call last):
    ValueError: Water already defined in chemicals
    >>> chemicals.extend(chemicals['Ethanol', 'Octanol'])
    Traceback (most recent call last):
    ValueError: Ethanol already defined in chemicals
    
    A Chemicals object can only contain Chemical objects:
        
    >>> chemicals.append(10)
    Traceback (most recent call last):
    TypeError: only 'Chemical' objects can be appended, not 'int'
    
    You can check whether a Chemicals object contains a given chemical:
        
    >>> 'Water' in chemicals
    True
    >>> chemicals.Water in chemicals
    True
    >>> 'Butane' in chemicals
    False
    
    An attempt to access a non-existent chemical raises an UndefinedChemicalAlias error:
    
    >>> chemicals['Butane']
    Traceback (most recent call last):
    UndefinedChemicalAlias: 'Butane'
    
    """
    def __new__(cls, chemicals, cache=None):
        self = super().__new__(cls)
        isa = isinstance
        setfield = setattr
        CASs = set()
        chemicals = [i if isa(i, Chemical) else Chemical(i, cache=cache) for i in chemicals]
        for i in chemicals:
            CAS = i.CAS
            if CAS in CASs: continue
            CASs.add(CAS)
            setfield(self, i.ID, i)
        return self
    
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
        
        """
        dct = self.__dict__
        try:
            if isinstance(key, str):
                return dct[key]
            else:
                return [dct[i] for i in key]
        except KeyError as key_error:
            raise UndefinedChemicalAlias(key_error.args[0])
    
    def copy(self):
        """Return a copy."""
        copy = object.__new__(Chemicals)
        for chem in self: setattr(copy, chem.ID, chem)
        return copy
    
    def append(self, chemical):
        """Append a Chemical."""
        if isinstance(chemical, str):
            chemical = Chemical(chemical)
        elif not isinstance(chemical, Chemical):
            raise TypeError("only 'Chemical' objects can be appended, "
                           f"not '{type(chemical).__name__}'")
        ID = chemical.ID
        if ID in self.__dict__:
            raise ValueError(f"{ID} already defined in chemicals")
        setattr(self, ID, chemical)
    
    def extend(self, chemicals):
        """Extend with more Chemical objects."""
        if isinstance(chemicals, Chemicals):
            self.__dict__.update(chemicals.__dict__)
        else:
            for chemical in chemicals: self.append(chemical)
    
    def compile(self, skip_checks=False):
        """
        Cast as a CompiledChemicals object.
        
        Parameters
        ----------
        skip_checks : bool, optional
            Whether to skip checks for missing or invalid properties.
            
        Warning
        -------
        If checks are skipped, certain features in thermosteam (e.g. phase equilibrium)
        cannot be guaranteed to function properly. 
        
        Examples
        --------
        Compile ethanol and water chemicals:
        
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> chemicals.compile()
        >>> chemicals
        CompiledChemicals([Water, Ethanol])
        
        Attempt to compile chemicals with missing properties:
            
        >>> Substance = tmo.Chemical('Substance', search_db=False)
        >>> chemicals = tmo.Chemicals([Substance])
        >>> chemicals.compile()
        Traceback (most recent call last):
        RuntimeError: Substance is missing key thermodynamic properties 
        (V, S, H, Cn, Psat, Tb and Hvap); use the `<Chemical>.get_missing_properties()` 
        to check all missing properties
        
        Compile chemicals with missing properties (skipping checks) and note 
        how certain features do not work:
        
        >>> chemicals.compile(skip_checks=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> s = tmo.Stream('s', Substance=10)
        >>> s.rho
        Traceback (most recent call last):
        RuntimeError: No liquid molar volume method selected for 
        component with CASRN 'Substance'
        
        """
        chemicals = tuple(self)
        setattr(self, '__class__', CompiledChemicals)
        try: self._compile(chemicals, skip_checks)
        except Exception as error:
            setattr(self, '__class__', Chemicals)
            setattr(self, '__dict__', {i.ID: i for i in chemicals})
            raise error
    
    kwarray = array = index = indices = must_compile
        
    def show(self):
        print(self)
    _ipython_display_ = show
    
    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, chemical):
        if isinstance(chemical, str):
            return chemical in self.__dict__
        elif isinstance(chemical, Chemical):
            return chemical in self.__dict__.values()
        else: # pragma: no cover
            return False
    
    def __iter__(self):
        yield from self.__dict__.values()
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(self.__dict__)}])"


@utils.read_only(methods=('append', 'extend', '__setitem__'))
class CompiledChemicals(Chemicals):
    """
    Create a CompiledChemicals object that contains Chemical objects as attributes.

    Parameters
    ----------
    chemicals : Iterable[str or Chemical]
           Strings should be one of the following [-]:
              * Name, in IUPAC form or common form or an alias registered in PubChem
              * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
              * InChI key, prefixed by 'InChIKey='
              * PubChem CID, prefixed by 'PubChem='
              * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
              * CAS number
    cache : optional
        Whether or not to use cached chemicals.
        
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
    >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
    >>> chemicals
    CompiledChemicals([Water, Ethanol])
    
    All chemicals are stored as attributes:
        
    >>> chemicals.Water, chemicals.Ethanol
    (Chemical('Water'), Chemical('Ethanol'))
    
    Note that because they are compiled, the append and extend methods do not work:
        
    >>> Propane = Chemical('Propane', cache=True)
    >>> chemicals.append(Propane)
    Traceback (most recent call last):
    TypeError: 'CompiledChemicals' object is read-only
    
    You can check whether a Chemicals object contains a given chemical:
        
    >>> 'Water' in chemicals
    True
    >>> chemicals.Water in chemicals
    True
    >>> 'Butane' in chemicals
    False
    
    """  
    _cache = {}
    
    def __new__(cls, chemicals, cache=None):
        chemicals = tmo.Chemicals(chemicals)
        chemicals_tuple = tuple(chemicals) 
        cache = cls._cache
        if chemicals in cache:
            self = cache[chemicals]
        else:
            chemicals.compile(cache)
            self = cache[chemicals_tuple] = chemicals
        return self
    
    def __hash__(self):
        return hash(self.IDs)
    
    def __dir__(self):
        return ('append', 'array', 'compile', 'extend', 
                'get_combustion_reactions', 'get_index',
                'get_lle_indices', 'get_aliases',
                'get_vle_indices', 'iarray', 'ikwarray',
                'index', 'indices', 'kwarray', 'refresh_constants', 
                'set_alias') + self.IDs
    
    def __reduce__(self):
        return CompiledChemicals, (self.tuple,)
    
    def compile(self, skip_checks=False):
        """Do nothing, CompiledChemicals objects are already compiled.""" 
    
    def define_group(self, name, IDs, composition=None, wt=False):
        """
        Define a group of chemicals.
        
        Parameters
        ----------
        name : str
            Name of group.
        IDs : List[str]
            IDs of chemicals in the group.
        composition : List[float], optional
            Default composition of chemical group. 
        wt : bool, optional
            Whether composition is given by weight. Defaults to False.
            
        Examples
        --------
        Create a chemical group and get the index:
        
        >>> import thermosteam as tmo
        >>> chemicals = tmo.CompiledChemicals(['Water', 'Methanol', 'Ethanol'], cache=True)
        >>> chemicals.define_group('Alcohol', ['Methanol', 'Ethanol'], composition=[0.5, 0.5])
        >>> chemicals.get_index('Alcohol')
        [1, 2]
        >>> chemicals.get_index(('Water', 'Alcohol'))
        [0, [1, 2]]
        
        By defining a chemical group, you can conveniently use indexers
        to retrieve the total value of the group:
            
        >>> # Single phase stream case
        >>> tmo.settings.set_thermo(chemicals)
        >>> s1 = tmo.Stream(ID='s1', Water=2)
        >>> s1.imol['Alcohol']
        0
        >>> s1.imol['Methanol', 'Ethanol'] = [2., 1.]
        >>> s1.imol['Water', 'Alcohol']
        array([2., 3.])
        
        >>> # Multi-phase stream case
        >>> s2 = tmo.Stream(ID='s2', Water=2, Methanol=1, Ethanol=1)
        >>> s2.vle(V=0.5, P=101325)
        >>> s2.imol['Alcohol']
        2.0
        >>> s2.imol['l', 'Alcohol']
        0.678
        >>> s2.imol['l', ('Water', 'Alcohol')]
        array([1.321, 0.679])
        
        Because groups are defined with a composition we can set values by 
        groups as well:
            
        >>> s1.imol['Alcohol'] = 3.
        >>> s1.imol['Ethanol', 'Methanol']
        array([1.5, 1.5])
        
        >>> s1.imol['Water', 'Alcohol'] = [3., 1.]
        >>> s1.imol['Water', 'Ethanol', 'Methanol']
        array([3. , 0.5, 0.5])
        
        >>> s2.imol['l', 'Alcohol'] = 1.
        >>> s2.imol['l', ('Ethanol', 'Methanol')]
        array([0.5, 0.5])
        
        For SplitIndexer objects, which reflect the separation of chemicals 
        in two streams, group names correspond to nested indexes (without a composition):
            
        >>> # Create a split-indexer
        >>> indexer = chemicals.isplit(0.7)
        >>> indexer.show()
        SplitIndexer:
         Water     0.7
         Methanol  0.7
         Ethanol   0.7
        
        >>> # Normal index
        >>> indexer['Alcohol'] = [0.50, 0.80] # Methanol and ethanol
        >>> indexer['Alcohol']
        array([0.5, 0.8])
        
        >>> # Broadcasted index
        >>> indexer['Alcohol'] = 0.9
        >>> indexer['Alcohol']
        array([0.9, 0.9])
        
        >>> # Nested index
        >>> indexer['Water', 'Alcohol'] = [0.2, [0.6, 0.7]]
        >>> indexer['Water', 'Alcohol']
        array([0.2, array([0.6, 0.7])], dtype=object)
        
        >>> # Broadcasted and nested index
        >>> indexer['Water', 'Alcohol'] = [0.2, 0.8]
        >>> indexer['Water', 'Alcohol']
        array([0.2, array([0.8, 0.8])], dtype=object)
        
        This feature allows splits to be easily defined for groups of chemicals 
        in BioSTEAM.
        
        """
        IDs = tuple(IDs)
        if composition is None:
            composition = np.ones(len(IDs))
        elif len(composition) != len(IDs): 
            raise ValueError('length of IDs and composition must be the same')
        for i in IDs: 
            if i in self._group_wt_compositions:
                raise ValueError(f"'{i}' is a group; cannot define new group using other groups")
        index = self.indices(IDs)
        self.__dict__[name] = [self.tuple[i] for i in index]
        self._index[name] = index
        composition = np.asarray(composition, float)
        if wt:
            composition_wt = composition
            composition_mol = composition / self.MW[index]
        else:
            composition_wt = composition * self.MW[index]
            composition_mol = composition
        self._group_wt_compositions[name] = composition_wt / composition_wt.sum()
        self._group_mol_compositions[name] = composition_mol / composition_mol.sum()
    
    @property
    def chemical_groups(self) -> frozenset[str]:
        """All defined chemical groups."""
        return frozenset(self._group_mol_compositions)
    
    def refresh_constants(self):
        """
        Refresh constant arrays according to their chemical values,
        including the molecular weight, heats of formation,
        and heats of combustion.
        
        """
        dct = self.__dict__
        chemicals = self.tuple
        dct['MW'] = chemical_data_array(chemicals, 'MW')
        dct['Hf'] = chemical_data_array(chemicals, 'Hf')
        dct['LHV'] = chemical_data_array(chemicals, 'LHV')
        dct['HHV'] = chemical_data_array(chemicals, 'HHV')

    def get_combustion_reactions(self):
        """
        Return a ParallelReactions object with all defined combustion reactions.
        
        Examples
        --------
        >>> chemicals = CompiledChemicals(['H2O', 'Methanol', 'Ethanol', 'CO2', 'O2'], cache=True)
        >>> rxns = chemicals.get_combustion_reactions()
        >>> rxns.show()
        ParallelReaction (by mol):
        index  stoichiometry                     reactant    X[%]
        [0]    Methanol + 1.5 O2 -> 2 H2O + CO2  Methanol  100.00
        [1]    Ethanol + 3 O2 -> 3 H2O + 2 CO2   Ethanol   100.00
        
        """
        reactions = [i.get_combustion_reaction(self) for i in self]
        return tmo.reaction.ParallelReaction([i for i in reactions if i is not None])

    def _compile(self, chemicals, skip_checks=False):
        dct = self.__dict__
        tuple_ = tuple
        free_energies = ('H', 'S', 'H_excess', 'S_excess')
        for chemical in chemicals:
            if chemical.get_missing_properties(free_energies):
                chemical.reset_free_energies()
            if skip_checks: continue
            key_properties = chemical.get_key_property_names()
            missing_properties = chemical.get_missing_properties(key_properties)
            if not missing_properties: continue
            missing = utils.repr_listed_values(missing_properties)
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
        dct['_group_wt_compositions'] = {}
        dct['_group_mol_compositions'] = {}
        dct['_index_cache'] = {}
        repeated_names = set()
        names = set()
        all_names_list = []
        isa = isinstance
        for i in chemicals:
            if not i.iupac_name: i.iupac_name = ()
            elif isa(i.iupac_name, str):
                i.iupac_name = (i.iupac_name,)
            all_names = set([*i.iupac_name, *i.aliases, i.common_name, i.formula])
            all_names_list.append(all_names)
            for name in all_names:
                if not name: continue
                if name in names:
                    repeated_names.add(name)
                else:
                    names.add(name)
        for all_names, i in zip(all_names_list, chemicals):
            ID = i.ID
            for name in all_names:
                if name and name not in repeated_names:
                    self.set_alias(ID, name)
        vle_chemicals = []
        lle_chemicals = []
        heavy_chemicals = []
        light_chemicals = []
        for i in chemicals:
            locked_phase = i.locked_state
            if locked_phase:
                if locked_phase in ('s', 'l'):
                    heavy_chemicals.append(i)
                    if i.Dortmund or i.UNIFAC or i.NIST or i.PSRK:
                        lle_chemicals.append(i)
                    if i.N_solutes is None: i._N_solutes = 0
                elif locked_phase == 'g':
                    light_chemicals.append(i)
                else:
                    raise Exception('chemical locked state has an invalid phase')
            else:
                vle_chemicals.append(i)
                if i.Dortmund or i.UNIFAC or i.NIST or i.PSRK: lle_chemicals.append(i)
        dct['vle_chemicals'] = tuple_(vle_chemicals)
        dct['lle_chemicals'] = tuple_(lle_chemicals)
        dct['heavy_chemicals'] = tuple_(heavy_chemicals)
        dct['light_chemicals'] = tuple_(light_chemicals)
        dct['_vle_index'] = [index[i.ID] for i in vle_chemicals]
        dct['_lle_index'] = [index[i.ID] for i in lle_chemicals]
        dct['_heavy_solutes'] = chemical_data_array(heavy_chemicals, 'N_solutes')
        dct['_heavy_indices'] = [index[i.ID] for i in heavy_chemicals]
        dct['_light_indices'] = [index[i.ID] for i in light_chemicals]
        
    @property
    def formula_array(self):
        """
        An array describing the formulas of all chemicals.
        Each column is a chemical and each row an element.
        Rows are ordered by atomic number.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol', 'Propane'], cache=True)
        >>> chemicals.formula_array.sum(0)
        array([ 3.,  9., 11.])
        
        """
        try: return self._formula_array
        except: pass
        self.__dict__['_formula_array'] = formula_array = np.zeros((118, self.size))
        atoms_to_array = tmo.chemicals.elements.atoms_to_array
        for i, chemical in enumerate(self):
            formula_array[:, i] = atoms_to_array(chemical.atoms)
        formula_array.setflags(0)
        return formula_array
    
    def get_parsable_alias(self, ID):
        """
        Return an alias of the given chemical identifier that can be 
        parsed by Python as a variable name
        
        Parameters
        ----------
        ID : str
            Chemical identifier.
            
        Examples
        --------
        Get parsable alias of 2,3-Butanediol:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['2,3-Butanediol'], cache=True)
        >>> chemicals.get_parsable_alias('2,3-Butanediol')
        'C4H10O2'
        
        """
        isvalid = utils.is_valid_ID      
        for i in self.get_aliases(ID):
            if isvalid(i): return i        
    
    get_parsable_synonym = get_parsable_alias
    
    def get_aliases(self, ID):
        """
        Return all aliases of a chemical.
        
        Parameters
        ----------
        ID : str
            Chemical identifier.
            
        Examples
        --------
        Get all aliases of water:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water'], cache=True)
        >>> aliases = chemicals.get_aliases('Water')
        >>> aliases.sort()
        >>> aliases
        ['7732-18-5', 'H2O', 'Water', 'oxidane', 'water']
        
        """
        k = self._index[ID]
        isa = isinstance
        return [i for i, j in self._index.items() if isa(j, int) and (j==k)]

    get_synonyms = get_aliases

    def set_alias(self, ID, alias):
        """
        Set a new alias for a chemical.
        
        Parameters
        ----------
        ID : str
            Chemical identifier.
        alias : str
            New identifier for chemical.
            
        Examples
        --------
        Set new alias for water:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
        >>> chemicals.set_alias('Water', 'H2O')
        >>> chemicals.H2O is chemicals.Water
        True
        
        Note that you cannot use one alias for two chemicals:
        
        >>> chemicals.set_alias('Ethanol', 'H2O')
        Traceback (most recent call last):
        ValueError: alias 'H2O' already in use by Chemical('Water')
        
        """
        dct = self.__dict__
        chemical = dct[ID]
        if alias in dct and dct[alias] is not chemical:
            raise ValueError(f"alias '{alias}' already in use by {repr(dct[alias])}")
        else:
            self._index[alias] = self._index[ID]
            dct[alias] = chemical
        chemical.aliases.add(alias)
    
    set_synonym = set_alias
    
    def zeros(self):
        """
        Return an array of zeros with entries that correspond to the orded chemical IDs.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
        >>> chemicals.zeros()
        array([0., 0.])
        
        """
        return np.zeros(self.size) 
    
    def ones(self):
        """
        Return an array of ones with entries that correspond to the ordered chemical IDs.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
        >>> chemicals.ones()
        array([1., 1.])
        
        """
        return np.ones(self.size) 
    
    def kwarray(self, ID_data):
        """
        Return an array with entries that correspond to the ordered chemical IDs.
        
        Parameters
        ----------
        ID_data : dict[str, float]
            ID-data pairs.
            
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
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
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
        >>> chemicals.array(['Water'], [2])
        array([2., 0.])
        
        """
        array =  np.zeros(self.size) 
        index, kind = self._get_index_and_kind(tuple(IDs))
        if kind == 0:
            array[index] = data
        elif kind == 1:
            raise ValueError('cannot create array by chemical groups')
        elif kind == 2:
            raise ValueError('cannot create array by chemical groups')
        else:
            raise IndexError('unknown error')
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
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'], cache=True)
        >>> indexer = chemicals.iarray(['Water', 'Ethanol'], [2., 1.])
        >>> indexer.show()
        ChemicalIndexer:
         Water    2
         Ethanol  1
        
        Note that indexers allow for computationally efficient indexing using identifiers:
            
        >>> indexer['Ethanol', 'Water']
        array([1., 2.])
        >>> indexer['Ethanol']
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
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'], cache=True)
        >>> indexer = chemicals.ikwarray(dict(Water=2., Ethanol=1.))
        >>> indexer.show()
        ChemicalIndexer:
         Water    2
         Ethanol  1
        
        Note that chemical indexers allow for computationally efficient 
        indexing using identifiers:
            
        >>> indexer['Ethanol', 'Water']
        array([1., 2.])
        >>> indexer['Ethanol']
        1.0
        
        """
        array = self.kwarray(ID_data)
        return ChemicalIndexer.from_data(array, chemicals=self)

    def isplit(self, split, order=None):
        """
        Create a SplitIndexer object that represents chemical splits.
    
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
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'], cache=True)
        >>> indexer = chemicals.isplit(dict(Water=0.5, Ethanol=1.))
        >>> indexer.show()
        SplitIndexer:
         Water    0.5
         Ethanol  1
        
        From iterable given the order:
        
        >>> indexer = chemicals.isplit([0.5, 1], ['Water', 'Ethanol'])
        >>> indexer.show()
        SplitIndexer:
         Water    0.5
         Ethanol  1
           
        From a fraction:
        
        >>> indexer = chemicals.isplit(0.75)
        >>> indexer.show()
        SplitIndexer:
         Water     0.75
         Methanol  0.75
         Ethanol   0.75
            
        From an iterable (assuming same order as the Chemicals object):
        
        >>> indexer = chemicals.isplit([0.5, 0, 1])
        >>> indexer.show()
        SplitIndexer:
         Water    0.5
         Ethanol  1

        """
        if isinstance(split, dict):
            assert not order, "cannot pass 'order' key word argument when split is a dictionary"
            order, split = zip(*split.items())
        
        if order:
            isplit = SplitIndexer.blank(chemicals=self)
            isplit[tuple(order)] = split
        elif hasattr(split, '__len__'):
            isplit = SplitIndexer.from_data(np.asarray(split),
                                            chemicals=self)
        else:
            split = split * np.ones(self.size)
            isplit = SplitIndexer.from_data(split,
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
            raise UndefinedChemicalAlias(ID)

    def indices(self, IDs):
        """
        Return indices of specified chemicals.

        Parameters
        ----------
        IDs : iterable
              Chemical identifiers.

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
        dct = self._index
        try:
            return [dct[i] for i in IDs]
        except KeyError as key_error:
            raise UndefinedChemicalAlias(key_error.args[0])
    
    def available_indices(self, IDs):
        """
        Return indices of all chemicals available.

        Parameters
        ----------
        IDs : iterable[str] or str
              Chemical identifiers.

        Notes
        -----
        CAS numbers are also supported.

        Examples
        --------
        Get available indices from IDs:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
        >>> IDs = ('Water', 'Ethanol', 'Octane')
        >>> chemicals.available_indices(IDs)
        [0, 1]

        """
        index = self._index
        return [index[i] for i in IDs if i in index]
            
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
        Get multiple indices with a tuple/list of IDs:
        
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Ethanol'], cache=True)
        >>> IDs = ('Water', 'Ethanol')
        >>> chemicals.get_index(IDs)
        [0, 1]
        
        Get a single index with a string:
        
        >>> chemicals.get_index('Ethanol')
        1
        
        An Ellipsis returns a slice:
        
        >>> chemicals.get_index(...)
        slice(None, None, None)

        Collections (without an order) raise an error:
        
        >>> chemicals.get_index({'Water', 'Ethanol'})
        Traceback (most recent call last):
        TypeError: only strings, sequences, and ellipsis are valid index keys

        """
        if isinstance(IDs, (str, Chemical)):
            return self.index(IDs)
        elif IDs is ...:
            return slice(None)
        elif isinstance(IDs, Sequence):
            return self.indices(IDs)
        else: # pragma: no cover
            raise TypeError("only strings, chemicals, sequences, and ellipsis are valid index keys")    
    
    def _get_index_and_kind(self, key):
        index_cache = self._index_cache
        try:
            if key.__hash__ is None: key = tuple(key)
            return index_cache[key]
        except KeyError:
            isa = isinstance
            kind = 0 # [int] Kind of index: 0 - normal, 1 - chemical group, 2 - nested chemical group
            if isa(key, str):
                index = self.index(key)
                if isa(index, list): kind = 1 
            elif isa(key, tuple):
                index = self.indices(key)
                for i in index:
                    if isa(i, list): 
                        kind = 2
                        break
            elif key is ...:
                index = slice(None)
            else: # pragma: no cover
                raise TypeError("only strings, sequences of strings, and ellipsis are valid index keys")
            index_cache[key] = index, kind
            if len(index_cache) > 100: index_cache.pop(index_cache.__iter__().__next__())
        except TypeError:
            raise TypeError("only strings, sequences of strings, and ellipsis are valid index keys")
        return index, kind
    
    def __len__(self):
        return self.size
    
    def __contains__(self, chemical):
        if isinstance(chemical, str):
            return chemical in self.__dict__
        elif isinstance(chemical, Chemical):
            return chemical in self.tuple
        else: # pragma: no cover
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
        >>> chemicals.get_vle_indices(data[data!=0])
        [1, 2]
        
        """
        return [i for i in self._vle_index if i in nonzeros]
    
    def get_lle_indices(self, nonzeros):
        """
        Return indices of species in liquid-liquid equilibrium given an array
        dictating whether or not the chemicals are present.
        
        Examples
        --------
        >>> from thermosteam import CompiledChemicals
        >>> chemicals = CompiledChemicals(['Water', 'Methanol', 'Ethanol'])
        >>> data = chemicals.kwarray(dict(Water=2., Ethanol=1.))
        >>> chemicals.get_lle_indices(data[data!=0])
        [1, 2]
        
        """
        return [i for i in self._lle_index if i in nonzeros]
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(self.IDs)}])"
