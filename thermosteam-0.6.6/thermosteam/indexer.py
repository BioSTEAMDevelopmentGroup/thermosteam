# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:41:50 2019

@author: yoelr
"""

from .base import UnitsOfMeasure
from .utils import repr_IDs_data, repr_couples, chemicals_user
from ._settings import settings
from .exceptions import UndefinedPhase, UndefinedChemical
from ._phase import Phase, LockedPhase, NoPhase
from free_properties import PropertyFactory, property_array
import numpy as np

__all__ = ('ChemicalIndexer',
           'MaterialIndexer',
           'ChemicalMolarFlowIndexer', 
           'MolarFlowIndexer',
           'ChemicalMassFlowIndexer', 
           'MassFlowIndexer',
           'ChemicalVolumetricFlowIndexer',
           'VolumetricFlowIndexer',
           'MassFlowProperty',
           'VolumetricFlowProperty')

# %% Utilities

isa = isinstance
_new = object.__new__

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]

class ChemicalIndex:
    __slots__ = ('value')
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"{type(self).__name__}({self.value})"


# %% Abstract indexer
    
class Indexer:
    """Abstract class for fast indexing."""
    __slots__ = ('_data',)
    units = None
    
    def copy(self):
        new = self._copy_without_data()
        new._data = self._data.copy()
        return new
    __copy__ = copy
    
    def get_data(self, units, *index):
        length = len(index)
        factor = self.units.conversion_factor(units)
        if length == 0:
            return factor * self._data
        elif length == 1:
            return factor * self[index[0]]
        else:
            return factor * self[index]
    
    def set_data(self, data, units, *index):
        length = len(index)
        data = np.asarray(data, dtype=float)
        scaled_data = data / self.units.conversion_factor(units)
        if length == 0:
            self._data[:] = scaled_data
        elif length == 1:
            self[index[0]] = scaled_data
        else:
            self[index] = scaled_data
    
    @property
    def data(self):
        return self._data


# %% Phase data

@chemicals_user
class ChemicalIndexer(Indexer):
    """Create a ChemicalIndexer that can index a single-phase, 1d-array given chemical IDs.
    
    Parameters
    ----------
    phase : [str or PhaseContainer] {'s', 'l', 'g', 'S', 'L', 'G'}
        Phase of data.
    units : str
        Units of measure of input data. 
    chemicals : Chemicals
        Required to define the chemicals that are present.
    **ID_data : float
        ID-value pairs
    
    Notes
    -----
    A ChemicalIndexer does not have any units defined. To use units of measure, use the 
    `ChemicalMolarIndexer`, `ChemicalMassIndexer`, or `ChemicalVolumetricIndexer`.
    
    """
    __slots__ = ('_chemicals', '_phase', '_data_cache')
    
    def __new__(cls, phase, units=None, chemicals=None, **ID_data):
        if ID_data:
            self = cls.from_data(chemicals.kwarray(ID_data), phase, chemicals)
            if units: self.set_data(self._data, units)
        else:
            self = cls.blank(phase, chemicals)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phase, self._chemicals, False)
    
    def __getitem__(self, key):
        return self._data[self.get_index(key)]
    
    def __setitem__(self, key, data):
        self._data[self.get_index(key)] = data
    
    @property
    def get_index(self):
        return self._chemicals.get_index
    
    def to_material_indexer(self, phases=()):
        material_array = self._MaterialIndexer.blank(phases, self._chemicals)
        material_array[self.phase] = self._data
        return material_array
    
    def copy_like(self, other):
        assert self._chemicals is other._chemicals, "chemicals must match"
        self._data[:] = other._data
        self.phase = other.phase
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._chemicals = self._chemicals
        new._phase = self._phase.copy()
        new._data_cache = {}
        return new
    
    @classmethod
    def blank(cls, phase, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._data = np.zeros(self._chemicals.size, float)
        self._phase = Phase.convert(phase)
        self._data_cache = {}
        return self
    
    @classmethod
    def from_data(cls, data, phase, chemicals=None, check_data=True):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._phase = Phase.convert(phase)
        if check_data:
            assert data.ndim == 1, 'material data must be a 1d numpy array'
            assert data.size == self._chemicals.size, ('size of material data must be equal to '
                                                       'size of chemicals')
        self._data = data
        self._data_cache = {}
        return self
    
    @property
    def phase(self):
        return self._phase.phase
    @phase.setter
    def phase(self, phase):
        self._phase.phase = phase
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        phase = f"phase={repr(self.phase)}"
        if tab:
            dlim = ",\n" + tab
            phase = '\n' + tab + phase
        else:
            dlim = ", "
        ID_data = repr_IDs_data(self._chemicals.IDs, self._data, dlim)
        return f"{type(self).__name__}({phase}{ID_data})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
        from thermosteam import Stream
        IDs = self.chemicals.IDs
        data = self.data
        IDs, data = nonzeros(IDs, data)
        N_IDs = len(IDs)
        if N_IDs == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n"
        else:
            basic_info = f"{type(self).__name__}:\n"
        beginning = f' ({self.phase}) ' if self.phase else " "
        new_line = '\n' + len(beginning) * ' '
        data_info = ''
        lengths = [len(i) for i in IDs]
        maxlen = max(lengths) + 1
        N_max = N or Stream.display_units.N
        too_many_chemicals = N_IDs > N_max
        N = N_max if too_many_chemicals else N_IDs
        for i in range(N):
            spaces = ' ' * (maxlen - lengths[i])
            if i != 0:
                data_info += new_line
            data_info += IDs[i] + spaces + f' {data[i]:.4g}'
        if too_many_chemicals:
            data_info += new_line + '...'
        return (basic_info
              + beginning
              + data_info)

    def show(self, N=None):
        """Print all specifications.
        
        Parameters
        ----------
        N: int, optional
            Number of compounds to display.
        
        """
        print(self._info(N))
    _ipython_display_ = show
      
@chemicals_user
class MaterialIndexer(Indexer):
    """Create a MaterialIndexer that can index a multi-phase, 2d-array given the phase and chemical IDs.
    
    Parameters
    ----------
    phases : tuple['s', 'l', 'g', 'S', 'L', 'G']
        Phases of data rows.
    units : str
        Units of measure of input data. 
    chemicals : Chemicals
        Required to define the chemicals that are present.
    **phase_data : tuple[str, float]
        phase-(ID, value) pairs
    
    Notes
    -----
    A MaterialIndexer does not have any units defined. To use units of measure, use the 
    `MolarIndexer`, `MassIndexer`, or `VolumetricIndexer`.
    
    """
    __slots__ = ('_chemicals', '_phases', '_phase_index',
                 '_index_cache', '_data_cache')
    _index_caches = {}
    _phase_index_cache = {}
    _ChemicalIndexer = ChemicalIndexer
    
    def __new__(cls, phases=None, units=None, chemicals=None, **phase_data):
        self = cls.blank(phases or phase_data, chemicals)
        if phase_data:
            data = self._data
            get_index = self._chemicals.get_index
            phase_index = self._get_phase_index
            for phase, ID_data in phase_data.items():
                IDs, row = zip(*ID_data)
                data[phase_index(phase), get_index(IDs)] = row
            if units:
                self.set_data(data, units)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases, self._chemicals, False)
    
    def copy_like(self, other):
        if isa(other, ChemicalIndexer):
            self._data[:] = 0
            self[other.phase] = other._data
        else:
            self._data[:] = other._data
    
    def _set_phases(self, phases):
        self._phases = phases = tuple(sorted(phases))
        cache = self._phase_index_cache
        if phases in cache:
            self._phase_index = cache[phases]
        else:
            self._phase_index = cache[phases] = {j:i for i,j in enumerate(phases)}
            
    def _set_cache(self):
        caches = self._index_caches
        key = self._phases, self._chemicals
        try:
            self._index_cache = caches[key]
        except KeyError:
            self._index_cache = caches[key] = {}
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._chemicals = self._chemicals
        new._phases = self._phases
        new._phase_index = self._phase_index
        new._index_cache = self._index_cache
        new._data_cache = {}
        return new
    
    @classmethod
    def blank(cls, phases, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        shape = (len(self._phases), self._chemicals.size)
        self._data = np.zeros(shape, float)
        self._data_cache = {}
        return self
    
    @classmethod
    def from_data(cls, data, phases, chemicals=None, check_data=True):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        if check_data:
            assert data.ndim == 2, ('material data must be an 2d numpy array')
            M_phases = len(self._phases)
            N_chemicals = self._chemicals.size
            M, N = data.shape
            assert M == M_phases, ('number of phases must be equal to '
                                   'the number of material data rows')
            assert N == N_chemicals, ('size of chemicals '
                                      'must be equal to '
                                      'number of material data columns')
        self._data = data
        self._data_cache = {}
        return self
    
    @property
    def phases(self):
        return self._phases
    
    def to_chemical_indexer(self, phase=NoPhase):
        return self._ChemicalIndexer.from_data(self._data.sum(0), phase, self._chemicals, False)
    
    def get_phase(self, phase):
        return self._ChemicalIndexer.from_data(self._data[self._get_phase_index(phase)],
                                               LockedPhase(phase), self._chemicals, False)
    
    def __getitem__(self, key):
        index = self.get_index(key)
        if isa(index, ChemicalIndex):
            values = self._data[..., index.value].sum(0)
        else:
            values = self._data[index]
        return values
    
    def __setitem__(self, key, data):
        index = self.get_index(key)
        if isa(index, ChemicalIndex):
            raise IndexError("multiple phases present; must include phase index "
                             "to set chemical data")                
        self._data[index] = data
    
    def get_index(self, key):
        cache = self._index_cache
        try: 
            index = cache[key]
        except KeyError:
            try:
                index = self._chemicals.get_index(key)
            except UndefinedChemical:
                index = self._get_index(key)
            else:
                index = ChemicalIndex(index)
            cache[key] = index
        except TypeError:
            raise TypeError("only strings, tuples, and ellipsis are valid index keys")
        return index
    
    def _get_index(self, phase_IDs):
        if isa(phase_IDs, str):
            index = self._get_phase_index(phase_IDs)
        elif phase_IDs is ...:
            index = phase_IDs 
        else:
            try:
                phase, IDs = phase_IDs
            except:
                raise IndexError("index by [phase, IDs] where phase is a "
                                 "(str, ellipsis, or missing), and IDs is a "
                                 "(str, tuple(str), ellipisis, or missing)")
            if isa(IDs, (str, tuple)):
                IDs_index = self._chemicals.get_index(IDs)
            elif IDs is ...:
                IDs_index = IDs
            else:
                raise TypeError("only strings, tuples, and ellipsis are valid index keys")
            if isa(phase, str):
                index = (self._get_phase_index(phase), IDs_index)
            elif phase is ...:
                index = (phase, IDs_index)
            else:
                raise IndexError("index by [phase, IDs] where phase is a "
                                 "(str, ellipsis, or missing), and IDs is a "
                                 "(str, tuple(str), ellipisis, or missing)")
        return index
    
    def _get_phase_index(self, phase):
        try:
            return self._phase_index[phase]
        except:
            raise UndefinedPhase(phase)
    
    def iter_data(self):
        """Iterate over phase-data pairs."""
        return zip(self._phases, self._data)
    
    def iter_composition(self):
        """Iterate over phase-composition pairs."""
        array = self._data
        total = array.sum() or 1.
        return zip(self._phases, array/total)
    
    def __format__(self, tabs="1"):
        IDs = self._chemicals.IDs
        phase_data = []
        for phase, data in self.iter_data():
            ID_data = repr_couples(", ", IDs, data)
            if ID_data:
                phase_data.append(f"{phase}=[{ID_data}]")
        tabs = int(tabs)
        if tabs:
            tab = tabs*4*" "
            dlim = ",\n" + tab 
        else:
            dlim = ", "
        phase_data = dlim.join(phase_data)
        if self.data.sum(1).all():
            phases = ""
            if phase_data:
                phase_data = "\n" + tab + phase_data
        else:
            phases = f'phases={self.phases}'
            if phase_data:
                phase_data = dlim + phase_data
        return f"{type(self).__name__}({phases}{phase_data})"
    
    def __repr__(self):
        return self.__format__("1")
    
    def _info(self, N):
        """Return string with all specifications."""
        from thermosteam import Stream
        N_max = N or Stream.display_units.N
        IDs = self.chemicals.IDs
        index, = np.where(self.data.sum(0) != 0)
        len_ = len(index)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n"
        else:
            basic_info = f"{type(self).__name__}:\n"
        all_IDs = tuple([IDs[i] for i in index])

        # Length of chemical column
        all_lengths = [len(i) for i in IDs]
        maxlen = max(all_lengths + [8])

        # Set up chemical data for all phases
        phases_data_info = ''
        for phase in self.phases:
            phase_data = self[phase, all_IDs]
            IDs, data = nonzeros(all_IDs, phase_data)
            if not IDs: continue
        
            # Get basic structure for phase data
            beginning = f' ({phase}) '
            new_line = '\n' + len(beginning) * ' '

            # Set chemical data
            data_info = ''
            N_IDs = len(data)
            too_many_chemicals = N_IDs > N_max
            N = N_max if too_many_chemicals else N_IDs
            lengths = [len(i) for i in IDs]
            for i in range(N):
                spaces = ' ' * (maxlen - lengths[i])
                if i: data_info += new_line
                data_info += f'{IDs[i]} ' + spaces + f' {data[i]:.4g}'
            if too_many_chemicals: data += new_line + '...'
            # Put it together
            phases_data_info += beginning + data_info + '\n'
            
        return basic_info + phases_data_info.rstrip('\n')
    show = ChemicalIndexer.show
    _ipython_display_ = show
    
def _replace_indexer_doc(Indexer, Parent):
    doc = Parent.__doc__
    doc = doc[:doc.index("Notes")]
    Indexer.__doc__ = doc.replace(Parent.__name__, Indexer.__name__)
    
def _new_Indexer(name, units):
    ChemicalIndexerSubclass = type('Chemical' + name + 'Indexer', (ChemicalIndexer,), {})
    MaterialIndexerSubclass = type(name + 'Indexer', (MaterialIndexer,), {})
    
    ChemicalIndexerSubclass.__slots__ = \
    MaterialIndexerSubclass.__slots__ = ()
    
    ChemicalIndexerSubclass.units = \
    MaterialIndexerSubclass.units = UnitsOfMeasure(units)
    
    MaterialIndexerSubclass._ChemicalIndexer = ChemicalIndexerSubclass
    ChemicalIndexerSubclass._MaterialIndexer = MaterialIndexerSubclass
    
    _replace_indexer_doc(ChemicalIndexerSubclass, ChemicalIndexer)
    _replace_indexer_doc(MaterialIndexerSubclass, MaterialIndexer)
    
    return ChemicalIndexerSubclass, MaterialIndexerSubclass

ChemicalIndexer._MaterialIndexer = MaterialIndexer
ChemicalMolarFlowIndexer, MolarFlowIndexer = _new_Indexer('MolarFlow', 'kmol/hr')
ChemicalMassFlowIndexer, MassFlowIndexer = _new_Indexer('MassFlow', 'kg/hr')
ChemicalVolumetricFlowIndexer, VolumetricFlowIndexer = _new_Indexer('VolumetricFlow', 'm^3/hr')

# %% Mass flow properties

@PropertyFactory(slots=('name', 'mol', 'index', 'MW'))
def MassFlowProperty(self):
    """Mass flow (kg/hr)."""
    return self.mol[self.index] * self.MW
    
@MassFlowProperty.setter
def MassFlowProperty(self, value):
    self.mol[self.index] = value/self.MW

def by_mass(self):
    """Return a ChemicalMassFlowIndexer that references this object's molar data."""
    try:
        mass = self._data_cache['mass']
    except:
        chemicals = self.chemicals
        mol = self.data
        mass = np.zeros_like(mol, dtype=object)
        for i, chem in enumerate(chemicals):
            mass[i] = MassFlowProperty(chem.ID, mol, i, chem.MW)
        self._data_cache['mass'] = mass = ChemicalMassFlowIndexer.from_data(
                                                        property_array(mass),
                                                        self._phase, chemicals,
                                                        False)
    return mass
ChemicalMolarFlowIndexer.by_mass = by_mass

def by_mass(self):
    """Return a MassFlowIndexer that references this object's molar data."""
    try:
        mass = self._data_cache['mass']
    except:
        phases = self.phases
        chemicals = self.chemicals
        mol = self.data
        mass = np.zeros_like(mol, dtype=object)
        for i, phase in enumerate(phases):
            for j, chem in enumerate(chemicals):
                index = (i, j)
                mass[index] = MassFlowProperty(chem.ID, mol, index, chem.MW)
        self._data_cache['mass'] = mass = MassFlowIndexer.from_data(
                                                        property_array(mass),
                                                        phases, chemicals,
                                                        False)
    return mass
MolarFlowIndexer.by_mass = by_mass; del by_mass


# %% Volumetric flow properties

@PropertyFactory(slots=('name', 'mol', 'index', 'V',
                        'TP', 'phase', 'phase_container'))
def VolumetricFlowProperty(self):
    """Volumetric flow (m^3/hr)."""
    f_mol = self.mol[self.index] 
    phase = self.phase or self.phase_container.phase
    V = getattr(self.V, phase) if hasattr(self.V, phase) else self.V
    return 1000. * f_mol * V(*self.TP) if f_mol else 0
    
@VolumetricFlowProperty.setter
def VolumetricFlowProperty(self, value):
    if value:
        phase = self.phase or self.phase_container.phase
        V = getattr(self.V, phase) if hasattr(self.V, phase) else self.V
        self.mol[self.index] = value / V(*self.TP) / 1000.
    else:
        self.mol[self.index] = 0

def by_volume(self, TP):
    """Return a ChemicalVolumetricFlowIndexer that references this object's molar data.
    
    Parameters
    ----------
    TP : ThermalCondition
    
    """
    try:
        vol = self._data_cache[TP]
    except:
        chemicals = self.chemicals
        mol = self.data
        vol = np.zeros_like(mol, dtype=object)
        for i, chem in enumerate(chemicals):
            vol[i] = VolumetricFlowProperty(chem.ID, mol, i, chem.V,
                                            TP, None, self._phase)
        self._data_cache[TP] = \
        vol = ChemicalVolumetricFlowIndexer.from_data(property_array(vol),
                                                      self._phase, chemicals,
                                                      False)
    return vol
ChemicalMolarFlowIndexer.by_volume = by_volume
	
def by_volume(self, TP):
    """Return a VolumetricFlowIndexer that references this object's molar data.
    
    Parameters
    ----------
    TP : ThermalCondition
    
    """
    try:
        vol = self._data_cache[TP]
    except:
        phases = self.phases
        chemicals = self.chemicals
        mol = self.data
        vol = np.zeros_like(mol, dtype=object)
        for i, phase in enumerate(phases):
            for j, chem in enumerate(chemicals):
                index = i, j
                phase_name = settings._phase_names[phase]
                vol[index] = VolumetricFlowProperty(f"{phase_name}{chem.ID}", 
                                                    mol, index, chem.V, TP, phase)
        self._data_cache[TP] = \
        vol = VolumetricFlowIndexer.from_data(property_array(vol),
                                              phases, chemicals,
                                              False)
    return vol
MolarFlowIndexer.by_volume = by_volume; del by_volume
del PropertyFactory
