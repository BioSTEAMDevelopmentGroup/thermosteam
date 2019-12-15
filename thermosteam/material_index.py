# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:41:50 2019

@author: yoelr
"""

from .base import Units
from .utils import repr_IDs_data, repr_couples
from .settings import settings
from .exceptions import UndefinedPhase
from .phase_container import phase_container
from free_properties import PropertyFactory, property_array
import numpy as np

__all__ = ('ChemicalIndex',
           'PhaseIndex',
           'MaterialIndex',
           'ChemicalMolarFlowIndex', 
           'PhaseMolarFlowIndex',
           'MolarFlowIndex',
           'ChemicalMassFlowIndex', 
           'PhaseMassFlowIndex',
           'MassFlowIndex',
           'ChemicalVolumetricFlowIndex',
           'PhaseVolumetricFlowIndex',
           'VolumetricFlowIndex',
           'MassFlowProperty',
           'VolumetricFlowProperty')

# %% Utilities

isa = isinstance
_new = object.__new__

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]

# %% Abstract data emulator
    
class IndexEmulator:
    __slots__ = ('_data', '_index_cache')
    _phase_index_cache = {}
    chemicals = None
    units = Units()
    
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
        data = data._data if isa(data, IndexEmulator) else np.asarray(data, dtype=float)
        scaled_data = data / self.units.conversion_factor(units)
        if length == 0:
            self._data[:] = scaled_data
        elif length == 1:
            self[index[0]] = scaled_data
        else:
            self[index] = scaled_data
    
    def get_index(self, key):
        cache = self._index_cache
        try: 
            index = cache[key]
        except KeyError: 
            cache[key] = index = self._get_index(key)
        except TypeError:
            raise IndexError(f"only strings, tuples, and ellipsis are valid indices")
        return index
    
    def __getitem__(self, key):
        return self._data[self.get_index(key)]
    
    def __setitem__(self, key, data):
        self._data[self.get_index(key)] = data._data if isa(data, IndexEmulator) else data
    
    @property
    def data(self):
        return self._data


# %% Phase data

class ChemicalIndex(IndexEmulator):
    __slots__ = ('_chemicals', '_phase', '_data_cache')
    _index_caches = {}
    
    def __new__(cls, phase, units=None, chemicals=None, **IDdata):
        if IDdata:
            chemicals = settings.get_chemicals(chemicals)
            self = cls.from_data(chemicals.kwarray(IDdata), phase, chemicals)
            if units: self.set_data(self._data, units)
        else:
            self = cls.blank(phase, chemicals)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._chemicals)
    
    def _set_chemicals(self, chemicals):
        self._chemicals = chemicals = settings.get_chemicals(chemicals)
    
    def _set_cache(self):
        caches = self._index_caches
        try:
            self._index_cache = caches[self._chemicals]
        except KeyError:
            self._index_cache = caches[self._chemicals] = {}
        
    def to_phase_index(self, phases=()):
        phase_array = self._PhaseIndex.blank(phases)
        phase_array[self.phase] = self._data.sum()
        return phase_array
    
    def to_material_index(self, phases=()):
        material_array = self._MaterialIndex.blank(phases, self._chemicals)
        material_array[self.phase] = self._data
        return material_array
    
    def copy_like(self, other):
        self._data[:] = other._data
        self.phase = other.phase
    
    @property
    def composition(self):
        array = self._data
        total = array.sum()
        composition = array/total if total else array.copy()
        return ChemicalIndex.from_data(composition, self._phase, self._chemicals)
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._chemicals = self._chemicals
        new._index_cache = self._index_cache
        new._phase = self._phase
        new._data_cache = {}
        return new
    
    @classmethod
    def blank(cls, phase, chemicals=None):
        self = _new(cls)
        self._set_chemicals(chemicals)
        self._set_cache()
        self._data = np.zeros(self._chemicals.size, float)
        self._phase = phase_container(phase)
        self._data_cache = {}
        return self
    
    @classmethod
    def from_data(cls, data, phase, chemicals=None):
        self = _new(cls)
        self._set_chemicals(chemicals)
        self._set_cache()
        self._phase = phase_container(phase)
        if settings._debug:
            assert isa(data, np.ndarray) and data.ndim == 1, (
                                                    'data must be a 1d numpy array')
            assert data.size == self._chemicals.size, ('size of data must be equal to '
                                                       'size of chemicals')
        self._data = data
        self._data_cache = {}
        return self
    
    @property
    def chemicals(self):
        return self._chemicals
    
    @property
    def phase(self):
        return self._phase[0]
    @phase.setter
    def phase(self, phase):
        try: self._phase[0] = phase
        except: raise AttributeError("can't set phase")
    
    def _get_index(self, IDs):
        if isa(IDs, str):
            return self._chemicals.index(IDs)
        elif isa(IDs, tuple):
            return self._chemicals.indices(IDs)
        elif IDs == ...:
            return IDs
        else:
            raise IndexError(f"only strings, tuples, and ellipsis are valid indices")
    
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
        IDdata = repr_IDs_data(self._chemicals.IDs, self._data, dlim)
        return f"{type(self).__name__}({phase}{IDdata})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
        IDs = self.chemicals.IDs
        data = self.data
        IDs, data = nonzeros(IDs, data)
        len_ = len(data)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n"
        else:
            basic_info = f"{type(self).__name__}:\n"
        beginning = f' ({self.phase}) ' if self.phase else " "
        new_line_spaces = len(beginning) * ' '
        data_info = ''
        lengths = [len(i) for i in IDs]
        maxlen = max(lengths) + 1
        _N = N - 1
        for i in range(len_-1):
            spaces = ' ' * (maxlen - lengths[i])
            if i == _N:
                data_info += '...\n' + new_line_spaces
                break
            data_info += IDs[i] + spaces + f' {data[i]:.4g}\n' + new_line_spaces
        spaces = ' ' * (maxlen - lengths[len_-1])
        data_info += IDs[len_-1] + spaces + f' {data[len_-1]:.4g}'
        return (basic_info
              + beginning
              + data_info)

    def show(self, N=5):
        """Print all specifications.
        
        Parameters
        ----------
        N: int, optional
            Number of compounds to display.
        
        """
        print(self._info(N))
    _ipython_display_ = show
      
    
class PhaseIndex(IndexEmulator):
    __slots__ = ('_phases', '_phase_index')
    _index_caches = {}
    
    def __new__(cls, phases=None, units=None, **phase_data):
        self = cls.blank(phases or phase_data)
        if phase_data:
            phases, data = zip(*phase_data.items())
            phases = tuple(phases)
            if units:
                self.set_data(data, units, phases)
            else:
                self[phases] = data
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases)
    
    def copy_like(self, other):
        self._data[:] = other._data
    
    @property
    def phases(self):
        return self._phases
    
    @property
    def phase_split(self):
        array = self._data
        total = array.sum()
        phase_split = array/total if total else array.copy()
        return PhaseIndex.from_data(phase_split, self._phases)
    
    def _set_phases(self, phases):
        self._phases = phases = tuple(sorted(phases))
        cache = self._phase_index_cache
        if phases in cache:
            self._phase_index = cache[phases]
        else:
            self._phase_index = cache[phases] = {j:i for i,j in enumerate(phases)}
    
    def _set_cache(self):
        caches = self._index_caches
        try:
            self._index_cache = caches[self._phases]
        except KeyError:
            self._index_cache = caches[self._phases] = {}
    
    def _get_phase_index(self, phase):
        try:
            return self._phase_index[phase]
        except:
            raise UndefinedPhase(phase)
    
    def _get_phase_indices(self, phases):
        index = self._phase_index
        try:
            return [index[i] for i in phases]
        except:
            for i in phases:
                if i not in index:
                    raise UndefinedPhase(i)
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._phases = self._phases
        new._index_cache = self._index_cache
        return new
    
    @classmethod
    def blank(cls, phases):
        self = _new(cls)
        self._set_phases(phases)
        self._set_cache()
        self._data = np.zeros(len(self._phases), float)
        return self
    
    @classmethod
    def from_data(cls, data, phases):
        self = _new(cls)
        self._set_phases(phases)
        self._set_cache()
        if settings._debug:
            assert isa(data, np.ndarray) and data.ndim == 1, (
                                                    'data must be an 1d numpy array')
            assert data.size == len(self._phases), ('size of data must be equal to '
                                                    'number of phases')
        self._data = data
        return self
    
    def _get_index(self, phases):
        if isa(phases, str):
            return self._get_phase_index(phases)
        elif isa(phases, tuple):
            return self._get_phase_indices(phases)
        elif phases == ...:
            return phases
        else:
            raise IndexError(f"only strings, tuples, and ellipsis are valid indices")
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        if tab:
            dlim = ",\n" + tab
            start = '\n' + tab
        else:
            dlim = ', '
            start = ""
        phase_data = repr_IDs_data(self.phases, self.data, start, dlim)
        return f"{type(self).__name__}({phase_data})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
        phases = self.phases
        data = self.data
        phases, data = nonzeros(phases, data)
        phases = [f'({i})' for i in phases]
        len_ = len(data)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n "
        else:
            basic_info = f"{type(self).__name__}:\n "
        new_line_spaces = ' '        
        data_info = ''
        lengths = [len(i) for i in phases]
        maxlen = max(lengths) + 1
        _N = N - 1
        for i in range(len_-1):
            spaces = ' ' * (maxlen - lengths[i])
            if i == _N:
                data_info += '...\n' + new_line_spaces
                break
            data_info += phases[i] + spaces + f' {data[i]:.4g}\n' + new_line_spaces
        spaces = ' ' * (maxlen - lengths[len_-1])
        data_info += phases[len_-1] + spaces + f' {data[len_-1]:.4g}'
        return (basic_info
              + data_info)

    def show(self, N=5):
        """Print all specifications.
        
        Parameters
        ----------
        N: int, optional
            Number of compounds to display.
        
        """
        print(self._info(N))
    _ipython_display_ = show
            

class MaterialIndex(IndexEmulator):
    __slots__ = ('_chemicals', '_phases', '_phase_index', '_data_cache')
    _index_caches = {}
    _ChemicalIndex = ChemicalIndex
    _PhaseIndex = PhaseIndex
    
    def __new__(cls, phases=None, units=None, chemicals=None, **phase_data):
        self = cls.blank(phases or phase_data, chemicals)
        if phase_data:
            data = self._data
            chemical_indices = self._chemicals.indices
            phase_index = self._get_phase_index
            for phase, IDdata in phase_data.items():
                IDs, row = zip(*IDdata)
                data[phase_index(phase), chemical_indices(IDs)] = row
            if units:
                self.set_data(data, units)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases, self._chemicals)
    
    def copy_like(self, other):
        self._data[:] = other._data
    
    @property
    def composition(self):
        array = self._data
        chemical_array = array.sum(0)
        total = chemical_array.sum()
        composition = chemical_array/total if total else chemical_array
        return ChemicalIndex.from_data(composition, (None,), self._chemicals)
    
    @property
    def phase_split(self):
        array = self._data
        phase_array = array.sum(1)
        total = array.sum()
        phase_split = phase_array/total if total else phase_array
        return PhaseIndex.from_data(phase_split, self._phases)
    
    @property
    def composition_by_phase(self):
        array = self._data
        phase_array = array.sum(1, keepdims=True)
        phase_array[phase_array == 0] = 1.
        return MaterialIndex.from_data(array/phase_array, self._phases, self._chemicals)
    
    _set_chemicals = ChemicalIndex._set_chemicals
    _set_phases = PhaseIndex._set_phases
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
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        shape = (len(self._phases), self._chemicals.size)
        self._data = np.zeros(shape, float)
        self._data_cache = {}
        return self
    
    @classmethod
    def from_data(cls, data, phases, chemicals=None):
        self = _new(cls)
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        if settings._debug:
            assert isa(data, np.ndarray) and data.ndim == 2, (
                                                    'data must be an 2d numpy array')
            M_phases = len(self._phases)
            N_chemicals = self._chemicals.size
            M, N = data.shape
            assert M == M_phases, ('number of phases must be equal to '
                                   'the number of data rows')
            assert N == N_chemicals, ('size of chemicals '
                                      'must be equal to '
                                      'number of data columns')
        self._data = data
        self._data_cache = {}
        return self
    
    phases =  PhaseIndex.phases
    chemicals = ChemicalIndex.chemicals
    
    def to_phase_index(self):
        return self._PhaseIndex.from_data(self._data.sum(1), self._phases)
    
    def to_chemical_index(self, phase=(None,)):
        return self._ChemicalIndex.from_data(self._data.sum(0), phase, self._chemicals)
    
    def get_phase(self, phase):
        return self._ChemicalIndex.from_data(self._data[self._get_phase_index(phase)],
                                             (phase,), self._chemicals)
    
    def _get_index(self, phase_IDs):
        if isa(phase_IDs, str):
            index = self._get_phase_index(phase_IDs)
        elif phase_IDs == ...:
            index = phase_IDs 
        else:
            try:
                phase, IDs = phase_IDs
            except:
                raise IndexError(f"please index with <{type(self).__name__}>[phase, IDs] "
                                  "where phase is a (str, or ellipsis), "
                                  "and IDs is a (str, tuple(str), ellipisis, or missing)")
            if isa(IDs, str):
                IDs_index = self._chemicals.index(IDs)
            elif isa(IDs, tuple):
                IDs_index = self._chemicals.indices(IDs)
            elif IDs == ...:
                IDs_index = IDs
            else:
                raise IndexError(f"only strings, tuples, and ellipsis are valid indices")
            if isa(phase, str):
                index = (self._get_phase_index(phase), IDs_index)
            elif phase == ...:
                index = (phase, IDs_index)
            else:
                raise IndexError(f"please index with <{type(self).__name__}>[phase, IDs] "
                                  "where phase is a (str, or ellipsis), "
                                  "and IDs is a (str, tuple(str), ellipisis, or missing)")
        return index
    
    _get_phase_index = PhaseIndex._get_phase_index
    
    def iter_phase_data(self):
        return zip(self._phases, self._data)
    
    def iter_phase_composition(self):
        array = self._data
        phase_array = array.sum(1, keepdims=True)
        phase_array[phase_array == 0] = 1.
        return zip(self._phases, array/phase_array)
    
    def __format__(self, tabs="1"):
        IDs = self._chemicals.IDs
        phase_data = []
        for phase, data in self.iter_phase_data():
            IDdata = repr_couples(", ", IDs, data)
            if IDdata:
                phase_data.append(f"{phase}=[{IDdata}]")
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
        phases_flowrates_info = ''
        for phase in self.phases:
            phase_data = self[phase, all_IDs]
            IDs, data = nonzeros(all_IDs, phase_data)
            if not IDs: continue
        
            # Get basic structure for phase data
            beginning = f' ({phase}) '
            new_line_spaces = len(beginning) * ' '

            # Set chemical data
            flowrates = ''
            l = len(data)
            lengths = [len(i) for i in IDs]
            _N = N - 1
            for i in range(l-1):
                spaces = ' ' * (maxlen - lengths[i])
                if i == _N:
                    flowrates += '...\n' + new_line_spaces
                    break
                flowrates += f'{IDs[i]} ' + spaces + \
                    f' {data[i]:.4g}\n' + new_line_spaces
            spaces = ' ' * (maxlen - lengths[l-1])
            flowrates += (f'{IDs[l-1]} ' + spaces
                          + f' {data[l-1]:.4g}')

            # Put it together
            phases_flowrates_info += beginning + flowrates + '\n'
            
        return basic_info + phases_flowrates_info[:-1]
    show = ChemicalIndex.show
    _ipython_display_ = show
    
def new_Index(name, units):
    ChemicalIndexSubclass = type('Chemical' + name + 'Index', (ChemicalIndex,), {})
    PhaseIndexSubclass = type('Phase' + name + 'Index', (PhaseIndex,), {})
    MaterialIndexSubclass = type(name + 'Index', (MaterialIndex,), {})
    
    ChemicalIndexSubclass.__slots__ = \
    PhaseIndexSubclass.__slots__ = \
    MaterialIndexSubclass.__slots__ = ()
    
    ChemicalIndexSubclass.units = \
    PhaseIndexSubclass.units = \
    MaterialIndexSubclass.units = Units(units)
    
    PhaseIndexSubclass._ChemicalIndex = \
    MaterialIndexSubclass._ChemicalIndex = ChemicalIndexSubclass
    
    MaterialIndexSubclass._PhaseIndex = \
    ChemicalIndexSubclass._PhaseIndex = PhaseIndexSubclass
    
    PhaseIndexSubclass._MaterialIndex = \
    ChemicalIndexSubclass._MaterialIndex = MaterialIndexSubclass
    
    return ChemicalIndexSubclass, PhaseIndexSubclass, MaterialIndexSubclass

ChemicalIndex._MaterialIndex = MaterialIndex
ChemicalIndex._PhaseIndex = PhaseIndex
PhaseIndex._ChemicalIndex = ChemicalIndex
PhaseIndex._MaterialIndex = MaterialIndex    
ChemicalMolarFlowIndex, PhaseMolarFlowIndex, MolarFlowIndex = new_Index('MolarFlow', 'kmol/hr')
ChemicalMolarFlowIndex.__slots__ = MolarFlowIndex.__slots__ = ('_mass_flow', '_volumetric_flow')
ChemicalMassFlowIndex, PhaseMassFlowIndex, MassFlowIndex = new_Index('MassFlow', 'kg/hr')
ChemicalVolumetricFlowIndex, PhaseVolumetricFlowIndex, VolumetricFlowIndex = new_Index('VolumetricFlow', 'm^3/hr')


# %% Mass flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'MW'))
def MassFlowProperty(self):
    """Mass flow (kg/hr)."""
    return self.molar_flow[self.index] * self.MW
    
@MassFlowProperty.setter
def MassFlowProperty(self, value):
    self.molar_flow[self.index] = value/self.MW

def by_mass(self):
    try:
        mass_flow = self._data_cache['mass_flow']
    except:
        chemicals = self.chemicals
        molar_flow = self.data
        mass_flow = np.zeros_like(molar_flow, dtype=object)
        for i, chem in enumerate(chemicals):
            mass_flow[i] = MassFlowProperty(chem.ID, molar_flow, i, chem.MW)
        self._data_cache['mass_flow'] = mass_flow = ChemicalMassFlowIndex.from_data(
                                                        property_array(mass_flow),
                                                        self._phase, chemicals)
    return mass_flow
ChemicalMolarFlowIndex.by_mass = by_mass

def by_mass(self):
    try:
        mass_flow = self._data_cache['mass_flow']
    except:
        phases = self.phases
        chemicals = self.chemicals
        molar_flow = self.data
        mass_flow = np.zeros_like(molar_flow, dtype=object)
        for i, phase in enumerate(phases):
            for j, chem in enumerate(chemicals):
                index = (i, j)
                mass_flow[index] = MassFlowProperty(chem.ID, molar_flow, index, chem.MW)
        self._data_cache['mass_flow'] = mass_flow = MassFlowIndex.from_data(
                                                        property_array(mass_flow),
                                                        phases, chemicals)
    return mass_flow
MolarFlowIndex.by_mass = by_mass; del by_mass


# %% Volumetric flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'V',
                        'TP', 'phase', 'phase_container'))
def VolumetricFlowProperty(self):
    """Volumetric flow (m^3/hr)."""
    T, P = self.TP
    return self.molar_flow[self.index] * self.V(self.phase or self.phase_container[0], T, P)
    
@VolumetricFlowProperty.setter
def VolumetricFlowProperty(self, value):
    T, P = self.TP
    self.molar_flow[self.index] = value / self.V(self.phase or self.phase_container[0], T, P)

def by_volume(self, TP):
    try:
        volumetric_flow = self._data_cache[TP]
    except:
        chemicals = self.chemicals
        molar_flow = self.data
        volumetric_flow = np.zeros_like(molar_flow, dtype=object)
        for i, chem in enumerate(chemicals):
            volumetric_flow[i] = VolumetricFlowProperty(chem.ID, molar_flow, i, chem.V,
                                                        TP, None, self._phase)
        self._data_cache['volumetric_flow'] = \
        volumetric_flow = ChemicalVolumetricFlowIndex.from_data(property_array(volumetric_flow),
                                                                self._phase, chemicals)
    return volumetric_flow
ChemicalMolarFlowIndex.by_volume = by_volume
	
def by_volume(self, TP):
    try:
        volumetric_flow = self._data_cache[TP]
    except:
        phases = self.phases
        chemicals = self.chemicals
        molar_flow = self.data
        volumetric_flow = np.zeros_like(molar_flow, dtype=object)
        for i, phase in enumerate(phases):
            for j, chem in enumerate(chemicals):
                index = i, j
                phase_name = settings._phase_name[phase]
                volumetric_flow[index] = VolumetricFlowProperty(f"{phase_name}{chem.ID}", molar_flow,
                                                                index, chem.V, TP, phase)
        self._data_cache['volumetric_flow'] = \
        volumetric_flow = VolumetricFlowIndex.from_data(property_array(volumetric_flow),
                                                        phases, chemicals)
    return volumetric_flow
MolarFlowIndex.by_volume = by_volume; del by_volume


# %% Cut out functionality

