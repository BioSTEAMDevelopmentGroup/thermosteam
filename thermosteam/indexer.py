# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from .units_of_measure import AbsoluteUnitsOfMeasure
from . import utils
from .exceptions import UndefinedChemical
from ._phase import Phase, LockedPhase, NoPhase, PhaseIndexer, phase_tuple, check_phase
from free_properties import PropertyFactory, property_array
from collections.abc import Iterable
import numpy as np

__all__ = (
    'SplitIndexer',
    'ChemicalIndexer',
    'MaterialIndexer',
    'ChemicalMolarFlowIndexer', 
    'MolarFlowIndexer',
    'ChemicalMassFlowIndexer', 
    'MassFlowIndexer',
    'ChemicalVolumetricFlowIndexer',
    'VolumetricFlowIndexer',
    'MassFlowProperty',
    'VolumetricFlowProperty'
)

# %% Utilities

_new = object.__new__

def raise_material_indexer_index_error():
    raise IndexError("index by [phase, IDs] where phase is a "
                     "(str, ellipsis, or missing), and IDs is a "
                     "(str, tuple(str), ellipisis, or missing)")

def find_main_phase(indexers, default):
    main_indexer, *indexers = indexers
    try:
        phase = main_indexer.phase
        for i in indexers:
            if phase != i.phase: return default
    except:
        return default
    return phase

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]

# %% Abstract indexer
    
class Indexer:
    """Abstract class for fast indexing."""
    __slots__ = ('_data',)
    units = None
    
    def empty(self):
        self._data[:] = 0
    
    def isempty(self):
        return (self._data == 0.).all()
    
    def copy(self):
        new = self._copy_without_data()
        new._data = self._data.copy()
        return new
    __copy__ = copy
    
    def get_conversion_factor(self, units):
        if self.units:
            return self.units.conversion_factor(units)
        else:
            raise TypeError(f"{type(self).__name__} object is unitless; "
                            f"cannot get conversion factor for {units}")
    
    def get_data(self, units, *index):
        length = len(index)
        factor = self.get_conversion_factor(units)
        if length == 0:
            return factor * self._data
        elif length == 1:
            return factor * self[index[0]]
        else:
            return factor * self[index]
    
    def set_data(self, data, units, *index):
        length = len(index)
        data = np.asarray(data, dtype=float)
        factor = self.get_conversion_factor(units)
        scaled_data = data / factor
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

@utils.chemicals_user
class SplitIndexer(Indexer):
    """
    Create a SplitIndexer that can index a 1d-array given
    chemical IDs.
    
    Parameters
    ----------
    chemicals : Chemicals
        Required to define the chemicals that are present.
    **ID_data : float
        ID-value pairs
    
    """
    __slots__ = ('_chemicals',)

    def __new__(cls, chemicals=None, **ID_data):
        if ID_data:
            chemicals = tmo.settings.get_default_chemicals(chemicals)
            self = cls.from_data(chemicals.kwarray(ID_data), chemicals)
        else:
            self = cls.blank(chemicals)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._chemicals, False)        
    
    @classmethod
    def blank(cls, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._data = np.zeros(self._chemicals.size, float)
        return self
    
    @classmethod
    def from_data(cls, data, chemicals=None, check_data=True):
        self = _new(cls)
        self._load_chemicals(chemicals)
        if check_data:
            assert data.ndim == 1, 'data must be a 1d numpy array'
            assert data.size == self._chemicals.size, ('size of data must be equal to '
                                                       'size of chemicals')
            assert (data <= 1.).all(), 'data must be less or equal to one'
        self._data = data
        return self
    
    def __getitem__(self, key):
        chemicals = self._chemicals
        index, kind = chemicals._get_index_and_kind(key)
        if kind == 0 or kind == 1:
            return self._data[index]
        elif kind == 2:
            data = self._data
            return np.array([data[i] for i in index], dtype=object)
        else:
            raise IndexError('unknown error')
    
    def __setitem__(self, key, data):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind == 0 or kind == 1:
            self._data[index] = data
        elif kind == 2:
            local_data = self._data
            isa = isinstance
            if isa(data, Iterable):
                for i, x in zip(index, data): local_data[i] = x
            else:
                for i in index: local_data[i] = data
        else:
            raise IndexError('unknown error')
                
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        if tab:
            dlim = ",\n" + tab
        else:
            dlim = ", "
        ID_data = utils.repr_IDs_data(self._chemicals.IDs, self._data, dlim, start='')
        return f"{type(self).__name__}({ID_data})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
        IDs = self.chemicals.IDs
        data = self.data
        IDs, data = nonzeros(IDs, data)
        N_IDs = len(IDs)
        if N_IDs == 0:
            return f"{type(self).__name__}: (all zeros)"
        else:
            basic_info = f"{type(self).__name__}:\n "
        new_line = '\n '
        data_info = ''
        lengths = [len(i) for i in IDs]
        maxlen = max(lengths) + 1
        N_max = N or tmo.Stream.display_units.N
        too_many_chemicals = N_IDs > N_max
        N = N_max if too_many_chemicals else N_IDs
        for i in range(N):
            spaces = ' ' * (maxlen - lengths[i])
            if i != 0:
                data_info += new_line
            data_info += IDs[i] + spaces + f' {data[i]:.3g}'
        if too_many_chemicals:
            data_info += new_line + '...'
        return (basic_info
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

@utils.chemicals_user
class ChemicalIndexer(Indexer):
    """
    Create a ChemicalIndexer that can index a single-phase, 1d-array given
    chemical IDs.
    
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
    A ChemicalIndexer does not have any units defined. To use units of
    measure, use the  `ChemicalMolarIndexer`, `ChemicalMassIndexer`, or
    `ChemicalVolumetricIndexer`.
    
    """
    __slots__ = ('_chemicals', '_phase', '_data_cache')
    
    def __new__(cls, phase=NoPhase, units=None, chemicals=None, **ID_data):
        if ID_data:
            chemicals = tmo.settings.get_default_chemicals(chemicals)
            self = cls.from_data(chemicals.kwarray(ID_data), phase, chemicals)
            if units: self.set_data(self._data, units)
        else:
            self = cls.blank(phase, chemicals)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phase, self._chemicals, False)
    
    def __getitem__(self, key):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind == 0:
            return self._data[index]
        elif kind == 1:
            return self._data[index].sum()
        elif kind == 2:
            arr = np.zeros(len(index))
            data = self._data
            isa = isinstance
            for d, s in enumerate(index):
                if isa(s, list): 
                    arr[d] = data[s].sum()
                else:
                    arr[d] = data[s]
            return arr
        else:
            raise IndexError('unknown index error')
    
    def __setitem__(self, key, data):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind == 0:
            self._data[index] = data
        elif kind == 1:
            raise IndexError(f"'{key}' is a chemical group; cannot set values by chemical group")
        elif kind == 1:
            for i, k in zip(index, key): 
                if isinstance(i, list):
                    raise IndexError(f"'{k}' is a chemical group; cannot set values by chemical group")
        else:
            raise IndexError('unknown error')
            
    def sum_across_phases(self):
        return self._data
    
    @property
    def get_index(self):
        return self._chemicals.get_index
    
    def mix_from(self, others):
        self.phase = find_main_phase(others, self.phase)
        chemicals = self._chemicals
        data = self._data
        chemicals_data = [(i._chemicals, i._data.copy() if i is self else i.sum_across_phases())
                          for i in others]
        data[:] = 0.
        for ichemicals, idata in chemicals_data:
            if chemicals is ichemicals:
                data[:] += idata
            else:
                other_index, = np.where(idata)
                IDs = ichemicals.IDs
                self_index = chemicals.indices([IDs[i] for i in other_index])
                data[self_index] += idata[other_index]
    
    def separate_out(self, other):
        if self._chemicals is other._chemicals:
            self._data[:] -= other.sum_across_phases()
        else:
            idata = other._data
            other_index, = np.where(idata)
            IDs = other._chemicals.IDs
            self_index = self._chemicals.indices([IDs[i] for i in other_index])
            self._data[self_index] -= idata[other_index]
    
    def to_material_indexer(self, phases):
        material_array = self._MaterialIndexer.blank(phases, self._chemicals)
        material_array[self.phase] = self._data
        return material_array
    
    def copy_like(self, other):
        if self is other: return
        if self.chemicals is other.chemicals:
            self._data[:] = other._data
        else:
            self.empty()
            other_index, = np.where(other._data)
            IDs = other.chemicals.IDs
            self_index = self.chemicals.indices([IDs[i] for i in other_index])
            self._data[self_index] = other._data[other_index]
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
    def from_data(cls, data, phase=NoPhase, chemicals=None, check_data=True):
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
    
    def get_phase_and_composition(self):
        """Return phase and composition."""
        data = self._data
        return self.phase, data / data.sum()
    
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
        ID_data = utils.repr_IDs_data(self._chemicals.IDs, self._data, dlim)
        return f"{type(self).__name__}({phase}{ID_data})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
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
        N_max = N or tmo.Stream.display_units.N
        too_many_chemicals = N_IDs > N_max
        N = N_max if too_many_chemicals else N_IDs
        for i in range(N):
            spaces = ' ' * (maxlen - lengths[i])
            if i != 0:
                data_info += new_line
            data_info += IDs[i] + spaces + f' {data[i]:.3g}'
        if too_many_chemicals:
            data_info += new_line + '...'
        return (basic_info
              + beginning
              + data_info)

    _ipython_display_ = show = SplitIndexer.show 
      
@utils.chemicals_user
class MaterialIndexer(Indexer):
    """
    Create a MaterialIndexer that can index a multi-phase, 2d-array given
    the phase and chemical IDs.
    
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
    __slots__ = ('_chemicals', '_phases', '_phase_indexer',
                 '_index_cache', '_data_cache')
    _index_caches = {}
    _ChemicalIndexer = ChemicalIndexer
    
    def __new__(cls, phases=None, units=None, chemicals=None, **phase_data):
        self = cls.blank(phases or phase_data, chemicals)
        if phase_data:
            data = self._data
            get_index = self._chemicals.get_index
            get_phase_index = self.get_phase_index
            for phase, ID_data in phase_data.items():
                check_phase(phase)
                IDs, row = zip(*ID_data)
                data[get_phase_index(phase), get_index(IDs)] = row
            if units: self.set_data(data, units)
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases, self._chemicals, False)
    
    def phases_are_empty(self, phases):
        get_phase_index = self.get_phase_index
        data = self._data
        for phase in set(self._phases).intersection(phases):
            if data[get_phase_index(phase)].any(): return False
        return True
    
    def sum_across_phases(self):
        return self._data.sum(0)
    
    def copy_like(self, other):
        if self is other: return
        if isinstance(other, ChemicalIndexer):
            self.empty()
            other_data = other._data
            phase_index = self.get_phase_index(other.phase)
            if self.chemicals is other.chemicals:
                self._data[phase_index, :] = other_data
            else:
                other_index, = np.where(other_data)
                IDs = other.chemicals.IDs
                self_index = self.chemicals.indices([IDs[i] for i in other_index])
                self._data[phase_index, self_index] += other._data[other_index]
        else:
            if self.chemicals is other.chemicals:
                self._data[:] = other._data
            else:
                self.empty()
                other_data = other._data
                other_index, = np.where(other_data.any(0))
                IDs = other.chemicals.IDs
                self_index = self.chemicals.indices([IDs[i] for i in other_index])
                self._data[:, self_index] = other_data[:, other_index]
    
    def mix_from(self, others):
        isa = isinstance
        data = self._data
        get_phase_index = self.get_phase_index
        chemicals = self._chemicals
        phases = self._phases
        indexer_data = [(i, i._data.copy() if i is self else i._data) for i in others]
        data[:] = 0.
        for i, idata in indexer_data:
            if isa(i, MaterialIndexer):
                if phases == i.phases:
                    if chemicals is i.chemicals:
                        data[:] += idata
                    else:
                        idata = i._data
                        other_index, = np.where(idata.any(0))
                        IDs = i.chemicals.IDs
                        self_index = chemicals.indices([IDs[i] for i in other_index])
                        data[:, self_index] += idata[:, other_index]
                else:
                    if chemicals is i.chemicals:
                        for phase, idata in zip(i.phases, idata):
                            if not idata.any(): continue
                            data[get_phase_index(phase), :] += idata
                    else:
                        for phase, idata in zip(i.phases, idata):
                            if not idata.any(): continue
                            other_index, = np.where(idata)
                            IDs = i.chemicals.IDs
                            self_index = chemicals.indices([IDs[i] for i in other_index])
                            data[get_phase_index(phase), self_index] += idata[other_index]
            elif isa(i, ChemicalIndexer):
                if chemicals is i.chemicals:
                    data[get_phase_index(i.phase), :] += idata
                else:
                    other_index, = np.where(idata != 0.)
                    IDs = i.chemicals.IDs
                    self_index = chemicals.indices([IDs[i] for i in other_index])
                    data[get_phase_index(i.phase), self_index] += idata[other_index]
            else:
                raise ValueError("can only mix from chemical or material indexers")
    
    def separate_out(self, other):
        isa = isinstance
        data = self._data
        get_phase_index = self.get_phase_index
        chemicals = self._chemicals
        phases = self._phases
        idata = other._data
        if isa(other, MaterialIndexer):
            if phases == other.phases:
                if chemicals is other.chemicals:
                    data[:] -= idata
                else:
                    idata = other._data
                    other_index, = np.where(idata.any(0))
                    IDs = other.chemicals.IDs
                    self_index = chemicals.indices([IDs[i] for i in other_index])
                    data[:, self_index] -= idata[:, other_index]
            else:
                if chemicals is other.chemicals:
                    for phase, idata in zip(other.phases, idata):
                        if not idata.any(): continue
                        data[get_phase_index(phase), :] -= idata
                else:
                    for phase, idata in zip(other.phases, idata):
                        if not idata.any(): continue
                        other_index, = np.where(idata)
                        IDs = other.chemicals.IDs
                        self_index = chemicals.indices([IDs[i] for i in other_index])
                        data[get_phase_index(phase), self_index] -= idata[other_index]
        elif isa(other, ChemicalIndexer):
            if chemicals is other.chemicals:
                data[get_phase_index(other.phase), :] -= idata
            else:
                other_index, = np.where(idata != 0.)
                IDs = other.chemicals.IDs
                self_index = chemicals.indices([IDs[i] for i in other_index])
                data[get_phase_index(other.phase), self_index] -= idata[other_index]
        else:
            raise ValueError("can only separate out from chemical or material indexers")
    
    def _set_phases(self, phases):
        self._phases = phases = phase_tuple(phases)
        self._phase_indexer = PhaseIndexer(phases)
    
    def _set_cache(self):
        caches = self._index_caches
        key = self._phases, self._chemicals
        try:
            self._index_cache = caches[key]
        except KeyError:
            self._index_cache = caches[key] = {}
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._phases = self._phases
        new._chemicals = self._chemicals
        new._phase_indexer = self._phase_indexer
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
    
    @property
    def get_phase_index(self):
        return self._phase_indexer
    
    def to_chemical_indexer(self, phase=NoPhase):
        return self._ChemicalIndexer.from_data(self._data.sum(0), phase, self._chemicals, False)
    
    def to_material_indexer(self, phases):
        material_indexer = self.__class__.blank(phases, self._chemicals)
        for phase, data in self:
            if data.any(): material_indexer[phase] = data
        return material_indexer
    
    def get_phase(self, phase):
        return self._ChemicalIndexer.from_data(self._data[self.get_phase_index(phase)],
                                               LockedPhase(phase), self._chemicals, False)
    
    def __getitem__(self, key):
        index, kind, sum_across_phases = self._get_index_data(key)
        if sum_across_phases:
            if kind == 0: # Normal
                values = self._data[:, index].sum(0)
            elif kind == 1: # Chemical group
                values = self._data[:, index].sum()
            elif kind == 2: # Nested chemical group
                data = self._data
                values = np.array([data[:, i].sum() for i in index], dtype=float)
        else:
            if kind == 0: # Normal
                return self._data[index]
            elif kind == 1: # Chemical group
                phase, index = index
                if phase == slice(None):
                    values = self._data[phase, index].sum(1)
                else:
                    values = self._data[phase, index].sum()
            elif kind == 2: # Nested chemical group
                data = self._data
                isa = isinstance
                phase, index = index
                if phase == slice(None):
                    values = np.zeros([len(self.phases), len(index)])
                    for d, s in enumerate(index):
                        if isa(s, list): 
                            values[:, d] = data[phase, s].sum(1)
                        else:
                            values[:, d] = data[phase, s]
                else:
                    values = np.zeros(len(index))
                    for d, s in enumerate(index):
                        if isa(s, list): 
                            values[d] = data[phase, s].sum()
                        else:
                            values[d] = data[phase, s]
        return values
    
    def __setitem__(self, key, data):
        index, kind, sum_across_phases = self._get_index_data(key)
        if sum_across_phases:
            raise IndexError("multiple phases present; must include phase key "
                             "to set chemical data")
        if kind == 0:
            self._data[index] = data
        elif kind == 1: # Chemical group
            phase, index = index
            if not sum_across_phases: _, key = key
            raise IndexError(f"'{key}' is a chemical group; cannot set values by chemical group")
        elif kind == 2: # Nested chemical group
            for i, k in zip(index, key): 
                if isinstance(i, list):
                    raise IndexError(f"'{k}' is a chemical group; cannot set values by chemical group")
            raise IndexError('unknown error')
        else:
            raise IndexError('unknown error')
    
    def _get_index_data(self, key):
        cache = self._index_cache
        try: 
            index_data = cache[key]
        except KeyError:
            try:
                index, kind = self._chemicals._get_index_and_kind(key)
            except UndefinedChemical as error:
                index, kind = self._get_index_and_kind(key, error)
                sum_across_phases = False
            else:
                sum_across_phases = True
            cache[key] = index_data = (index, kind, sum_across_phases)
            utils.trim_cache(cache)
        except TypeError:
            raise TypeError("only strings, tuples, and ellipsis are valid index keys")
        return index_data
    
    def _get_index_and_kind(self, phase_IDs, undefined_chemical_error):
        isa = isinstance
        if isa(phase_IDs, str):
            if len(phase_IDs) == 1: 
                index = self.get_phase_index(phase_IDs)
                kind = 0
            else:
                raise undefined_chemical_error
        elif phase_IDs is ...:
            index = slice(None)
            kind = 0
        else:
            phase = phase_IDs[0]
            if isa(phase, str):
                if len(phase) == 1:
                    phase_index = self.get_phase_index(phase)
                else:
                    raise undefined_chemical_error
            elif phase is ...:
                phase_index = slice(None)
            else:
                raise_material_indexer_index_error()
            try:
                phase, IDs = phase_IDs
            except:
                raise_material_indexer_index_error()
            chemical_index, kind = self._chemicals._get_index_and_kind(IDs)
            index = (phase_index, chemical_index)
        return index, kind
    
    def __iter__(self):
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
        for phase, data in self:
            ID_data = utils.repr_couples(", ", IDs, data)
            if ID_data:
                phase_data.append(f"{phase}=[{ID_data}]")
        tabs = int(tabs) if tabs else 1
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
            phases = f'phases={self._phases}'
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
        for phase in self._phases:
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
                data_info += f'{IDs[i]} ' + spaces + f' {data[i]:.3g}'
            if too_many_chemicals: data += new_line + '...'
            # Put it together
            phases_data_info += beginning + data_info + '\n'
            
        return basic_info + phases_data_info.rstrip('\n')
    
    _ipython_display_ = show = ChemicalIndexer.show   
    
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
    MaterialIndexerSubclass.units = AbsoluteUnitsOfMeasure(units)
    
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
    return 1000. * f_mol * V(*self.TP) if f_mol else 0.
    
@VolumetricFlowProperty.setter
def VolumetricFlowProperty(self, value):
    if value:
        phase = self.phase or self.phase_container.phase
        V = getattr(self.V, phase) if hasattr(self.V, phase) else self.V
        self.mol[self.index] = value / V(*self.TP) / 1000.
    else:
        self.mol[self.index] = 0.

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
                phase_name = tmo.settings._phase_names[phase]
                vol[index] = VolumetricFlowProperty(f"{phase_name}{chem.ID}", 
                                                    mol, index, chem.V, TP, phase)
        self._data_cache[TP] = \
        vol = VolumetricFlowIndexer.from_data(property_array(vol),
                                              phases, chemicals,
                                              False)
    return vol
MolarFlowIndexer.by_volume = by_volume; del by_volume
del PropertyFactory
