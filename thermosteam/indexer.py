# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentVector/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from .units_of_measure import AbsoluteUnitsOfMeasure
from . import utils
from .exceptions import UndefinedChemicalAlias
from .base import (
    SparseVector, SparseArray, sparse_vector, sparse_array,
    MassFlowDict, VolumetricFlowDict,
)
from ._phase import Phase, LockedPhase, NoPhase, PhaseIndexer, phase_tuple
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
)

phase_names = {
    'g': 'Gas',
    'l': 'Liquid',
    's': 'Solid',
    'L': 'LIQUID',
    'S': 'SOLID',
}
    

# %% Utilities

_new = object.__new__

def raise_material_indexer_index_error():
    raise IndexError("index by [phase, IDs] where phase is a "
                     "(str, ellipsis, or missing), and IDs is a "
                     "(str, Sequence[str], ellipisis, or missing)")

def find_main_phase(indexers, default):
    main_indexer, *indexers = indexers
    try:
        phase = main_indexer._phase.phase
        for i in indexers:
            if phase != i._phase.phase: return default
    except:
        return default
    return phase

def nonzeros(IDs, data):
    if hasattr(IDs, 'dct'):
        dct = data.dct
        index = sorted(data.nonzero_keys())
        return  [IDs[i] for i in index], [dct[i] for i in index]
    else:
        index, = np.where(data)
        return [IDs[i] for i in index], [data[i] for i in index]

def index_overlap(left_chemicals, right_chemicals, right_index):
    CASs_all = right_chemicals.CASs
    CASs = tuple([CASs_all[i] for i in right_index])
    cache = left_chemicals._index_cache
    if CASs in cache:
        left_index, kind = cache[CASs]
        if kind:
            raise RuntimeError('conflict in chemical groups and aliases between property packages')
        else:
            return left_index, right_index
    else:
        dct = left_chemicals._index
        N = len(CASs)
        left_index = [0] * N
        for i in range(N):
            CAS = CASs[i]
            if CAS in dct:
                index = dct[CAS]
                if hasattr(index, '__iter__'): raise RuntimeError('conflict in chemical groups and aliases between property packages')
                left_index[i] = index
            else:
                raise UndefinedChemicalAlias(CAS)
        cache[CASs] = (left_index, 0)
        if len(cache) > 100: cache.pop(cache.__iter__().__next__())
        return left_index, right_index

# %% Abstract indexer
    
class Indexer:
    """Abstract class for fast indexing."""
    __slots__ = ('data',)
    units = None
    
    def empty(self):
        self.data.clear()
    
    def isempty(self):
        return not self.data.any()
    
    def copy(self):
        new = self._copy_without_data()
        new.data = self.data.copy()
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
            return factor * self.data
        elif length == 1:
            return factor * self[index[0]]
        else:
            return factor * self[index]
    
    def set_data(self, data, units, *index):
        length = len(index)
        factor = self.get_conversion_factor(units)
        scaled_data = data / factor
        if length == 0:
            self.data[:] = scaled_data
        elif length == 1:
            self[index[0]] = scaled_data
        else:
            self[index] = scaled_data
    

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
        self = cls.blank(chemicals)
        if ID_data:
            IDs = tuple(ID_data)
            values = list(ID_data.values())
            self[IDs] = values
        return self
    
    def __reduce__(self):
        return self.from_data, (self.data, self._chemicals, False)        
    
    def reset_chemicals(self, chemicals, container=None):
        old_data = self.data
        if container is None:
            self.data = data = SparseVector.from_size(chemicals.size)
        else:
            self.data = data = container
            data.clear()
        for CAS, split in zip(self._chemicals.CASs, old_data):
            if CAS in chemicals: data.dct[chemicals.index(CAS)] = split
        self._chemicals = chemicals
        return old_data
    
    @classmethod
    def blank(cls, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self.data = SparseVector.from_size(self._chemicals.size)
        return self
    
    @classmethod
    def from_data(cls, data, chemicals=None, check_data=True):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self.data = data = sparse_vector(data)
        if check_data:
            assert data.ndim == 1, 'data must be a 1d numpy array'
            assert data.size == self._chemicals.size, ('size of data must be equal to '
                                                       'size of chemicals')
            assert (data <= 1.).all(), 'data must be less or equal to one'
        return self
    
    def __getitem__(self, key):
        chemicals = self._chemicals
        index, kind = chemicals._get_index_and_kind(key)
        if kind == 0 or kind == 1:
            return self.data[index]
        elif kind == 2:
            data = self.data
            return np.array([data[i] for i in index], dtype=object)
        else:
            raise IndexError('unknown error')
    
    def __setitem__(self, key, data):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind == 0 or kind == 1:
            self.data[index] = data
        elif kind == 2:
            sparse_data = self.data
            if hasattr(data, '__iter__'):
                for i, x in zip(index, data): sparse_data[i] = x
            else:
                for i in index: sparse_data[i] = data
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
        ID_data = utils.repr_IDs_data(self._chemicals.IDs, self.data.to_array(self._chemicals.size), dlim, start='')
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
    phase : [str or PhaseContainer] {'s', 'l', 'g', 'S', 'L'}
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
    __slots__ = ('_chemicals', '_phase', '_cache_data')
    
    def __new__(cls, phase=NoPhase, units=None, chemicals=None, **ID_data):
        self = cls.blank(phase, chemicals)
        if ID_data:
            IDs = tuple(ID_data)
            values = list(ID_data.values())
            self[IDs] = values
            if units: self.set_data(self.data, units)
        return self
    
    def reset_chemicals(self, chemicals, container=None):
        old_data = self.data
        old_container = (old_data, self._cache_data)
        if container is None:
            self.data = data = SparseVector.from_size(chemicals.size)
            self._cache_data = {}
        else:
            data, self._cache_data = container
            self.data =  data
            data.clear()
        for CAS, value in zip(self._chemicals.CASs, old_data):
            if value: data.dct[chemicals.index(CAS)] = value
        self._chemicals = chemicals
        return old_container
    
    def __reduce__(self):
        return self.from_data, (self.data, self._phase, self._chemicals, False)
    
    def __getitem__(self, key):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind == 0:
            return self.data[index]
        elif kind == 1:
            return self.data.sum_of(index)
        elif kind == 2:
            arr = np.zeros(len(index))
            data = self.data
            for d, s in enumerate(index):
                arr[d] = data.sum_of(s)
            return arr
        else:
            raise IndexError('unknown index error')
    
    def __setitem__(self, key, data):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind == 0:
            self.data[index] = data
        elif kind == 1:
            composition = self.group_compositions[key]
            self.data[index] = data * composition
        elif kind == 2:
            sparse_data = self.data
            group_compositions = self.group_compositions
            for n in range(len(index)):
                i = index[n]
                sparse_data[i] = data[n] * group_compositions[key[n]] if hasattr(i, '__iter__') else data[n]
        else:
            raise IndexError('unknown error')
    
    def sum_across_phases(self):
        return self.data
    
    @property
    def get_index(self):
        return self._chemicals.get_index
    
    def mix_from(self, others):
        self.phase = find_main_phase(others, self.phase)
        chemicals = self._chemicals
        data = self.data
        sc_data = [] # Same chemicals
        other_data = [] # Different chemicals
        repeated_data = 0
        for i in others:
            if i is self:
                repeated_data += 1
            else:
                idata = i.sum_across_phases()
                if idata.shares_data_with(data): idata = idata.copy()
                ichemicals = i._chemicals
                if ichemicals is chemicals:
                    sc_data.append(idata)
                else:
                    other_data.append(
                        (idata, *index_overlap(chemicals, ichemicals, [*idata.nonzero_keys()]))
                    )
        if repeated_data == 0:
            data.clear()
        elif repeated_data > 1:
            data *= repeated_data
        for i in sc_data: data += i
        for idata, left_index, right_index in other_data: 
            data[left_index] += idata[right_index]
    
    def separate_out(self, other):
        if self._chemicals is other._chemicals:
            self.data -= other.sum_across_phases()
        else:
            other_data = other.data
            left_index, right_index = index_overlap(self._chemicals, other._chemicals, [*other_data.nonzero_keys()])
            self.data[left_index] -= other_data[right_index]
    
    def to_material_indexer(self, phases):
        material_array = self._MaterialIndexer.blank(phases, self._chemicals)
        material_array[self.phase].copy_like(self.data)
        return material_array
    
    def copy_like(self, other):
        if self is other: return
        if self.chemicals is other.chemicals:
            self.data.copy_like(other.data)
        else:
            self.empty()
            other_data = other.data
            left_index, right_index = index_overlap(self._chemicals, other._chemicals, [*other_data.nonzero_keys()])
            self.data[left_index] = other_data[right_index]
        self.phase = other.phase
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._chemicals = self._chemicals
        new._phase = self._phase.copy()
        new._cache_data = {}
        return new
    
    @classmethod
    def blank(cls, phase, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self.data = SparseVector.from_size(chemicals.size)
        self._phase = Phase.convert(phase)
        self._cache_data = {}
        return self
    
    @classmethod
    def from_data(cls, data, phase=NoPhase, chemicals=None, check_data=True):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._phase = Phase.convert(phase)
        self.data = data = sparse_vector(data)
        if check_data:
            assert data.ndim == 1, 'material data must be a 1d numpy array'
            assert data.size == self._chemicals.size, ('size of material data must be equal to '
                                                       'size of chemicals')
        self._cache_data = {}
        return self
    
    @property
    def phase(self):
        return self._phase._phase
    @phase.setter
    def phase(self, phase):
        self._phase.phase = phase
    
    def get_phase_and_composition(self):
        """Return phase and composition."""
        data = self.data
        total = data.sum()
        if total <= 0.: raise RuntimeError(f"'{phase_names[self.phase]}' phase does not exist")
        return self.phase, data / total
    
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
        ID_data = utils.repr_IDs_data(self._chemicals.IDs, self.data.to_array(), dlim)
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
                 '_index_cache', '_cache_data')
    _index_caches = {}
    _ChemicalIndexer = ChemicalIndexer
    
    def __new__(cls, phases=None, units=None, chemicals=None, **phase_data):
        self = cls.blank(phases or phase_data, chemicals)
        if phase_data:
            for phase, ID_data in phase_data.items():
                IDs, data = zip(*ID_data)
                self[phase, IDs] = data
            if units: self.set_data(data, units)
        return self
    
    def reset_chemicals(self, chemicals, container=None):
        old_data = self.data
        old__cache_data = self._cache_data
        N_phases = len(self._phases)
        if container is None:
            self.data = data = SparseArray.from_shape([N_phases, chemicals.size])
            self._cache_data = {}
        else:
            data, cache = container
            data[:] = 0.
        old_chemicals = self._chemicals
        old_index = range(old_chemicals.size)
        CASs = old_chemicals.CASs
        for i in range(N_phases):
            for j in old_index:
                value = old_data[i, j]
                if value: data[i, chemicals.index(CASs[j])] = value
        self._load_chemicals(chemicals)
        self._set_cache()
        return (old_data, old__cache_data)
    
    def __reduce__(self):
        return self.from_data, (self.data, self._phases, self._chemicals, False)
    
    def phases_are_empty(self, phases):
        get_phase_index = self.get_phase_index
        data = self.data
        for phase in set(self._phases).intersection(phases):
            if data[get_phase_index(phase)].any(): return False
        return True
    
    def sum_across_phases(self):
        return self.data.sum(0)
    
    def copy_like(self, other):
        if self is other: return
        if isinstance(other, ChemicalIndexer):
            self.empty()
            other_data = other.data
            phase_index = self.get_phase_index(other.phase)
            if self.chemicals is other.chemicals:
                self.data[phase_index, :] = other_data
            else:
                other_data = other.data
                left_index, right_index = index_overlap(self._chemicals, other._chemicals, [*other_data.nonzero_keys()])
                self.data[phase_index][left_index] = other_data[right_index] 
        else:
            if self.chemicals is other.chemicals:
                self.data[:] = other.data
            else:
                self.empty()
                other_data = other.data
                left_index, other_data = index_overlap(self._chemicals, other._chemicals, [*other_data.nonzero_columns()])
                self.data[:, left_index] = other_data[:, right_index]
    
    def mix_from(self, others):
        isa = isinstance
        data = self.data
        get_phase_index = self.get_phase_index
        chemicals = self._chemicals
        phases = self._phases
        repeated_data = 0
        spsc_data = [] # Same phases, same chemicals
        sp_data = [] # Same phases, different chemicals
        opsc_data = [] # One phase, same chemicals
        op_data = [] # One phase, different chemicals
        for i in others:
            if i is self:
                repeated_data += 1
            else:
                idata = i.data
                if idata.shares_data_with(data): idata = idata.copy()
                ichemicals = i._chemicals
                if isa(i, MaterialIndexer):
                    if phases == i.phases:
                        if chemicals is ichemicals:
                            spsc_data.append(idata)
                        else:
                            sp_data.append(
                                (idata, *index_overlap(chemicals, ichemicals, [*idata.nonzero_colums()]))
                            )
                    else:
                        if chemicals is ichemicals:
                            for phase, idata in zip(i.phases, idata):
                                if not idata.any(): continue
                                opsc_data.append(
                                    (idata, get_phase_index(i.phase))
                                )
                        else:
                            for phase, idata in zip(i.phases, idata):
                                if not idata.any(): continue
                                op_data.append(
                                    (get_phase_index(phase), idata, *index_overlap(chemicals, ichemicals, [*idata.nonzero_keys()]))
                                )
                elif isa(i, ChemicalIndexer):
                    if idata.shares_data_with(data): idata = idata.copy()
                    if chemicals is ichemicals:
                        opsc_data.append(
                            (idata, get_phase_index(i.phase))
                        )
                    else:
                        op_data.append(
                            (get_phase_index(phase), idata, *index_overlap(chemicals, ichemicals, [*idata.nonzero_keys()]))
                        )
                else:
                    raise ValueError("can only mix from chemical or material indexers")
        if repeated_data == 0:
            data[:] = 0.
        elif repeated_data > 1:
            data *= repeated_data
        for i in spsc_data: data[:] += i
        for idata, left_index, right_index in sp_data: 
            data[:, left_index] += idata[:, right_index]
        for idata, left_index in opsc_data: data[left_index, :] += idata
        for phase_index, idata, left_index, right_index in op_data: 
            data[phase_index, left_index] += idata[right_index]
    
    def separate_out(self, other):
        isa = isinstance
        data = self.data
        get_phase_index = self.get_phase_index
        chemicals = self._chemicals
        phases = self._phases
        idata = other.data
        if isa(other, MaterialIndexer):
            if phases == other.phases:
                if chemicals is other.chemicals:
                    data -= idata
                else:
                    idata = other.data
                    other_index, = np.where(idata.any(0))
                    CASs = other.chemicals.CASs
                    self_index = chemicals.indices([CASs[i] for i in other_index])
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
                        CASs = other.chemicals.CASs
                        self_index = chemicals.indices([CASs[i] for i in other_index])
                        data[get_phase_index(phase), self_index] -= idata[other_index]
        elif isa(other, ChemicalIndexer):
            if chemicals is other.chemicals:
                data[get_phase_index(other.phase), :] -= idata
            else:
                other_index, = np.where(idata != 0.)
                CASs = other.chemicals.CASs
                self_index = chemicals.indices([CASs[i] for i in other_index])
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
        new._cache_data = {}
        return new
    
    @classmethod
    def blank(cls, phases, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        self.data = SparseArray.from_shape([len(phases), self._chemicals.size])
        self._cache_data = {}
        return self
    
    @classmethod
    def from_data(cls, data, phases, chemicals=None, check_data=True):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        self.data = data = sparse_array(data)
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
        self._cache_data = {}
        return self
    
    @property
    def phases(self):
        return self._phases
    
    @property
    def get_phase_index(self):
        return self._phase_indexer
    
    def to_chemical_indexer(self, phase=NoPhase):
        return self._ChemicalIndexer.from_data(sum(self.data), phase, self._chemicals, False)
    
    def to_material_indexer(self, phases):
        material_indexer = self.__class__.blank(phases, self._chemicals)
        for phase, data in self:
            if data.any(): material_indexer[phase] = data
        return material_indexer
    
    def get_phase(self, phase):
        return self._ChemicalIndexer.from_data(self.data[self.get_phase_index(phase)],
                                               LockedPhase(phase), self._chemicals, False)
    
    def __getitem__(self, key):
        index, kind, sum_across_phases = self._get_index_data(key)
        if sum_across_phases:
            if kind == 0: # Normal
                values = self.data[:, index].sum(0)
            elif kind == 1: # Chemical group
                values = self.data[:, index].sum()
            elif kind == 2: # Nested chemical group
                data = self.data
                values = np.array([data[:, i].sum() for i in index], dtype=float)
        else:
            if kind == 0: # Normal
                return self.data[index]
            elif kind == 1: # Chemical group
                phase, index = index
                if phase == slice(None):
                    values = self.data[phase, index].sum(1)
                else:
                    values = self.data[phase, index].sum()
            elif kind == 2: # Nested chemical group
                data = self.data
                phase, index = index
                if phase == slice(None):
                    values = np.zeros([len(self.phases), len(index)])
                    for d, s in enumerate(index):
                        if hasattr(s, '__iter__'): 
                            values[:, d] = data[phase, s].sum(1)
                        else:
                            values[:, d] = data[phase, s]
                else:
                    values = np.zeros(len(index))
                    for d, s in enumerate(index):
                        if hasattr(s, '__iter__'): 
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
            self.data[index] = data
        elif kind == 1: # Chemical group
            phase, index = index
            _, key = key
            composition = self.group_compositions[key]
            self.data[phase, index] = data * composition
        elif kind == 2: # Nested chemical group
            phase, index = index
            sparse_data = self.data
            group_compositions = self.group_compositions
            for n in range(len(index)):
                i = index[n]
                sparse_data[phase, i] = data[n] * group_compositions[key[n]] if hasattr(i, '__iter__') else data[n]
        else:
            raise IndexError('unknown error')
    
    def _get_index_data(self, key):
        cache = self._index_cache
        try:
            index_data = cache[key]
        except KeyError:
            try:
                index, kind = self._chemicals._get_index_and_kind(key)
            except UndefinedChemicalAlias as error:
                index, kind = self._get_index_and_kind(key, error)
                sum_across_phases = False
            else:
                sum_across_phases = True
            cache[key] = index_data = (index, kind, sum_across_phases)
            utils.trim_cache(cache)
        except TypeError:
            try:
                key = tuple([i if i.__hash__ else tuple(i) for i in key])
                index_data = cache[key]
            except KeyError:
                try:
                    index, kind = self._chemicals._get_index_and_kind(key)
                except UndefinedChemicalAlias as error:
                    index, kind = self._get_index_and_kind(key, error)
                    sum_across_phases = False
                else:
                    sum_across_phases = True
                cache[key] = index_data = (index, kind, sum_across_phases)
                utils.trim_cache(cache)
            except TypeError:
                raise TypeError("only strings, sequences of strings, and ellipsis are valid index keys")
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
        return zip(self._phases, self.data)
    
    def iter_composition(self):
        """Iterate over phase-composition pairs."""
        array = self.data
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
    
def _new_Indexer(name, units, f_group_composition):
    dct = {'group_compositions': f_group_composition}
    ChemicalIndexerSubclass = type('Chemical' + name + 'Indexer', (ChemicalIndexer,), dct)
    MaterialIndexerSubclass = type(name + 'Indexer', (MaterialIndexer,), dct)
    
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

@property
def group_wt_compositions(self):
    return self._chemicals._group_wt_compositions

@property
def group_mol_compositions(self):
    return self._chemicals._group_mol_compositions

@property
def group_vol_composition(self):
    raise AttributeError('cannot set groups by volumetric flow')

ChemicalMolarFlowIndexer, MolarFlowIndexer = _new_Indexer('MolarFlow', 'kmol/hr', group_mol_compositions)
ChemicalMassFlowIndexer, MassFlowIndexer = _new_Indexer('MassFlow', 'kg/hr', group_wt_compositions)
ChemicalVolumetricFlowIndexer, VolumetricFlowIndexer = _new_Indexer('VolumetricFlow', 'm^3/hr', group_vol_composition)

# %% Mass flow properties

def by_mass(self):
    """Return a ChemicalMassFlowIndexer that references this object's molar data."""
    try:
        mass = self._cache_data['mass']
    except:
        chemicals = self.chemicals
        self._cache_data['mass'] = mass = \
        ChemicalMassFlowIndexer.from_data(
            SparseVector.from_dict(
                MassFlowDict(self.data.dct, chemicals.MW),
                chemicals.size
            ),
            self._phase, chemicals,
            False
        )
    return mass
ChemicalMolarFlowIndexer.by_mass = by_mass

def by_mass(self):
    """Return a MassFlowIndexer that references this object's molar data."""
    try:
        mass = self._cache_data['mass']
    except:
        chemicals = self.chemicals
        size = chemicals.size
        MW = chemicals.MW
        self._cache_data['mass'] = mass = \
        MassFlowIndexer.from_data(
            SparseArray.from_rows([
                SparseVector.from_dict(MassFlowDict(i.dct, MW), size)
                for i in self.data
            ]),
            self.phases, chemicals,
            False
        )
    return mass
MolarFlowIndexer.by_mass = by_mass; del by_mass


# %% Volumetric flow properties

def by_volume(self, TP):
    """Return a ChemicalVolumetricFlowIndexer that references this object's molar data.
    
    Parameters
    ----------
    TP : ThermalCondition
    
    """
    try:
        vol = self._cache_data['vol', TP]
    except:
        chemicals = self._chemicals
        V = [i.V for i in chemicals]
        phase = self._phase
        self._cache_data['vol', TP] = \
        vol = ChemicalVolumetricFlowIndexer.from_data(
            SparseVector.from_dict(
                VolumetricFlowDict(self.data.dct, TP, V, None, phase, {}),
                chemicals.size
            ),
            phase, chemicals,
            False
        )
    return vol
ChemicalMolarFlowIndexer.by_volume = by_volume
	
def by_volume(self, TP):
    """Return a VolumetricFlowIndexer that references this object's molar data.
    
    Parameters
    ----------
    TP : ThermalCondition
    
    """
    try:
        vol = self._cache_data[TP]
    except:
        phases = self._phases
        chemicals = self._chemicals
        V = [i.V for i in chemicals]
        size = chemicals.size
        self._cache_data[TP] = \
        vol = VolumetricFlowIndexer.from_data(
            SparseArray.from_rows([
                SparseVector.from_dict(VolumetricFlowDict(i.dct, TP, V, j, None, {}), size)
                for i, j in zip(self.data, self._phases)
            ]),
            phases, chemicals,
            False
        )
    return vol
MolarFlowIndexer.by_volume = by_volume; del by_volume

