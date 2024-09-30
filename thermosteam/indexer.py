# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from .units_of_measure import AbsoluteUnitsOfMeasure
from . import utils
from .exceptions import UndefinedChemicalAlias, UndefinedPhase
from .base import (
    SparseVector, SparseArray, sparse_vector, sparse_array,
    MassFlowDict, VolumetricFlowDict, get_ndim
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

def set_main_phase(main_indexer, indexers):
    other_indexer, *indexers = indexers
    try:
        phase = other_indexer._phase.phase
        for i in indexers:
            if phase != i._phase.phase: return
        main_indexer.phase = phase
    except: pass

def raise_material_indexer_index_error():
    raise IndexError("index by [phase, IDs] where phase is a "
                     "(str, ellipsis, or missing), and IDs is a "
                     "(str, Sequence[str], ellipsis, or missing)")

def nonzeros(IDs, data):
    if hasattr(IDs, 'dct'):
        dct = data.dct
        return  [IDs[i] for i in dct], [*dct.values()]
    else:
        index, = np.where(data)
        return [IDs[i] for i in index], [data[i] for i in index]

def index_overlap(left_chemicals, right_chemicals, right_index):
    CASs_all = right_chemicals.CASs
    CASs = tuple([CASs_all[i] for i in right_index])
    cache = left_chemicals._index_cache
    if CASs in cache:
        left_index, kind = cache[CASs]
        if kind == 0 or kind == 3:
            return left_index, right_index
        else:
            raise RuntimeError('conflict in chemical groups and aliases between property packages')
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

def get_sparse_chemical_data(sparse, index, kind):
    if kind is None: return sparse
    dct = sparse.dct
    if kind == 0:
        return dct.get(index, 0.)
    elif kind == 1:
        return sum([dct[i] for i in index if i in dct])
    elif kind == 2:
        return np.array([
            (sum([dct[j] for j in i if j in dct]) if i.__class__ is list else dct.get(i, 0.))
            for n, i in enumerate(index)
        ])
    elif kind == 3:
        return np.array([dct.get(i, 0.) for i in index])
    else:
        raise IndexError('invalid index kind')

def reset_sparse_chemical_data(sparse, data):
    if data is sparse: return
    dct = sparse.dct
    dct.clear()
    if data.__class__ is SparseVector:
        dct.update(data.dct)
    else:
        ndim = get_ndim(data)
        if ndim == 0:
            if data:
                data = float(data)
                for i in range(sparse.size): dct[i] = data
        elif ndim == 1:
            for i, j in enumerate(data):
                if j: dct[i] = float(j)
                elif i in dct: del dct[i]  
        else:
            raise IndexError(
                'cannot set an array element with a sequence'
            )

def set_sparse_chemical_data(sparse, index, kind, data, key, parent):
    if kind is None:
        reset_sparse_chemical_data(sparse, data)
        return
    ndim = get_ndim(data)
    dct = sparse.dct
    if kind == 0:
        if ndim:
            raise IndexError(
                'cannot set an array element with a sequence'
            )
        if data:
            dct[index] = float(data)
        elif index in dct: 
            del dct[index]
    elif kind == 1:
        if ndim == 0:
            composition = parent.group_compositions[key]
            values = data * composition
            for i, j in zip(index, values):
                if j: dct[i] = float(j)
                elif i in dct: del dct[i]
        elif ndim == 1:
            for i, j in zip(index, data):
                if j: dct[i] = float(j)
                elif i in dct: del dct[i]
        else:
            raise IndexError(
                'cannot set an array element with a sequence'
            )
    elif kind == 2:
        if ndim == 0:
            if data:
                data = float(data)
                for n, i in enumerate(index):
                    if i.__class__ is list:
                        values = data * parent.group_compositions[key[n]]
                        for k, j in zip(i, values):
                            if j: dct[k] = float(j)
                            elif k in dct: del dct[k]
                    else:
                        dct[i] = data
            else:
                for i in index:
                    if i.__class__ is list:
                        for j in i:
                            if j in dct: del dct[j]
                    elif i in dct:
                        del dct[i]
        elif ndim == 1:
            for n, i in enumerate(index):
                if i.__class__ is list:
                    values = data[n] * parent.group_compositions[key[n]]
                    for k, j in zip(i, values):
                        if j: dct[k] = float(j)
                        elif k in dct: del dct[k]
                else:
                    j = data[n]
                    if j: dct[i] = float(j)
                    elif i in dct: del dct[i]
        else:
            raise IndexError(
                'cannot set an array element with a sequence'
            )
    elif kind == 3:
        if ndim == 0:
            if data:
                data = float(data)
                for i in index: dct[i] = data
            else:
                for i in index:
                    if i in dct: del dct[i]
        elif ndim == 1:
            for i, j in zip(index, data):
                if j: dct[i] = float(j)
                elif i in dct: del dct[i]
        else:
            raise IndexError(
                'cannot set an array element with a sequence'
            )
    else:
        raise IndexError('invalid index kind') 

# %% Abstract indexer
    
class Indexer:
    """Abstract class for fast indexing."""
    __slots__ = ('data',)
    units = None
    
    @property
    def _data(self): # For backwards compatibility
        return self.data
    
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
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind is None: return self.data
        dct = self.data.dct
        if kind == 0:
            return dct.get(index, 0.)
        elif kind == 1:
            return np.array([dct.get(i, 0.) for i in index])
        elif kind == 2:
            return np.array([
                (np.array([dct.get(j, 0.) for j in i]) if i.__class__ is list else dct.get(i, 0.))
                for n, i in enumerate(index)
            ], dtype=object)
        elif kind == 3:
            return np.array([dct.get(i, 0.) for i in index])
        else:
            raise IndexError('invalid index kind')
    
    def __setitem__(self, key, data):
        index, kind = self._chemicals._get_index_and_kind(key)
        if kind is None:
            reset_sparse_chemical_data(self.data, data)
            return
        ndim = get_ndim(data)
        dct = self.data.dct
        if kind == 0:
            if ndim:
                raise IndexError(
                    'cannot set an array element with a sequence'
                )
            if data:
                dct[index] = float(data)
            elif index in dct: 
                del dct[index]
        elif kind == 1:
            if ndim == 0:
                if data:
                    data = float(data)
                    for i in index: dct[i] = data
                else:
                    for i in index:
                        if i in dct: del dct[i]
            elif ndim == 1:
                for i, j in zip(index, data):
                    if j: dct[i] = float(j)
                    elif i in dct: del dct[i]
            else:
                raise IndexError(
                    'cannot set an array element with a sequence'
                )
        elif kind == 2:
            if ndim == 0:
                if data:
                    data = float(data)
                    for i in index:
                        if i.__class__ is list:
                            for j in i: dct[j] = data
                        else:
                            dct[i] = data
                else:
                    for i in index:
                        if i.__class__ is list:
                            for j in i:
                                if j in dct: del dct[j]
                        elif i in dct:
                            del dct[i]
            else:
                for n, i in enumerate(index):
                    if i.__class__ is list:
                        k = data[n]
                        if hasattr(k, '__iter__'):
                            for j, m in zip(i, k):
                                if m: dct[j] = m
                                elif j in dct: del dct[j]
                        else:
                            if k:
                                k = float(k)
                                for j in i: dct[j] = k
                            else:
                                for j in i:
                                    if j in dct: del dct[j]  
                    else:
                        j = data[n]
                        if j: dct[i] = float(j)
                        elif i in dct: del dct[i]
        elif kind == 3:
            if ndim == 0:
                if data:
                    data = float(data)
                    for i in index: dct[i] = data
                else:
                    for i in index:
                        if i in dct: del dct[i]
            elif ndim == 1:
                for i, j in zip(index, data):
                    if j: dct[i] = float(j)
                    elif i in dct: del dct[i]
            else:
                raise IndexError(
                    'cannot set an array element with a sequence'
                )
        else:
            raise IndexError('invalid index kind') 
                
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
            basic_info = f"{type(self).__name__}:\n"
        new_line = '\n'
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
    __slots__ = ('_chemicals', '_phase', '_data_cache')
    
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
        old_container = (old_data, self._data_cache)
        if container is None:
            self.data = data = SparseVector.from_size(chemicals.size)
            self._data_cache = {}
        else:
            data, self._data_cache = container
            self.data =  data
            data.clear()
        for CAS, value in zip(self._chemicals.CASs, old_data):
            if value: data.dct[chemicals.index(CAS)] = value
        self._chemicals = chemicals
        return old_container
    
    def __reduce__(self):
        return self.from_data, (self.data, self._phase, self._chemicals, False)
    
    def __getitem__(self, key):
        return get_sparse_chemical_data(self.data, *self._chemicals._get_index_and_kind(key))
    
    def __setitem__(self, key, data):
        set_sparse_chemical_data(
            self.data, *self._chemicals._get_index_and_kind(key), 
            data, key, self
        )
    
    def sum_across_phases(self):
        return self.data
    
    @property
    def get_index(self):
        return self._chemicals.get_index
    
    def mix_from(self, others):
        set_main_phase(self, others)
        chemicals = self._chemicals
        data = self.data
        sc_data = [] # Same chemicals
        other_data = [] # Different chemicals
        isa = isinstance
        for i in others:
            ichemicals = i._chemicals
            idata = i.data
            if isa(i, MaterialIndexer):
                if ichemicals is chemicals:
                    sc_data.extend(idata.rows)
                else:
                    idata = idata.sum(0)
                    sc_data.append(idata)
                    other_data.append(
                        (i, *index_overlap(chemicals, ichemicals, idata.nonzero_keys()))
                    )
            elif ichemicals is chemicals:
                sc_data.append(idata)
            else:
                other_data.append(
                    (idata, *index_overlap(chemicals, ichemicals, idata.nonzero_keys()))
                )
        data.mix_from(sc_data)
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
        phase = self.phase
        if phase not in phases: 
            if phase.isupper():
                phase = phase.lower()
            else:
                phase = phase.upper()
        material_array[phase].copy_like(self.data)
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
        new._data_cache = {}
        return new
    
    @classmethod
    def blank(cls, phase, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self.data = SparseVector.from_size(chemicals.size)
        self._phase = Phase.convert(phase)
        self._data_cache = {}
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
        self._data_cache = {}
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
        beginning = f'({self.phase}) ' if self.phase else " "
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
            for phase, ID_data in phase_data.items():
                IDs, data = zip(*ID_data)
                self[phase, IDs] = data
            if units: self.set_data(data, units)
        return self
    
    def reset_chemicals(self, chemicals, container=None):
        old_data = self.data
        old__data_cache = self._data_cache
        N_phases = len(self._phases)
        if container is None:
            self.data = data = SparseArray.from_shape([N_phases, chemicals.size])
            self._data_cache = {}
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
        return (old_data, old__data_cache)
    
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
        phase_indexer = self._phase_indexer
        if isinstance(other, ChemicalIndexer):
            self.empty()
            other_data = other.data
            phase = other.phase
            if phase not in phase_indexer: self._expand_phases(phase)
            phase_index = phase_indexer(phase)
            if self.chemicals is other.chemicals:
                self.data.rows[phase_index].copy_like(other_data)
            else:
                other_data = other.data
                left_index, right_index = index_overlap(self._chemicals, other._chemicals, [*other_data.nonzero_keys()])
                self.data.rows[phase_index][left_index] = other_data[right_index] 
        else:
            other_phase_indexer = other._phase_indexer
            if self.chemicals is other.chemicals:
                if phase_indexer is other_phase_indexer:
                    self.data.copy_like(other.data)
                elif phase_indexer.compatible_with(other_phase_indexer):
                    self.empty()
                    data = self.data
                    for i, j in other: data[phase_indexer(i)] = j
                else:
                    self._expand_phases(other._phases)
                    self.data.copy_like(other.data)
            else:
                self.empty()
                other_data = other.data
                data = self.data
                left_index, right_index = index_overlap(self._chemicals, other._chemicals, [*other_data.nonzero_keys()])
                if phase_indexer is other_phase_indexer:
                    data[:, left_index] = other_data[:, right_index]
                elif phase_indexer.compatible_with(other_phase_indexer):
                    for i, j in other: data[phase_indexer(i)] += j
                else:
                    self._expand_phases(other._phases)
                    data[:, left_index] = other_data[:, right_index]
                    
    
    def _expand_phases(self, other_phases=None):
        phases = self._phases
        other_phases = set(other_phases)
        new_phases = other_phases.difference(phases)
        if new_phases: 
            data = self.data
            data_by_phase = {i: j for i, j in zip(phases, data.rows)}
            all_phases = new_phases.union(phases)
            self._set_phases(all_phases)
            size = self._chemicals.size
            for i in new_phases: data_by_phase[i] = SparseVector.from_size(size)
            phases = self._phases
            data.rows = [data_by_phase[i] for i in phases]
            self._set_cache()
            
    def mix_from(self, others):
        isa = isinstance
        chemicals = self._chemicals
        material_indexers = []
        chemical_indexers = []
        for i in others:
            if isa(i, MaterialIndexer): material_indexers.append(i)
            elif isa(i, ChemicalIndexer): chemical_indexers.append(i)
            else: raise ValueError("can only mix from chemical or material indexers")
        other_phases = [i.phase for i in chemical_indexers]
        for i in material_indexers: other_phases.extend(i._phases)
        other_phases = set(other_phases)
        phase_indexer = self._phase_indexer
        new_phases = [i for i in other_phases if i not in phase_indexer]
        phases = self._phases
        if new_phases: self._expand_phases(other_phases)
        scp_data = {i: [] for i in phases} # Same chemicals by phase
        dcp_data = {i: [] for i in phases} # Different chemicals by phase
        for i in other_phases.difference(phases):
            if i.isupper():
                ilow = i.lower()
                scp_data[i] = scp_data[ilow]
                dcp_data[i] = dcp_data[ilow]
            else:
                iup = i.upper()
                scp_data[i] = scp_data[iup]
                dcp_data[i] = dcp_data[iup]
        for i in material_indexers:
            ichemicals = i._chemicals
            idata = i.data
            if chemicals is ichemicals:
                for i, j in zip(i._phases, idata.rows):
                    scp_data[i].append(j)
            else:
                left_index, right_index = index_overlap(chemicals, ichemicals, idata.nonzero_keys())
                for i, j in zip(i._phases, i.data.rows):
                    dcp_data[i].append((j, left_index, right_index))
        for i in chemical_indexers:
            ichemicals = i._chemicals
            idata = i.data
            if chemicals is ichemicals:
                scp_data[i.phase].append(idata)
            else:
                dcp_data[i.phase].append((idata, *index_overlap(chemicals, ichemicals, idata.nonzero_keys())))
        for phase, sv in zip(phases, self.data.rows):
            sv.mix_from(scp_data[phase])
            for idata, left_index, right_index in dcp_data[phase]:
                sv[left_index] += idata[right_index]
    
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
                    other_index, = idata.any(0).nonzero()
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
                        other_index, = idata.nonzero()
                        CASs = other.chemicals.CASs
                        self_index = chemicals.indices([CASs[i] for i in other_index])
                        data[get_phase_index(phase), self_index] -= idata[other_index]
        elif isa(other, ChemicalIndexer):
            if chemicals is other.chemicals:
                data[get_phase_index(other.phase), :] -= idata
            else:
                other_index, = idata.nonzero()
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
        new._data_cache = {}
        return new
    
    @classmethod
    def blank(cls, phases, chemicals=None):
        self = _new(cls)
        self._load_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        self.data = SparseArray.from_shape([len(phases), self._chemicals.size])
        self._data_cache = {}
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
        self._data_cache = {}
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
            if data.any(): 
                if phase not in phases:
                    if phase.isupper():
                        phase = phase.lower()
                    else:
                        phase = phase.upper()
                material_indexer[phase] += data
        return material_indexer
    
    def get_phase(self, phase):
        return self._ChemicalIndexer.from_data(self.data.rows[self.get_phase_index(phase)],
                                               LockedPhase(phase), self._chemicals, False)
    
    def __getitem__(self, key):
        index, kind, sum_across_phases = self._get_index_data(key)
        if sum_across_phases:
            dcts = [i.dct for i in self.data.rows]
            if kind == 0: # Chemical
                values = sum([i[index] for i in dcts if index in i])
            elif kind == 1: # Chemical group
                values = sum([j[i] for i in index for j in dcts if i in j])
            elif kind == 2: # Nested chemical group
                values = np.array([
                    (sum([dct[j] for j in i for dct in dcts if j in dct]) 
                     if i.__class__ is list 
                     else sum([dct[i] for dct in dcts if i in dct]))
                    for n, i in enumerate(index)
                ])
            elif kind == 3: # List
                values = np.array([sum([dct[i] for dct in dcts if i in dct]) for i in index])
            elif kind is None:
                values = self.data.sum(0)
            else:
                raise IndexError('invalid index kind')
        else:
            if kind is None:
                values = self.data if index is None else self.data.rows[index]
            else:
                phase_index, chemical_index = index
                if phase_index is None:
                    values = np.array([
                        get_sparse_chemical_data(i, chemical_index, kind) for i in self.data.rows
                    ])
                else:
                    phase_index, chemical_index = index
                    values = get_sparse_chemical_data(self.data.rows[phase_index], chemical_index, kind)
        return values
    
    def __setitem__(self, key, data):
        index, kind, sum_across_phases = self._get_index_data(key)
        if sum_across_phases:
            raise IndexError("multiple phases present; must include phase key "
                             "to set chemical data")
        if kind is None:
            if index is None:
                self.data[:] = data
            else:
                reset_sparse_chemical_data(self.data.rows[index], data)
        else:
            phase_index, chemical_index = index
            _, key = key
            if phase_index is None:
                if kind in (0, 3):
                    self.data[:, chemical_index] = data
                elif kind == 1: # Chemical group
                    phase, index = index
                    composition = self.group_compositions[key]
                    self.data[:, chemical_index] = data * composition
                elif kind == 2: # Nested chemical group
                    phase, index = index
                    sparse_data = self.data
                    group_compositions = self.group_compositions
                    for n, i in enumerate(index):
                        sparse_data[:, i] = data[n] * group_compositions[key[n]] if i.__class__ is list else data[n]
                else:
                    raise IndexError('invalid index kind')
            else:
                set_sparse_chemical_data(
                    self.data[phase_index], chemical_index, kind, 
                    data, key, self
                )
    
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
                kind = None
            else:
                raise undefined_chemical_error
        elif phase_IDs is ...:
            phase_index = index = kind = None
        else:
            phase = phase_IDs[0]
            if isa(phase, str):
                if len(phase) == 1:
                    phase_index = self.get_phase_index(phase)
                else:
                    raise undefined_chemical_error
            elif phase is ...:
                phase_index = None
            else:
                raise_material_indexer_index_error()
            try:
                phase, IDs = phase_IDs
            except:
                raise_material_indexer_index_error()
            chemical_index, kind = self._chemicals._get_index_and_kind(IDs)
            index = (phase_index, chemical_index)
        return index, kind, 
    
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
        index, = self.data.any(0).nonzero()
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
            beginning = f'({phase}) '
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
        mass = self._data_cache['mass']
    except:
        chemicals = self.chemicals
        self._data_cache['mass'] = mass = \
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
        mass = self._data_cache['mass']
    except:
        chemicals = self.chemicals
        size = chemicals.size
        MW = chemicals.MW
        self._data_cache['mass'] = mass = \
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
        vol = self._data_cache['vol', TP]
    except:
        chemicals = self._chemicals
        V = [i.V for i in chemicals]
        phase = self._phase
        self._data_cache['vol', TP] = \
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
        vol = self._data_cache[TP]
    except:
        phases = self._phases
        chemicals = self._chemicals
        V = [i.V for i in chemicals]
        size = chemicals.size
        self._data_cache[TP] = \
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

