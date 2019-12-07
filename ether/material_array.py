# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:41:50 2019

@author: yoelr
"""

from .base import UnitsConverter
from .settings import settings
from .exceptions import UndefinedPhase
from free_properties import PropertyFactory, property_array
import numpy as np

__all__ = ('ChemicalArray',
           'PhaseArray',
           'MaterialArray',
           'ChemicalMolarFlow', 
           'PhaseMolarFlow',
           'MolarFlow',
           'ChemicalMassFlow', 
           'PhaseMassFlow',
           'MassFlow',
           'ChemicalVolumetricFlow',
           'PhaseVolumetricFlow',
           'VolumetricFlow',
           'MassFlowProperty',
           'VolumetricFlowProperty')

# %% Utilities

isa = isinstance
_new = object.__new__

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]


# %% Abstract data emulator
    
class ArrayEmulator:
    __slots__ = ('_data', '_cached_index')
    _cached_phase_index = {}
    chemicals = units = None
    _units_converter = UnitsConverter()
    
    def copy(self, data=False):
        if data:
            return self._data.copy()
        else:
            new = self._copy_without_data()
            new._data = self._data.copy()
            return new
    
    def get_data(self, units, *index):
        length = len(index)
        factor = self._units_converter(units)
        if length == 0:
            return factor * self._data
        elif length == 1:
            return factor * self[index[0]]
        else:
            return factor * self[index]
    
    def set_data(self, data, units, *index):
        length = len(index)
        scaled_data = data / self._units_converter(units)
        if length == 0:
            self._data[:] = scaled_data
        elif length == 1:
            self._data[index[0]] = scaled_data
        else:
            self[index] = scaled_data
    
    def _get_index(self, key):
        cache = self._cached_index
        try: 
            index = cache[key]
        except KeyError: 
            cache[key] = index = self.get_data_index(key)
        except TypeError:
            raise IndexError(f"index must be hashable")
        return index
    
    def __getitem__(self, key):
        return self._data[self._get_index(key)]
    
    def __setitem__(self, key, data):
        self._data[self._get_index(key)] = data
    
    @property
    def data(self):
        return self._data
    @property
    def all(self):
        return self._data.all
    @property
    def any(self):
        return self._data.any
    @property
    def argmax(self):
        return self._data.argmax
    @property
    def argmin(self):
        return self._data.argmin
    @property
    def argpartition(self):
        return self._data.argpartition
    @property
    def argsort(self):
        return self._data.argsort
    @property
    def astype(self):
        return self._data.astype
    @property
    def byteswap(self):
        return self._data.byteswap
    @property
    def choose(self):
        return self._data.choose
    @property
    def clip(self):
        return self._data.clip
    @property
    def compress(self):
        return self._data.compress
    @property
    def conj(self):
        return self._data.conj
    @property
    def conjugate(self):
        return self._data.conjugate
    @property
    def cumprod(self):
        return self._data.cumprod
    @property
    def cumsum(self):
        return self._data.cumsum
    @property
    def diagonal(self):
        return self._data.diagonal
    @property
    def dot(self):
        return self._data.dot
    @property
    def fill(self):
        return self._data.fill
    @property
    def flatten(self):
        return self._data.flatten
    @property
    def getfield(self):
        return self._data.getfield
    @property
    def item(self):
        return self._data.item
    @property
    def itemset(self):
        return self._data.itemset
    @property
    def max(self):
        return self._data.max
    @property
    def mean(self):
        return self._data.mean
    @property
    def min(self):
        return self._data.min
    @property
    def newbyteorder(self):
        return self._data.newbyteorder
    @property
    def nonzero(self):
        return self._data.nonzero
    @property
    def partition(self):
        return self._data.partition
    @property
    def prod(self):
        return self._data.prod
    @property
    def ptp(self):
        return self._data.ptp
    @property
    def put(self):
        return self._data.put
    @property
    def ravel(self):
        return self._data.ravel
    @property
    def repeat(self):
        return self._data.repeat
    @property
    def reshape(self):
        return self._data.reshape
    @property
    def resize(self):
        return self._data.resize
    @property
    def round(self):
        return self._data.round
    @property
    def searchsorted(self):
        return self._data.searchsorted
    @property
    def setfield(self):
        return self._data.setfield
    @property
    def setflags(self):
        return self._data.setflags
    @property
    def sort(self):
        return self._data.sort
    @property
    def squeeze(self):
        return self._data.squeeze
    @property
    def std(self):
        return self._data.std
    @property
    def sum(self):
        return self._data.sum
    @property
    def swapaxes(self):
        return self._data.swapaxes
    @property
    def take(self):
        return self._data.take
    @property
    def tobytes(self):
        return self._data.tobytes
    @property
    def tofile(self):
        return self._data.tofile
    @property
    def tolist(self):
        return self._data.tolist
    @property
    def tostring(self):
        return self._data.tostring
    @property
    def trace(self):
        return self._data.trace
    @property
    def transpose(self):
        return self._data.tranpose
    @property
    def var(self):
        return self._data.var
    @property
    def view(self):
        return self._data.view
    @property
    def ndim(self):
        return self._data.ndim
    @property
    def flags(self):
        return self._data.flags
    @property
    def shape(self):
        return self._data.shape
    @property
    def strides(self):
        return self._data.strides
    @property
    def itemsize(self):
        return self._data.itemsize
    @property
    def size(self):
        return self._data.size
    @property
    def nbytes(self):
        return self._data.nbytes
    @property
    def base(self):
        return self._data.base
    @property
    def dtype(self):
        return self._data.dtype
    @property
    def real(self):
        return self._data.real
    @property
    def imag(self):
        return self._data.imag
    @property
    def flat(self):
        return self._data.flat
    @property
    def ctypes(self):
        return self._data.ctypes
    @property
    def T(self):
        return self._data.T
    
    def __iter__(self):
        return self._data.__iter__()
    
    def __lt__(self, other):
        return self._data < (other._data if isa(other, ArrayEmulator) else other)
    
    def __le__(self, other):
        return self._data <= (other._data if isa(other, ArrayEmulator) else other)
    
    def __eq__(self, other):
        return self._data == (other._data if isa(other, ArrayEmulator) else other)
    
    def __ne__(self, other):
        return self._data != (other._data if isa(other, ArrayEmulator) else other)
    
    def __gt__(self, other):
        return self._data > (other._data if isa(other, ArrayEmulator) else other)
    
    def __ge__(self, other):
        return self._data >= (other._data if isa(other, ArrayEmulator) else other)
    
    def __add__(self, other):
        return self._data + (other._data if isa(other, ArrayEmulator) else other)
    
    def __sub__(self, other):
        return self._data - (other._data if isa(other, ArrayEmulator) else other)
    
    def __mul__(self, other):
        return self._data * (other._data if isa(other, ArrayEmulator) else other)
    
    def __matmul__(self, other):
        return self._data @ (other._data if isa(other, ArrayEmulator) else other)
    
    def __truediv__(self, other):
        return self._data / (other._data if isa(other, ArrayEmulator) else other)
    
    def __floordiv__(self, other):
        return self._data // (other._data if isa(other, ArrayEmulator) else other)
    
    def __mod__(self, other):
        return self._data % (other._data if isa(other, ArrayEmulator) else other)
    
    def __divmod__(self, other):
        return self._data.__divmod__(other._data if isa(other, ArrayEmulator) else other)
    
    def __pow__(self, other):
        return self._data ** (other._data if isa(other, ArrayEmulator) else other)
    
    def __lshift__(self, other):
        return self._data << (other._data if isa(other, ArrayEmulator) else other)
        
    def __rshift__(self, other):
        return self._data >> (other._data if isa(other, ArrayEmulator) else other)
    
    def __and__(self, other):
        return self._data & (other._data if isa(other, ArrayEmulator) else other)
    
    def __xor__(self, other):
        return self._data ^ (other._data if isa(other, ArrayEmulator) else other)
    
    def __or__(self, other):
        return self._data | (other._data if isa(other, ArrayEmulator) else other)
    
    def __radd__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) + self._data
        
    def __rsub__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) - self._data
    
    def __rmul__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) * self._data
        
    def __rmatmul__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) @ self._data 
    
    def __rtruediv__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) / self._data
        
    def __rfloordiv__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) // self._data
        
    def __rmod__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) % self._data
        
    def __rdivmod__(self, other):
        return self._data.__rdivmod__(other._data if isa(other, ArrayEmulator) else other)
    
    def __rpow__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) ** self._data
    
    def __rlshift__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) << self._data
    
    def __rrshift__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) >> self._data
    
    def __rand__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) & self._data
    
    def __rxor__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) ^ self._data
    
    def __ror__(self, other):
        return (other._data if isa(other, ArrayEmulator) else other) | self._data

    def __iadd__(self, other):
        self._data.__iadd__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __isub__(self, other):
        self._data.__isub__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __imul__(self, other):
        self._data.__imul__(other._data if isa(other, ArrayEmulator) else other)
        return self
    
    def __imatmul__(self, other):
        raise TypeError("in-place matrix multiplication is not (yet) supported")
    
    def __itruediv__(self, other):
        self._data.__itruediv__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __ifloordiv__(self, other):
        self._data.__ifloordiv__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __imod__(self, other):
        self._data.__imod__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __ipow__(self, other):
        self._data.__ipow__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __ilshift__(self, other):
        self._data.__ilshift__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __irshift__(self, other):
        self._data.__irshift__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __iand__(self, other):
        self._data.__iand__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __ixor__(self, other):
        self._data.__ixor__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __ior__(self, other):
        self._data.__ior__(other._data if isa(other, ArrayEmulator) else other)
        return self
        
    def __neg__(self):
        return -self._data
        
    def __pos__(self):
        return self._data
        
    def __abs__(self):
        return self._data.abs()


# %% Phase data

class ChemicalArray(ArrayEmulator):
    __slots__ = ('_chemicals', 'phase')
    _index_caches = {}
    
    def __new__(cls, phase, chemicals=None, **IDdata):
        if IDdata:
            chemicals = settings.get_chemicals(chemicals)
            cls.from_data(chemicals.array(IDdata), phase, chemicals)
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
            self._cached_index = caches[self._chemicals]
        except KeyError:
            self._cached_index = caches[self._chemicals] = {}
    
    def to_phase_array(self, phases=(), data=False):
        if data:
            array = np.zeros_like(phases, dtype=float)
            array[phases.index(self.phase)] = self._data.sum()
            return array
        else:
            phase_array = self._PhaseArray.blank(phases)
            phase_array[self.phase] = self._data.sum()
            return phase_array
        
    def to_material_array(self, phases=(), data=False):
        if data:
            size = (len(phases), self._chemicals.size)
            array = np.zeros(size, dtype=float)
            array[phases.index(self.phase)] = self._data
            return array
        else:
            material_array = self._MaterialArray.blank(phases, self._chemicals)
            material_array[self.phase] = self._data
            return material_array
    
    def composition(self, data=False):
        array = self._data
        total = array.sum()
        composition = array/total if total else array.copy()
        if data:
            return composition
        else:
            return ChemicalArray.from_data(composition,
                                           self.phase,
                                           self._chemicals)
    
    def copy(self):
        new = self._copy_without_data()
        new._data = self._data.copy()
        return new
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._chemicals = self._chemicals
        new._cached_index = self._cached_index
        new.phase = self.phase
        return new
    
    @classmethod
    def blank(cls, phase, chemicals=None):
        self = _new(cls)
        self._set_chemicals(chemicals)
        self._set_cache()
        self._data = np.zeros(self._chemicals.size, float)
        self.phase = phase
        return self
    
    @classmethod
    def from_data(cls, data, phase, chemicals=None):
        self = _new(cls)
        self._set_chemicals(chemicals)
        self._set_cache()
        self.phase = phase
        if settings._debug:
            assert isa(data, np.ndarray) and data.ndim == 1, (
                                                    'data must be an 1d numpy array')
            assert data.size == self._chemicals.size, ('size of data must be equal to '
                                                       'size of chemicals')
        self._data = data
        return self
    
    @property
    def chemicals(self):
        return self._chemicals
        
    def get_data_index(self, IDs):
        if isa(IDs, str):
            return self._chemicals.index(IDs)
        elif isa(IDs, slice):
            return IDs
        else:
            return self._chemicals.indices(IDs)
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        IDdata = [f"{ID}={i:.4g}" for ID, i in zip(self._chemicals.IDs, self._data) if i]
        phase = f"phase={repr(self.phase)}"
        if len(IDdata) > 1 and tab:
            dlim = ",\n" + tab
            phase = '\n' + tab + phase
        else:
            dlim = ", "
        IDdata = dlim.join(IDdata)
        return f"{type(self).__name__}({phase}{dlim}{IDdata})"
    
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
      
    
class PhaseArray(ArrayEmulator):
    __slots__ = ('_phases', '_phase_index')
    _index_caches = {}
    
    def __new__(cls, phases, **phase_data):
        self = cls.blank(phases or phase_data)
        if phase_data:
            phases, data = zip(*phase_data)
            self._data[self._get_phase_indices(phases)] = data
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases)
    
    
    def to_chemical_array(self, composition, phase=None, chemicals=None, data=False):
        if settings._debug:
            assert isa(composition, np.ndarray), "composition must be 1d array"
            assert composition.ndim == 1, "composition must be 1d array"
            assert composition.sum() == 1., "compositions must be normalized"
        if data:
            return self._data.sum() * composition
        else:
            return self._ChemicalArray.from_data(self._data.sum() * composition,
                                                  phase,         
                                                  chemicals)
    
    def to_material_array(self, compositions, chemicals=None, data=False):
        if settings._debug:
            assert isa(compositions, np.ndarray), "compositions must be 2d array"
            assert compositions.ndim == 2, "compositions must be 2d array"
            assert compositions.sum() == 1., "compositions must be normalized"
        if data:
            return self._data[:, np.newaxis] * compositions
        else:
            return self._MaterialArray.from_data(self._data[:, np.newaxis] * compositions,
                                                  self._phases, chemicals)
    
    @property
    def phases(self):
        return self._phases
    
    def phase_split(self, data=False):
        array = self._data
        total = array.sum()
        phase_split = array/total if total else array.copy()
        if data:
            return phase_split
        else:
            return PhaseArray.from_data(phase_split, self._phases)
    
    def _set_phases(self, phases):
        self._phases = phases = tuple(phases)
        cached = self._cached_phase_index
        if phases in cached:
            self._phase_index = cached[phases]
        else:
            self._phase_index = cached[phases] = {j:i for i,j in enumerate(phases)}
    
    def _set_cache(self):
        caches = self._index_caches
        try:
            self._cached_index = caches[self._phases]
        except KeyError:
            self._cached_index = caches[self._phases] = {}
    
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
        new._cached_index = self._cached_index
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
    
    def get_data_index(self, phases):
        if len(phases) == 1:
            return self._get_phase_index(phases)
        elif isa(phases, slice):
            return phases
        else:
            return self._get_phase_indices(phases)
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        phase_data = [f"{phase}={i:.4g}" for phase, i in zip(self.phases, self.data) if i]
        if len(phase_data) > 1 and tab:
            dlim = ",\n" + tab
            phase_data = "\n" + tab + dlim.join(phase_data)
        else:
            phase_data = ', '.join(phase_data)
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
            

class MaterialArray(ArrayEmulator):
    __slots__ = ('_chemicals', '_phases', '_phase_index', '_phase_data')
    _index_caches = {}
    _ChemicalArray = ChemicalArray
    _PhaseArray = PhaseArray
    
    def __new__(cls, phases=None, chemicals=None, **phase_data):
        self = cls.blank(phases or phase_data, chemicals)
        if phase_data:
            data = self._data
            chemical_indices = self._chemicals.indices
            phase_index = self._get_phase_index
            for phase, IDdata in phase_data.items():
                IDs, row = zip(*IDdata)
                data[phase_index(phase), chemical_indices(IDs)] = row
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases, self._chemicals)
    
    def composition(self, data=False):
        array = self._data
        total = array.sum(keepdims=True)
        composition = array/total if total else array.copy()
        if data:
            return composition
        else:
            return MaterialArray.from_data(composition, self._phases, self._chemicals)
    
    def phase_composition(self, data=False):
        array = self._data
        phase_array = array.sum(1, keepdims=True)
        phase_array[phase_array == 0] = 1.
        if data:
            return phase_array
        else:
            return MaterialArray.from_data(array/phase_array,
                                           self._phases, self._chemicals)
    
    def phase_split(self, data=False):
        array = self._data
        phase_array = array.sum(0, keepdims=True)
        phase_array[phase_array == 0] = 1.
        if data:
            return phase_array
        else:
            return MaterialArray.from_data(array/phase_array,
                                           self._phases, self._chemicals)
    
    _set_chemicals = ChemicalArray._set_chemicals
    _set_phases = PhaseArray._set_phases
    def _set_cache(self):
        caches = self._index_caches
        key = self._phases, self._chemicals
        try:
            self._cached_index = caches[key]
        except KeyError:
            self._cached_index = caches[key] = {}
    
    def _copy_without_data(self):
        new = _new(self.__class__)
        new._chemicals = self._chemicals
        new._phases = self._phases
        new._phase_index = self._phase_index
        new._cached_index = self._cached_index
        return new
    
    @classmethod
    def blank(cls, phases, chemicals=None):
        self = _new(cls)
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        self._set_cache()
        shape = (len(self._phases), self._chemicals.size)
        self._data = np.zeros(shape, float)
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
        return self
    
    phases =  PhaseArray.phases
    chemicals = ChemicalArray.chemicals
    
    def to_phase_array(self, data=False):
        if data:
            return self._data.sum(1)
        else:
            return self._PhaseArray.from_data(self._data.sum(1), self._phases)
    
    def to_chemical_array(self, phase=None, data=False):
        if data:
            return self._data.sum(0)
        else:
            return self._ChemicalArray.from_data(self._data.sum(0), phase, self._chemicals)
    
    def get_phase(self, phase):
        return self._ChemicalArray.from_data(self._data[self._phase_index[phase]],
                                             phase, self._chemicals)
    
    def get_data_index(self, phase_IDs):
        if isa(phase_IDs, str):
            return self._get_phase_index(phase_IDs)
        elif isa(phase_IDs, slice) or phase_IDs == ...:
            return phase_IDs 
        else:
            phase, IDs = phase_IDs
            if isa(IDs, str):
                IDs_index = self._chemicals.index(IDs)
            elif isa(IDs, slice) or IDs == ...:
                IDs_index = IDs
            else:
                IDs_index = self._chemicals.indices(IDs)
            if isa(phase, slice) or phase == ...:
                return (phase, IDs_index)
            else:
                return (self._get_phase_index(phase), IDs_index)
            
    _get_phase_index = PhaseArray._get_phase_index
    
    @property
    def phase_data(self):
        try:
            phase_data = self._phase_data
        except:
            self._phase_data = phase_data = tuple(zip(self._phases, self._data))
        return phase_data
    
    def __format__(self, tabs="1"):
        IDs = self._chemicals.IDs
        phase_data = []
        for phase, data in self.phase_data:
            IDdata = ", ".join([f"('{ID}', {i:.4g})" for ID, i in zip(IDs, data) if i])
            if IDdata:
                phase_data.append(f"{phase}=[{IDdata}]")
        tabs = int(tabs)
        if tabs:
            tab = tabs*4*" "
            dlim = ",\n" + tab 
        else:
            dlim = ", "
        phase_data = dlim.join(phase_data)
        if self.to_phase_array(data=True).all():
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
        index, = np.where(self.to_chemical_array(data=True) != 0)
        len_ = len(index)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n"
        else:
            basic_info = f"{type(self).__name__}:\n"
        all_IDs = tuple([IDs[i] for i in index])

        # Length of species column
        all_lengths = [len(i) for i in IDs]
        maxlen = max(all_lengths + [8])  # include length of the word 'species'

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
    show = ChemicalArray.show
    _ipython_display_ = show
    
def new_Array(name, units):
    ChemicalArraySubclass = type('Chemical' + name, (ChemicalArray,), {})
    PhaseArraySubclass = type('Phase' + name, (PhaseArray,), {})
    MaterialArraySubclass = type(name, (MaterialArray,), {})
    
    ChemicalArraySubclass.__slots__ = \
    PhaseArraySubclass.__slots__ = \
    MaterialArraySubclass.__slots__ = ()
    
    ChemicalArraySubclass.units = \
    PhaseArraySubclass.units = \
    MaterialArraySubclass.units = units
    
    ChemicalArraySubclass._units_converter = \
    PhaseArraySubclass._units_converter = \
    MaterialArraySubclass._units_converter = UnitsConverter(units)
    
    PhaseArraySubclass._ChemicalArray = \
    MaterialArraySubclass._ChemicalArray = ChemicalArraySubclass
    
    MaterialArraySubclass._PhaseArray = \
    ChemicalArraySubclass._PhaseArray = PhaseArraySubclass
    
    PhaseArraySubclass._MaterialArray = \
    ChemicalArraySubclass._MaterialArray = MaterialArraySubclass
    
    return ChemicalArraySubclass, PhaseArraySubclass, MaterialArraySubclass

ChemicalArray._MaterialArray = MaterialArray
ChemicalArray._PhaseArray = PhaseArray
PhaseArray._ChemicalArray = ChemicalArray
PhaseArray._MaterialArray = MaterialArray    
ChemicalMolarFlow, PhaseMolarFlow, MolarFlow = new_Array('MolarFlow', 'kmol/hr')
ChemicalMassFlow, PhaseMassFlow, MassFlow = new_Array('MassFlow', 'kg/hr')
ChemicalVolumetricFlow, PhaseVolumetricFlow, VolumetricFlow = new_Array('VolumetricFlow', 'm^3/hr')


# %% Mass flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'MW'))
def MassFlowProperty(self):
    """Mass flow (kg/hr)."""
    return self.molar_flow[self.index] * self.MW
    
@MassFlowProperty.setter
def MassFlowProperty(self, value):
    self.molar_flow[self.index] = value/self.MW

def as_chemical_mass_flow(self):
    chemicals = self.chemicals
    molar_flow = self.data
    mass_flow = np.zeros_like(molar_flow, dtype=object)
    for i, chem in enumerate(chemicals):
        mass_flow[i] = MassFlowProperty(chem.ID, molar_flow, i, chem.MW)
    return ChemicalMassFlow.from_data(mass_flow, chemicals)
ChemicalMolarFlow.as_chemical_mass_flow = as_chemical_mass_flow
del as_chemical_mass_flow    

def as_mass_flow(self):
    phases = self.phases
    chemicals = self.chemicals
    molar_flow = self.data
    mass_flow = np.zeros_like(molar_flow, dtype=object)
    for i, phase in enumerate(phases):
        for j, chem in enumerate(chemicals):
            index = (i, j)
            mass_flow[index] = MassFlowProperty(chem.ID, molar_flow, index, chem.MW)
    return MassFlow.from_data(mass_flow, phases, chemicals)
MolarFlow.as_mass_flow = as_mass_flow
del as_mass_flow


# %% Volumetric flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'V', 'thermal_condition', 'phase'))
def VolumetricFlowProperty(self):
    """Volumetric flow (m^3/hr)."""
    T, P = self.thermal_condition
    return self.molar_flow[self.index] * self.V(self.phase or self.molar_flow.phase, T, P)
    
@VolumetricFlowProperty.setter
def VolumetricFlowProperty(self, value):
    T, P = self.thermal_condition
    self.molar_flow[self.index] = value / self.V(self.phase or self.molar_flow.phase, T, P)

def as_chemical_volumetric_flow(self, thermal_condition):
    chemicals = self.chemicals
    molar_flow = self.data
    volumetric_flow = np.zeros_like(molar_flow, dtype=object)
    for i, chem in enumerate(chemicals):
        volumetric_flow[i] = VolumetricFlowProperty(chem.ID, molar_flow, i, chem.V, thermal_condition, None)
    return ChemicalVolumetricFlow(property_array(volumetric_flow), chemicals)
ChemicalMolarFlow.as_chemical_volumetric_flow = as_chemical_volumetric_flow
del as_chemical_volumetric_flow
	
def as_volumetric_flow(self, thermal_condition):
    phases = self.phases
    chemicals = self.chemicals
    molar_flow = self.data
    volumetric_flow = np.zeros_like(molar_flow, dtype=object)
    for i, phase in enumerate(phases):
        for j, chem in enumerate(chemicals):
            index = i, j
            volumetric_flow[index] = VolumetricFlowProperty(f"[{phase}] {chem.ID}", molar_flow, index, chem.V, thermal_condition, phase)
    return VolumetricFlow.from_data(property_array(volumetric_flow), phases, chemicals)
MolarFlow.as_volumetric_flow = as_volumetric_flow
del as_volumetric_flow

# %% Cut out functionality

