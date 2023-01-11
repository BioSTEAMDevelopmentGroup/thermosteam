# -*- coding: utf-8 -*-
"""
"""
import numpy as np
__all__ = (
    'nonzero_items',
    'sparse_vector',
    'sparse_array',
    'SparseVector',
    'SparseArray',
    'sparse',
)

bools = (bool, np.bool_)
open_slice = slice(None)

def unpack_index(index, ndim):
    indim = index.__len__()
    if indim == ndim:
        return index
    elif indim < ndim:
        raise IndexError(
            f'too few indices for array; array is {ndim}-dimensional, '
            f'but {indim} were indexed'
        )
    else:
        raise IndexError(
            f'too many indices for array; array is {ndim}-dimensional, '
            f'but {indim} were indexed'
        )

def sparse_vector(arr, copy=False, size=None):
    """
    Convert 1-d array to a SparseVector object.

    """
    if arr.__class__ is SparseVector:
        return arr.copy() if copy else arr
    else:
        return SparseVector(arr, size)

def nonzero_items(arr):
    if arr.__class__ is SparseVector:
        return arr.dct.items()
    else:
        return [(i, j) for i, j in enumerate(arr) if j]

def sparse_array(arr, copy=False, vector_size=None):
    """
    Convert 2-d array to a SparseArray object.

    """
    if arr.__class__ is SparseArray:
        return arr.copy() if copy else arr
    else:
        return SparseArray(arr, vector_size)

def sparse(arr, copy=False, vector_size=None):
    if arr.__class__ in (SparseArray, SparseVector):
        return arr
    elif (ndim:=get_ndim(arr)) == 1:
        return SparseVector(arr)
    elif ndim == 2:
        return SparseArray(arr)
    else:
        raise ValueError(f'cannot convert {ndim}-d object to a sparse array or vector')
    
def get_ndim(value):
    if hasattr(value, 'ndim'):
        return value.ndim
    elif hasattr(value, '__iter__'):
        for i in value: return get_ndim(i) + 1
    return 0

def get_index_properties(index):
    if hasattr(index, 'ndim'):
        return index.ndim, index.dtype in bools
    elif hasattr(index, '__iter__'):
        for i in index: 
            ndim, has_bool = get_index_properties(i)
            return ndim + 1, has_bool
        return 1, False
    else:
        return 0, index.__class__ in bools

def sum_sparse_vectors(svs, dct=None):
    if dct is None: dct = {}
    for other in svs:
        for i, j in other.dct.items():
            if i in dct:
                dct[i] += j
            else:
                dct[i] = j
    return dct

class SparseArray:
    """
    Create a SparseArray object that can be used for array-like arithmetic operations
    (i.e., +, -, *, /) of sparse 2-dimensional arrays. 
    
    In contrast to Scipy's sparse 2-d arrays, sparse arrays do not have a strict row length 
    (but still have a strict column length). 
    
    """
    __slots__ = ('rows', '_base')
    ndim = 2
    dtype = float
    
    def __init__(self, obj=None, vector_size=None):
        if obj is None:
            self.rows = []
        elif hasattr(obj, '__iter__'):
            self.rows = [sparse_vector(row, vector_size) for row in obj]
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse array')
    
    @classmethod
    def from_shape(cls, shape):
        m, n = shape
        return cls.from_rows([SparseVector.from_size(n) for i in range(m)])
    
    @classmethod
    def from_rows(cls, rows):
        new = cls.__new__(cls)
        new.rows = rows
        return new
    
    def clear(self):
        for i in self.rows: i.dct.clear()
    
    def copy(self):
        return SparseArray.from_rows([i.copy() for i in self.rows])
    
    def __bool__(self):
        raise ValueError(
            'the truth value of an array with more than one element is ambiguous; '
            'use any() or all() methods instead'
        )
    
    def __iter__(self):
        return self.rows.__iter__()
        
    def __len__(self):
        return self.rows.__len__()
    
    def __abs__(self):
        positive = abs
        rows = [row.copy() for row in self.rows]
        for row in rows:
            dct = row.dct
            for i in dct: dct[i] = positive(dct[i])
        return SparseArray(rows)
    
    def nonzero_index(self):
        m = []; n = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                m.append(i); n.append(j)
        return m, n
    nonzero = nonzero_index
    
    def nonzero_values(self):
        for row in self.rows:
            yield from row.values()
    
    def nonzero_items(self):
        for i, row in enumerate(self.rows):
            for j, k in row.dct.items(): yield ((i, j), k)
    
    def nonzero_rows(self):
        return [i for i, j in enumerate(self.rows) if j.dct]
    
    def nonzero_keys(self):
        keys = set()
        for i in self.rows: keys.update(i.dct)
        return keys
    
    def negative_index(self):
        m = []; n = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value < 0.:
                    m.append(i); n.append(j)
        return m, n
    
    def negative_rows(self):
        m = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value < 0.: 
                    m.append(i)
                    break
        return m
    
    def negative_keys(self):
        n = set()
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value < 0.: n.add(j)
        return n
    
    def has_negatives(self):
        for i, row in enumerate(self.rows):
            if row.has_negatives(): return True
        return False
    
    def remove_negatives(self):
        for i in self.rows: i.remove_negatives
    
    def shares_data_with(self, other):
        return bool(self.base.intersection(other.base))
    
    def to_flat_array(self, arr=None):
        vector_size = self.vector_size
        dtype = self.dtype
        rows = self.rows
        N = len(rows)
        if arr is None: arr = np.zeros(N * vector_size, dtype=dtype)
        for i in range(N):
            row = rows[i]
            for j, value in row.dct.items(): 
                arr[i * vector_size + j] = value
        return arr
        
    def from_flat_array(self, arr=None):
        rows = self.rows
        vector_size = self.vector_size
        dcts = [i.dct for i in rows]
        for i in dcts: i.clear()
        for j, value in enumerate(arr):
            if value: 
                i = int(j / vector_size)
                j -= i * vector_size
                dcts[i][j] = value
    
    @property
    def base(self):
        try:
            base = self._base
        except:
            self._base = base = frozenset([id(i.dct) for i in self.rows])
        return base
    
    def sparse_equal(self, other):
        return all([i.dct == j.dct for i, j in zip(self.rows, sparse_array(other).rows)])
    
    def __getitem__(self, index):
        rows = self.rows
        if index.__class__ is tuple:
            m, n = unpack_index(index, self.ndim)
            md = get_ndim(m)
            nd = get_ndim(n)
            if not md and m == open_slice:
                if not nd and n == open_slice:
                    return self
                else:
                    value = np.array([i[n] for i in rows])
                    value.setflags(0)
                    return value
            elif not nd and n == open_slice:
                if md == 0:
                    return rows[m]
                elif md == 1:
                    return SparseArray.from_rows([rows[i] for i in m]).to_array()
                else:
                    raise IndexError(f'row index can be at most 1-d, not {md}-d')
            elif md == 0: 
                return rows[m][n]
            elif md == 1: 
                if nd == 0:
                    n = n.__index__()
                    return np.array([rows[i].dct.get(n, 0.) for i in m])
                elif nd == 1:
                    return np.array([rows[i].dct.get(j.__index__(), 0.) for i, j in zip(m, n)])
                else:
                    raise IndexError(f'column index can be at most 1-d, not {nd}-d')
            else:
                raise IndexError(f'row index can be at most 1-d, not {md}-d')
        else:
            ndim, has_bool = get_index_properties(index)
            if has_bool: 
                if ndim != 2:
                    raise IndexError(
                        f'boolean index is {ndim}-d but sparse array is 2-d '
                    )
                return self[np.where(index)]
            elif ndim == 1:
                return SparseArray.from_rows([rows[i] for i in index])
            elif ndim > 1:
                raise IndexError('must use tuple for multidimensional indexing')
            elif index == open_slice:
                return self
            else:
                value = rows[index]
                if value.__class__ is list: value = SparseArray.from_rows(value)
                return value
        
    def __setitem__(self, index, value):
        rows = self.rows
        if index.__class__ is tuple:
            m, n = unpack_index(index, self.ndim)
            md = get_ndim(m)
            nd = get_ndim(n)
            if not md and m == open_slice:
                if not nd and n == open_slice:
                    self[:] = value
                else:
                    vd = get_ndim(value)
                    if vd == 0.:
                        for i in rows: i[n] = value
                    elif vd == 1:
                        if nd == 0.:
                            for i, j in zip(rows, value): i[n] = j
                        elif nd == 1:
                            for i, j in zip(rows, n): i[j] = value
                        else:
                            raise IndexError(
                                f'cannot broadcast {vd}-d array on to array column'
                            )
                    elif vd == 2:
                        value_length = len(value)
                        self_length = len(rows)
                        if self_length == value_length:
                            for i, j in zip(rows, value): i[n] = j
                        else:
                            raise IndexError(
                                f'cannot broadcast input array with length {value_length} to length {self_length}'
                            )
                    else:
                        raise IndexError(
                            f'cannot broadcast {vd}-d array on to 2-d sparse array'
                        )
            elif not nd and n == open_slice:
                md = get_ndim(m)
                if md == 0:
                    rows[m][:] = value
                elif md == 1:
                    ndim = get_ndim(value)
                    if ndim in (0, 1):
                        for i in m: rows[i][:] = value
                    elif ndim == 2:
                        if len(m) == len(value):
                            for i, j in zip(m, value):
                                rows[i][:] = j
                        else:
                            raise IndexError(
                                f'cannot broadcast input array with length {len(value)} to length {len(m)}'
                            )
                    else:
                        raise IndexError(
                            f'cannot broadcast {ndim}-d array on to 1-d '
                        )
                else:
                    raise IndexError(f'column index can be at most 1-d, not {md}-d')
            elif md == 0: 
                rows[m][n] = value
            elif md == 1: 
                vd = get_ndim(value)
                if vd == 0:
                    if value:
                        for i, j in zip(*unpack_index(index, self.ndim)): 
                            rows[i].dct[j.__index__()] = value
                    else:
                        for i, j in zip(*unpack_index(index, self.ndim)): 
                            dct = rows[i].dct
                            j = j.__index__()
                            if j in dct: del dct[j]
                elif vd == 1:
                    for i, j, k in zip(*unpack_index(index, self.ndim), value): 
                        dct = rows[i].dct
                        j = j.__index__()
                        if k: rows[i].dct[j] = k
                        elif j in dct: del dct[j]
                else:
                    raise IndexError(
                        'cannot set an array element with a sequence'
                    )
            else:
                raise IndexError(f'row index can be at most 1-d, not {md}-d')
        else:
            ndim, has_bool = get_index_properties(index)
            if has_bool:
                if ndim != 2:
                    raise IndexError(
                        f'boolean index is {ndim}-d but sparse array is 2-d '
                    )
                self[np.where(index)] = value
            elif ndim == 1:
                SparseArray.from_rows([rows[i] for i in index])[:] = value
            elif ndim == 2:
                raise IndexError('must use tuple for multidimensional indexing')
            elif index == open_slice:
                vd = get_ndim(value)
                if vd in (0, 1):
                    for i in rows: i[:] = value
                elif vd == 2:
                    value_length = len(value)
                    self_length = len(rows)
                    if self_length == value_length:
                        for i, j in zip(rows, value): i[:] = j
                    elif value_length == 1:
                        value = value[0]
                        for i in rows: i[:] = value
                    else:
                        raise IndexError(
                            f'cannot broadcast input array with length {value_length} to length {self_length}'
                        )
                else:
                    raise IndexError(
                        f'cannot broadcast {vd}-d array on to 2-d sparse array'
                    )
            else:
                rows[index][:] = value
    
    @property
    def vector_size(self):
        vector_size = 0
        for i in self.rows:
            vector_size = i.size
            break
        return vector_size
    
    @property
    def shape(self):
        return (len(self.rows), self.vector_size)
    
    @property
    def size(self):
        return len(self.rows) * self.vector_size
    
    def to_array(self, dtype=None):
        rows = self.rows
        N = len(rows)
        arr = np.zeros([N, self.vector_size], dtype=self.dtype if dtype is None else dtype)
        for i in range(N):
            row = rows[i]
            for j, value in row.dct.items():
                arr[i, j] = value
        return arr
        
    def sum_of(self, index, axis=0):
        rows = self.rows
        if axis == 0:
            return sum([i[index] for i in rows])
        elif axis == 1:
            arr = np.zeros(len(rows))
            for i, j in enumerate(rows): arr[i] = j.sum_of(index)
            return arr
        else:
            raise ValueError('axis must be either 0, 1')
    
    def all(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = all([i.all() for i in rows])
            if keepdims: arr = np.array([[arr]])
        elif axis == 0:
            size = self.vector_size
            dcts = [i.dct for i in rows]
            if keepdims:
                arr = np.zeros([1, size])
                for i in range(size): arr[0, i] = all([i in dct for dct in dcts])
            else:
                arr = np.zeros(size)
                for i in range(size): arr[i] = all([i in dct for dct in dcts])
        elif axis == 1:
            arr = np.zeros([len(rows), 1] if keepdims else len(rows))
            for i, j in enumerate(rows): arr[i] = j.all()
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def any(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = any([i.any() for i in rows])
            if keepdims: arr = np.array([[arr]])
        elif axis == 0:
            keys = set()
            for i in rows: keys.update(i.dct)
            if keepdims:
                arr = np.zeros([1, self.vector_size])
                for i in keys: arr[0, i] = True
            else:
                arr = np.zeros(self.vector_size)
                for i in keys: arr[i] = True
        elif axis == 1:
            arr = np.zeros([len(rows), 1] if keepdims else len(rows))
            for i, j in enumerate(rows): arr[i] = j.any()
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
            
    def sum(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = sum([sum(i.dct.values()) for i in rows])
            if keepdims: arr = SparseArray([[arr]], 1)
        elif axis == 0:
            arr = SparseVector(sum_sparse_vectors(rows), size=self.vector_size)
            if keepdims: arr = SparseArray([arr])
        elif axis == 1:
            arr = np.zeros([len(rows), 1] if keepdims else len(rows))
            for i, j in enumerate(rows): arr[i] = sum(j.dct.values())
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def mean(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = sum([sum(i.dct.values()) for i in rows]) / sum([i.size for i in self.rows])
            if keepdims: arr = SparseArray([[arr]], 1)
            return arr
        elif axis == 0:
            arr = SparseVector(sum_sparse_vectors(rows), size=self.vector_size)
            arr /= len(rows)
            if keepdims: arr = SparseArray([arr])
            return arr
        elif axis == 1:
            arr = np.zeros([len(rows), 1] if keepdims else len(rows))
            for i, j in enumerate(rows): arr[i] = sum(j.dct.values()) / j.size
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def max(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = max([i.max() for i in rows])
            if keepdims: arr = SparseArray([[arr]], 1)
        elif axis == 0:
            keys = set()
            dcts = [i.dct for i in rows]
            for i in dcts: keys.update(i)
            arr = SparseVector.from_dict(
                {i: x for i in keys if (x:=max([(j[i] if i in j else 0.) for j in dcts]))},
                self.vector_size
            )
            if keepdims: arr = SparseArray([arr])
        elif axis == 1:
            arr = np.zeros([len(rows), 1] if keepdims else len(rows))
            for i, j in enumerate(rows): arr[i] = j.max()
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def min(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = min([i.min() for i in rows])
            if keepdims: arr = SparseArray([[arr]], 1)
        elif axis == 0:
            keys = set()
            dcts = [i.dct for i in rows]
            for i in dcts: keys.update(i)
            arr = SparseVector.from_dict(
                {i: x for i in keys if (x:=min([(j[i] if i in j else 0.) for j in dcts]))},
                self.vector_size
            )
            if keepdims: arr = SparseArray([arr])
        elif axis == 1:
            arr = np.zeros([len(rows), 1] if keepdims else len(rows))
            for i, j in enumerate(rows): arr[i] = j.min()
            return arr
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def __iadd__(self, value):
        rows = self.rows
        if get_ndim(value) == 2 and value.__class__ is not SparseVector:
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i += j
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        else:
            for i in rows: i += value
        return self
    
    def __add__(self, value):
        new = self.copy()
        new += value
        return new
            
    def __isub__(self, value):
        rows = self.rows
        if get_ndim(value) == 2 and value.__class__ is not SparseVector:
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i -= j
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        else:
            for i in rows: i -= value
        return self
    
    def __sub__(self, value):
        new = self.copy()
        new -= value
        return new
    
    def __imul__(self, value):
        rows = self.rows
        if get_ndim(value) == 2 and value.__class__ is not SparseVector:
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i *= j
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        else:
            for i in rows: i *= value
        return self
    
    def __mul__(self, value):
        new = self.copy()
        new *= value
        return new
        
    def __itruediv__(self, value):
        rows = self.rows
        if get_ndim(value) == 2 and value.__class__ is not SparseVector:
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i /= j
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        else:
            for i in rows: i /= value
        return self
    
    def __truediv__(self, value):
        new = self.copy()
        new /= value
        return new
    
    def __rtruediv__(self, value):
        new = self.copy()
        rows = new.rows
        if get_ndim(value) == 2 and value.__class__ is not SparseVector:
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i[:] = j / i
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        else:
            for i in rows: i[:] = value / i
        return new
    
    def __neg__(self):
        return SparseArray.from_rows([-i for i in self.rows])
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return -self + other
    
    def __rmul__(self, other):
        return self * other
    
    # Not yet optimized methods
    
    def __eq__(self, other):
        return self.to_array() == other
    
    def __ne__(self, other):
        return self.to_array() != other
    
    def __gt__(self, other):
        return self.to_array() > other
    
    def __lt__(self, other):
        return self.to_array() < other
    
    def __ge__(self, other):
        return self.to_array() >= other
    
    def __le__(self, other):
        return self.to_array() <= other
    
    def argmin(self, *args, **kwargs):
        return self.to_array().argmin(*args, **kwargs)

    def argmax(self, *args, **kwargs):
        return self.to_array().argmax(*args, **kwargs)
    
    def argpartition(self, *args, **kwargs):
        return self.to_array().argpartition(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return self.to_array().argsort(*args, **kwargs)

    def choose(self, *args, **kwargs):
        return self.to_array().choose(*args, **kwargs)

    def clip(self, *args, **kwargs):
        return self.to_array().clip(*args, **kwargs)

    def conj(self):
        return self.to_array().conj()

    def conjugate(self):
        return self.to_array().conjugate()

    def cumprod(self, *args, **kwargs):
        return self.to_array().cumprod(*args, **kwargs)

    def cumsum(self, *args, **kwargs):
        return self.to_array().cumsum(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return self.to_array().dot(*args, **kwargs)

    def prod(self, *args, **kwargs):
        return self.to_array().prod(*args, **kwargs)

    def ptp(self, *args, **kwargs):
        return self.to_array().ptp(*args, **kwargs)

    def put(self, *args, **kwargs):
        return self.to_array().put(*args, **kwargs)

    def round(self, *args, **kwargs):
        return self.to_array().round(*args, **kwargs)

    def std(self, *args, **kwargs):
        return self.to_array().std(*args, **kwargs)

    def trace(self, *args, **kwargs):
        return self.to_array().trace(*args, **kwargs)

    def var(self, *args, **kwargs):
        return self.to_array().var(*args, **kwargs)

    def __matmul__(self, other):
        return self.to_array() @ other

    def __floordiv__(self, other):
        return self.to_array() // other

    def __mod__(self, other):
        return self.to_array() % other

    def __pow__(self, other):
        return self.to_array() ** other

    def __lshift__(self, other):
        return self.to_array() << other

    def __rshift__(self, other):
        return self.to_array() >> other

    def __and__(self, other): 
        return self.to_array() & other

    def __xor__(self, other): 
        return self.to_array() ^ other

    def __or__(self, other):
        return self.to_array() | other

    def __rmatmul__(self, other):
        return other @ self.to_array()

    def __rfloordiv__(self, other):
        return other // self.to_array()

    def __rmod__(self, other):
        return other % self.to_array()

    def __rpow__(self, other):
        return other ** self.to_array()

    def __rlshift__(self, other):
        return other << self.to_array() 

    def __rrshift__(self, other):
        return other >> self.to_array()

    def __rand__(self, other):
        return other & self.to_array()

    def __rxor__(self, other):
        return other ^ self.to_array()

    def __ror__(self, other):
        return other | self.to_array()

    def __imatmul__(self, other):
        raise TypeError("in-place matrix multiplication is not (yet) supported")

    def __ifloordiv__(self, other):
        self[:] = self.to_array() // other
        return self

    def __imod__(self, other): 
        self[:] =self.to_array() % other
        return self

    def __ipow__(self, other):
        self[:] = self.to_array() ** other
        return self

    def __ilshift__(self, other):
        self[:] = self.to_array() << other
        return self

    def __irshift__(self, other):
        self[:] = self.to_array() >> other
        return self

    def __iand__(self, other): 
        self[:] = self.to_array() & other
        return self

    def __ixor__(self, other): 
        self[:] = self.to_array() ^ other
        return self

    def __ior__(self, other):
        self[:] = self.to_array() | other
        return self
    
    # Representation
    
    def __repr__(self):
        name = type(self).__name__
        n_spaces = len(name) - 5
        nums = repr(self.to_array())[5:].replace('\n', '\n' + n_spaces * ' ')
        return f'{name}{nums}'
    
    def __str__(self):
        arr = self.to_array()
        return str(arr)
            

class SparseVector:
    """
    Create a SparseVector object that can be used for array-like arithmetic operations
    (i.e., +, -, *, /) of sparse 1-dimensional arrays. 
    
    """
    __slots__ = ('dct', 'read_only', 'size', '_base')
    ndim = 1
    dtype = float
    
    def __init__(self, obj=None, size=None):
        self.read_only = False
        if obj is None:
            self.dct = {}
            if size is None: raise ValueError('must pass size if no object given')
            self.size = size
        elif isinstance(obj, dict):
            self.dct = obj
            self.size = size
            if size is None: raise ValueError('must pass size if object is a dictionary')
        elif isinstance(obj, SparseVector):
            self.dct = obj.dct.copy()
            self.size = obj.size if size is None else size
        elif hasattr(obj, '__iter__'):
            self.dct = dct = {}
            for i, j in enumerate(obj):
                if j: dct[i] = j
            self.size = len(obj) if size is None else size
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse vector')
    
    def __abs__(self):
        positive = abs
        dct = self.dct.copy()
        for i in dct: dct[i] = positive(dct[i])
        return SparseVector.from_dict(dct, self.size)
    
    def __iter__(self):
        dct = self.dct
        for i in range(self.size):
            yield dct.get(i, 0.)
    
    def __len__(self):
        return self.size
    
    def __float__(self):
        return self.to_array(float)
    
    def __int__(self):
        return self.to_array(int)
    
    def to_flat_array(self, arr=None):
        if arr is None:
            return self.to_array()
        else:
            for i, j in self.dct.items(): arr[i] = j
            return arr
    
    def from_flat_array(self, arr=None):
        self[:] = arr
    
    @classmethod
    def from_size(cls, size):
        return cls.from_dict({}, size)
    
    @property
    def vector_size(self):
        return self.size
    
    @property
    def shape(self):
        return (self.size,)
    
    @property
    def base(self):
        try:
            base = self._base
        except:
            self._base = base = frozenset([id(self.dct)])
        return base
    
    @classmethod
    def from_dict(cls, dct, size):
        new = cls.__new__(cls)
        new.dct = dct
        new.size = size
        new.read_only = False
        return new
        
    def sum_of(self, index):
        dct = self.dct
        if hasattr(index, '__iter__'):
            return sum([dct[i] for i in index if i in dct])
        else:
            return dct.get(index, 0.)
    
    shares_data_with = SparseArray.shares_data_with
    
    def remove_negatives(self):
        dct = self.dct
        for i in tuple(dct): 
            if dct[i] < 0.: del dct[i]
    
    def has_negatives(self):
        for value in self.dct.values():
            if value < 0.: return True
        return False
    
    def negative_index(self):
        dct = self.dct
        return [i for i in dct if dct[i] < 0.],
    
    def positive_index(self):
        dct = self.dct
        return [i for i in dct if dct[i] > 0.],
    
    def nonzero_index(self):
        return [*self.dct.keys()],
    nonzero = nonzero_index
    
    def nonzero_keys(self):
        return self.dct.keys()
    
    def nonzero_values(self):
        return self.dct.values()
    
    def nonzero_items(self):
        return self.dct.items()
    
    def sparse_equal(self, other):
        other = sparse_vector(other)
        return self.dct == other.dct
    
    def any(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse vector')
        arr = bool(self.dct)
        if keepdims: arr = np.array([arr])
        return arr
    
    def all(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse vector')
        arr = len(self.dct) == self.size
        if keepdims: arr = np.array([arr])
        return arr
    
    def sum(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse vector')
        arr = sum(self.dct.values())
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def mean(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse vector')
        arr = sum(self.dct.values()) / self.size if self.dct else 0.
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def max(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse vector')
        dct = self.dct
        if dct:
            arr = max(dct.values())
            if arr < 0 and len(dct) < self.size: arr = 0
        elif self.size:
            arr = 0.
        else:
            raise ValueError('zero-size vector to reduction has no identity')
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def min(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse vector')
        dct = self.dct
        if dct:
            arr = min(dct.values())
            if arr > 0 and len(dct) < self.size: arr = 0
        elif self.size:
            arr = 0.
        else:
            raise ValueError('zero-size vector to reduction has no identity')
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def to_array(self, dtype=None):
        if dtype is None: dtype = self.dtype
        arr = np.zeros(self.size, dtype=dtype)
        for i, j in self.dct.items(): arr[i] = j
        return arr
    
    def copy(self):
        return SparseVector.from_dict(self.dct.copy(), self.size)
    
    def copy_like(self, other):
        dct = self.dct
        if dct is other.dct: return
        dct.clear()
        dct.update(other.dct)
    
    def clear(self):
        if self.read_only: raise ValueError('assignment destination is read-only')
        self.dct.clear()
    
    def __getitem__(self, index):
        dct = self.dct
        if index.__class__ is tuple:
            index, = unpack_index(index, self.ndim)
        ndim, has_bool = get_index_properties(index)
        if has_bool: 
            if ndim != 1:
                raise IndexError(
                    f'boolean index is {ndim}-d but sparse vector is 1-d '
                )
            return self[np.where(index)]
        if ndim == 1:
            arr = np.zeros(len(index))
            for n, i in enumerate(index):
                i = i.__index__()
                if i in dct: arr[n] = dct[i]
            return arr
        elif index == open_slice:
            return self
        else:
            return dct.get(index.__index__(), 0.)

    def __setitem__(self, index, value):
        if self.read_only: raise ValueError('assignment destination is read-only')
        dct = self.dct
        if index.__class__ is tuple:
            index, = unpack_index(index, self.ndim)
        ndim, has_bool = get_index_properties(index)
        if has_bool: 
            if ndim != 1:
                raise IndexError(
                    f'boolean index is {ndim}-d but sparse vector is 1-d '
                )
            index, = np.where(index)
        if ndim:
            if (vd:=get_ndim(value)) == 1:
                for i, j in zip(index, value): 
                    i = i.__index__()
                    if j: dct[i] = j
                    elif i in dct: del dct[i]
                        
            elif vd > 1:
                raise IndexError(
                    f'cannot broadcast {vd}-d array on to 1-d sparse vector'
                )
            elif value:
                for i in index:
                    i = i.__index__()
                    dct[i] = value
            else:
                for i in index:
                    i = i.__index__()
                    if i in dct: del dct[i]
        elif index == open_slice:
            if value is self: return
            dct.clear()
            if value.__class__ is SparseVector:
                dct.update(value.dct)
            elif hasattr(value, '__iter__'):
                for i, j in enumerate(value):
                    if j: dct[i] = j
                    elif i in dct: del dct[i]  
            elif value != 0.:
                for i in range(self.size): dct[i] = value
        elif hasattr(value, '__iter__'):
            raise IndexError(
                'cannot set an array element with a sequence'
            )
        else:
            index = index.__index__()
            if value:
                dct[index] = value
            elif index in dct:
                del dct[index]
    
    def __iadd__(self, other):
        dct = self.dct
        if other.__class__ is SparseVector:
            for i, j in other.dct.items():
                if i in dct:
                    j += dct[i]
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = j
        elif hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j += dct[i]
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = j
        elif other != 0.:
            for i in range(self.size):
                if i in dct: 
                    j = dct[i] + other
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = other
        return self
    
    def __add__(self, other):
        new = self.copy()
        new += other
        return new
            
    def __isub__(self, other):
        dct = self.dct
        if other.__class__ is SparseVector:
            for i, j in other.dct.items():
                if i in dct:
                    j = dct[i] - j
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -j
        elif hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] - j
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -j
        elif other != 0.:
            for i in range(self.size):
                if i in dct: 
                    j = dct[i] - other
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -other
        return self
    
    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new
    
    def __imul__(self, other):
        dct = self.dct
        if other.__class__ is SparseVector:
            other = other.dct
            for i in tuple(dct):
                if i in other:
                    dct[i] *= other[i]
                else:
                    del dct[i]
        elif hasattr(other, '__iter__'):
            for i in tuple(dct):
                j = other[i]
                if j: dct[i] *= j
                else: del dct[i]
        else:
            other = other
            if other:
                for i in dct: dct[i] *= other
            else:
                dct.clear()
        return self
    
    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new
        
    def __itruediv__(self, other):
        if self.read_only: raise ValueError('assignment destination is read-only')
        dct = self.dct
        if other.__class__ is SparseVector:
            other = other.dct
            for i in dct: dct[i] /= other[i]
        elif hasattr(other, '__iter__'):
            if other.__len__() == 1:
                other, = other
                if other:
                    for i in dct: dct[i] /= other
                elif dct:
                    raise FloatingPointError('division by zero')
            else:
                for i in dct: dct[i] /= other[i]
        else:
            if other:
                for i in dct: dct[i] /= other
            elif dct:
                raise FloatingPointError('division by zero')
        return self
    
    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new
    
    def __neg__(self):
        return SparseVector.from_dict({i: -j for i, j in self.dct.items()}, self.size)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return -self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        dct = self.dct.copy()
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if j:
                    if i in dct:
                        dct[i] = j / dct[i] 
                    else:
                        raise FloatingPointError('division by zero')
                else: del dct[i]
        elif other:
            if len(dct) != self.size: raise FloatingPointError('division by zero')
            for i in dct: dct[i] = other / dct[i]
        else:
            dct = {}
        return SparseVector.from_dict(dct, self.size)
    
    # Not yet optimized methods
    
    __eq__ = SparseArray.__eq__
    __ne__ = SparseArray.__ne__
    __gt__ = SparseArray.__gt__
    __lt__ = SparseArray.__lt__
    __ge__ = SparseArray.__ge__
    __le__ = SparseArray.__le__
    
    argmin = SparseArray.argmin
    argmax = SparseArray.argmax
    argpartition = SparseArray.argpartition
    argsort = SparseArray.argsort
    choose = SparseArray.choose
    clip = SparseArray.clip
    conj = SparseArray.conj
    conjugate = SparseArray.conjugate
    cumprod = SparseArray.cumprod
    cumsum = SparseArray.cumsum
    dot = SparseArray.dot
    prod = SparseArray.prod
    ptp = SparseArray.ptp
    put = SparseArray.put
    round = SparseArray.round
    std = SparseArray.std
    trace = SparseArray.trace
    var = SparseArray.var
    __matmul__ = SparseArray.__matmul__
    __floordiv__ = SparseArray.__floordiv__ 
    __mod__ = SparseArray.__mod__
    __pow__ = SparseArray.__pow__ 
    __lshift__ = SparseArray.__lshift__ 
    __rshift__ = SparseArray.__rshift__ 
    __and__ = SparseArray.__and__ 
    __xor__ = SparseArray.__xor__ 
    __or__ = SparseArray.__or__ 
    __rmatmul__ = SparseArray.__rmatmul__
    __rfloordiv__ = SparseArray.__rfloordiv__
    __rmod__ = SparseArray.__rmod__
    __rpow__ = SparseArray.__rpow__
    __rlshift__ = SparseArray.__rlshift__
    __rrshift__ = SparseArray.__rrshift__
    __rand__ = SparseArray.__rand__
    __rxor__ = SparseArray.__rxor__
    __ror__ = SparseArray.__ror__
    __imatmul__ = SparseArray.__imatmul__
    __ifloordiv__ = SparseArray.__ifloordiv__
    __imod__ = SparseArray.__imod__ 
    __ipow__ = SparseArray.__ipow__
    __ilshift__ = SparseArray.__ilshift__
    __irshift__ = SparseArray.__irshift__
    __iand__ = SparseArray.__iand__
    __ixor__ = SparseArray.__ixor__
    __ior__ = SparseArray.__ior__
    
    # Representation
    
    __bool__ = SparseArray.__bool__
    __repr__ = SparseArray.__repr__
    __str__ = SparseArray.__str__
    
# %% For column slicing in sparse array objects
    
# class DictionaryColumn: 
#     __slots__ = ('rows', 'index')

#     def __init__(self, rows, index):
#         self.rows = rows # List of dictionaries
#         self.index = index

#     def __eq__(self, other):
#         return self.copy() == other

#     def get(self, key, default=None):
#         try:
#             dct = self.rows[key]
#         except:
#             return default
#         return dct.get(self.index, default)

#     def __contains__(self, key):
#         try:
#             dct = self.rows[key]
#         except:
#             return False
#         return self.index in dct

#     def __delitem__(self, key):
#         del self.rows[key][self.index]
        
#     def __bool__(self):
#         index = self.index
#         return any([index in dct for dct in self.rows])
        
#     def keys(self):
#         index = self.index
#         for i, dct in enumerate(self.rows):
#             if index in dct: yield i
#     __iter__ = keys
    
#     def clear(self):
#         index = self.index
#         for i, dct in enumerate(self.rows):
#             if index in dct: del dct[index]
    
#     def __len__(self):
#         index = self.index
#         return sum([1 for dct in self.rows if index in dct])
    
#     def __getitem__(self, key):
#         return self.rows[key][self.index]
    
#     def __setitem__(self, key, value):
#         self.rows[key][self.index] = value
    
#     def items(self):
#         index = self.index
#         for i, dct in enumerate(self.rows):
#             if index in dct: yield (i, dct[index])
            
#     def values(self):
#         index = self.index
#         for dct in self.rows:
#             if index in dct: yield dct[index]
        
#     def copy(self):
#         index = self.index
#         return {i: dct[index] for i, dct in enumerate(self.rows) if index in dct}
    
#     def update(self, dct):
#         for i, j in dct.items(): self[i] = j
    
#     def pop(self, key):
#         raise NotImplementedError
        
#     def popitem(self):
#         raise NotImplementedError
    
#     def setdefault(self, key, default=None):
#         raise NotImplementedError
        
#     def from_keys(self, keys, value=None):
#         raise NotImplementedError