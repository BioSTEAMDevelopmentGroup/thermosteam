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

open_slice = slice(None)

def default_range(slice, max):
    return range(
        0 if slice.start is None else slice.start,
        max if slice.stop is None else slice.stop, 
        1 if slice.step is None else slice.step,
    )

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
    if arr.__class__ in (SparseVector, SparseArray):
        return arr.nonzero_items()
    else:
        return [(i, j) for i, j in enumerate(arr.flat) if j]

def sparse_array(arr, copy=False, vector_size=None):
    """
    Convert 2-d array to a SparseArray object.

    """
    if arr.__class__ is SparseArray:
        return arr.copy() if copy else arr
    else:
        return SparseArray(arr, vector_size)

def sparse(arr, copy=False, vector_size=None):
    """
    Create a sparse array that can be used for array-like arithmetic operations
    (i.e., +, -, *, /) of sparse 1 or 2-dimensional arrays.  
    
    Parameters
    ----------
    arr : array-like
        Structure to be converted to a sparse array. 
    
    Examples
    --------
    Create a sparse array from an array-like object.
    
    >>> from thermosteam.base import sparse
    >>> sa = sparse([[0, 1, 2], [3, 2, 0]])
    >>> sa
    sparse([[0., 1., 2.],
            [3., 2., 0.]])
    
    Create a sparse array from an a list of dictionaries of index-nonzero value pairs.
    
    >>> sa = sparse(
    ...     [{1: 1, 2: 2},
    ...      {0: 3, 1: 2}],
    ...     vector_size=3,
    ... )
    >>> sa
    sparse([[0., 1., 2.],
            [3., 2., 0.]])
    
    Sparse arrays support arithmetic operations just like dense arrays
    
    >>> sa * sa 
    sparse([[0., 1., 4.],
            [9., 4., 0.]])
    
    Sparse arrays assume sparsity across columns (0-axis) but not across rows. 
    For this reason, indexing rows will return sparse arrays while indexing
    columns will return NumPy dense arrays:
        
    >>> sa[0]
    sparse([0., 1., 2.])
    
    >>> sa[:, 0]
    array([0., 3.])
    
    """
    if arr.__class__ in (SparseArray, SparseVector):
        return arr
    elif (ndim:=get_ndim(arr)) == 1:
        return SparseVector(arr, vector_size)
    elif ndim == 2:
        return SparseArray(arr, vector_size)
    else:
        raise ValueError(f'cannot convert {ndim}-d object to a sparse array or vector')
    
def get_ndim(value):
    if hasattr(value, 'ndim'):
        return value.ndim
    elif hasattr(value, '__iter__'):
        for i in value: return get_ndim(i) + 1
        return 1
    return 0

def get_index_properties(index, bools=frozenset([bool, np.bool_, np.dtype('bool')])):
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
    __doc__ = sparse.__doc__
    __slots__ = ('rows', '_base')
    ndim = 2
    dtype = float
    
    def __init__(self, obj=None, vector_size=None):
        if obj is None:
            self.rows = []
        elif hasattr(obj, '__iter__'):
            self.rows = [sparse_vector(row, size=vector_size) for row in obj]
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
    
    def copy_like(self, other):
        rows = self.rows
        for i, j in zip(rows, other.rows):
            i.copy_like(j)
    
    def nonzero_index(self):
        m = []; n = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                m.append(i); n.append(j)
        return m, n
    nonzero = nonzero_index
    
    def nonzero_values(self):
        for row in self.rows:
            yield from row.dct.values()
    
    def nonzero_items(self):
        for i, row in enumerate(self.rows):
            for j, k in row.dct.items(): yield ((i, j), k)
    
    def nonzero_rows(self):
        return [i for i, j in enumerate(self.rows) if j.dct]
    
    def nonzero_keys(self):
        keys = set()
        for i in self.rows: keys.update(i.dct)
        return keys
    
    def positive_index(self):
        m = []; n = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value > 0.:
                    m.append(i); n.append(j)
        return m, n
    
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
        for i in self.rows: i.remove_negatives()
    
    def shares_data_with(self, other):
        return bool(self.base.intersection(other.base))
    
    def to_flat_array(self, arr=None):
        vector_size = self.vector_size
        dtype = self.dtype
        rows = self.rows
        N = rows.__len__()
        if arr is None: arr = np.zeros(N * vector_size, dtype=dtype)
        for i in range(N):
            row = rows[i]
            for j, value in row.dct.items(): 
                arr[i * vector_size + j] = value
        return arr
        
    def from_flat_array(self, arr):
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
            if m.__class__ is slice:
                if m == open_slice:
                    if n == open_slice:
                        return self
                    else:
                        value = np.array([i[n] for i in rows])
                else:
                    value = np.array([rows[i][n] for i in default_range(m, len(rows))])
                value.setflags(0)
                return value
            elif n.__class__ == slice:
                md, misbool = get_index_properties(m)
                if n == open_slice:
                    if md == 0:
                        return rows[m]
                    elif md == 1:
                        if misbool:
                            return SparseArray.from_rows([rows[i] for i, j in enumerate(m) if j])
                        else:
                            return SparseArray.from_rows([rows[i] for i in m])
                    else:
                        raise IndexError(f'row index can be at most 1-d, not {md}-d')
                elif md == 0:
                    return rows[m][n]
                elif md == 1:
                    if misbool:
                        value = np.array([rows[i][n] for i, j in enumerate(m) if j])
                    else:
                        value = np.array([rows[i][n] for i in m])
                    value.setflags(0)
                    return value
                else:
                    raise IndexError(f'row index can be at most 1-d, not {md}-d')
                    
            else:
                md = get_ndim(m)
                if md == 0: 
                    return rows[m][n]
                elif md == 1: 
                    nd = get_ndim(n)
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
                if ndim == 1: 
                    return SparseArray.from_rows([rows[i] for i, j in enumerate(index) if j])
                else:
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
            if m.__class__ is slice:
                if m != open_slice:
                    rows = [rows[i] for i in default_range(m, len(rows))]
                if n.__class__ is slice:
                    if n == open_slice:
                        self[:] = value
                        return
                    else:
                        n = default_range(n, self.vector_size)
                vd = get_ndim(value)
                if vd == 0.:
                    for i in rows: i[n] = value
                elif vd == 1:
                    nd, nisbool = get_index_properties(n)
                    if nd:
                        if nisbool: n, = np.where(n)
                        for i in rows: i[n] = value
                    else:
                        for i, j in zip(rows, value): i[n] = j
                elif vd == 2:
                    for i, j in zip(rows, value): i[n] = j # TODO: With python 3.10, use strict=True zip kwarg
                else:
                    raise IndexError(
                        'cannot set an array element with a sequence'
                    )
            elif n.__class__ is slice:
                md, misbool = get_index_properties(m)
                if md == 0:
                    rows[m][n] = value
                elif md == 1:
                    vd = get_ndim(value)
                    if vd in (0, 1):
                        if misbool:
                            for i, j in enumerate(m): 
                                if j: rows[i][n] = value
                        else:
                            for i in m: rows[i][n] = value
                    elif vd == 2:
                        for i, j in zip(m, value): rows[i][n] = j # TODO: With python 3.10, use strict=True zip kwarg
                    else:
                        raise IndexError(
                            f'cannot broadcast {vd}-d array on to 1-d '
                        )
                else:
                    raise IndexError(f'row index can be at most 1-d, not {md}-d')
            elif (md:=get_ndim(m)) == 0: 
                rows[m][n] = value
            elif md == 1: 
                vd = get_ndim(value)
                if vd == 0:
                    if value:
                        value = value.__float__()
                        for i, j in zip(m, n): 
                            rows[i].dct[j.__index__()] = value
                    else:
                        for i, j in zip(m, n): 
                            dct = rows[i].dct
                            j = j.__index__()
                            if j in dct: del dct[j]
                elif vd == 1:
                    for i, j, k in zip(m, n, value): 
                        dct = rows[i].dct
                        j = j.__index__()
                        if k: rows[i].dct[j] = k.__float__()
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
                if ndim == 1: 
                    vd = get_ndim(value)
                    if vd == 0:
                        for i, j in enumerate(index):
                            if j: rows[i][:] = value
                    else:
                        for i, j in enumerate(index):
                            if j: rows[i][:] = value[i]
                else:
                    self[np.where(index)] = value
                return
            elif ndim == 1:
                rows = [rows[i] for i in index]
            elif index.__class__ is slice:
                if index != open_slice: rows = [rows[i] for i in default_range(index, len(rows))]
            elif ndim == 0:
                rows[index][:] = value
                return
            else:
                raise IndexError('must use tuple for multidimensional indexing')
            vd = get_ndim(value)
            if vd in (0, 1):
                for i in rows: i[:] = value
            elif vd == 2:
                for i, j in zip(rows, value): i[:] = j # TODO: With python 3.10, use strict=True zip kwarg
            else:
                raise IndexError(
                    'cannot set an array element with a sequence'
                )
    
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
    
    @property
    def value(self):
        return self.to_array() # For backwards compatibility with property objects used by QSDsan
    
    def setflags(self, flag):
        if flag == 0: 
            for i in self.rows: i.read_only = True # For backwards compatibility with arrays
        else: raise NotImplementedError(f'flag {flag} not yet implemented')
    
    def to_array(self, dtype=None):
        rows = self.rows
        N = rows.__len__()
        arr = np.zeros([N, self.vector_size], dtype=self.dtype if dtype is None else dtype)
        for i in range(N):
            row = rows[i]
            for j, value in row.dct.items():
                arr[i, j] = value
        return arr
    astype = to_array    
    
    def sum_of(self, index, axis=0):
        rows = self.rows
        if axis == 0:
            return sum([i[index] for i in rows])
        elif axis == 1:
            arr = np.zeros(rows.__len__())
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
            arr = np.zeros([rows.__len__(), 1] if keepdims else rows.__len__())
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
            arr = np.zeros([rows.__len__(), 1] if keepdims else rows.__len__())
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
            arr = np.zeros([rows.__len__(), 1] if keepdims else rows.__len__())
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
            arr /= rows.__len__()
            if keepdims: arr = SparseArray([arr])
            return arr
        elif axis == 1:
            arr = np.zeros([rows.__len__(), 1] if keepdims else rows.__len__())
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
            arr = np.zeros([rows.__len__(), 1] if keepdims else rows.__len__())
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
            arr = np.zeros([rows.__len__(), 1] if keepdims else rows.__len__())
            for i, j in enumerate(rows): arr[i] = j.min()
            return arr
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def __iadd__(self, value):
        rows = self.rows
        if get_ndim(value) > 1:
            for i, j in zip(rows, value): i += j # TODO: With python 3.10, use strict=True zip kwarg
        else:
            for i in rows: i += value
        return self
    
    def __add__(self, value):
        new = self.copy()
        new += value
        return new
            
    def __isub__(self, value):
        rows = self.rows
        if get_ndim(value) > 1:
            for i, j in zip(rows, value): i -= j # TODO: With python 3.10, use strict=True zip kwarg
        else:
            for i in rows: i -= value
        return self
    
    def __sub__(self, value):
        new = self.copy()
        new -= value
        return new
    
    def __imul__(self, value):
        rows = self.rows
        if get_ndim(value) > 1:
            for i, j in zip(rows, value): i *= j # TODO: With python 3.10, use strict=True zip kwarg
        else:
            for i in rows: i *= value
        return self
    
    def __mul__(self, value):
        new = self.copy()
        new *= value
        return new
        
    def __itruediv__(self, value):
        rows = self.rows
        if get_ndim(value) > 1:
            for i, j in zip(rows, value): i /= j # TODO: With python 3.10, use strict=True zip kwarg
        else:
            for i in rows: i /= value
        return self
    
    def __truediv__(self, value):
        new = self.copy()
        new /= value
        return new
    
    def __rtruediv__(self, value):
        new = self.copy()
        if get_ndim(value) > 1:
            for i, j in zip(new.rows, value): i[:] = j / i # TODO: With python 3.10, use strict=True zip kwarg
        else:
            for i in new.rows: i[:] = value / i
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
    
    def __eq__(self, other): # pragma: no cover
        return self.to_array() == other
    
    def __ne__(self, other): # pragma: no cover
        return self.to_array() != other
    
    def __gt__(self, other): # pragma: no cover
        return self.to_array() > other
    
    def __lt__(self, other): # pragma: no cover
        return self.to_array() < other
    
    def __ge__(self, other): # pragma: no cover
        return self.to_array() >= other
    
    def __le__(self, other): # pragma: no cover
        return self.to_array() <= other
    
    def argmin(self, *args, **kwargs): # pragma: no cover
        return self.to_array().argmin(*args, **kwargs)

    def argmax(self, *args, **kwargs): # pragma: no cover
        return self.to_array().argmax(*args, **kwargs)
    
    def argpartition(self, *args, **kwargs): # pragma: no cover
        return self.to_array().argpartition(*args, **kwargs)

    def argsort(self, *args, **kwargs): # pragma: no cover
        return self.to_array().argsort(*args, **kwargs)

    def choose(self, *args, **kwargs): # pragma: no cover
        return self.to_array().choose(*args, **kwargs)

    def clip(self, *args, **kwargs): # pragma: no cover
        return self.to_array().clip(*args, **kwargs)

    def conj(self): # pragma: no cover
        return self.to_array().conj()

    def conjugate(self): # pragma: no cover
        return self.to_array().conjugate()

    def cumprod(self, *args, **kwargs): # pragma: no cover
        return self.to_array().cumprod(*args, **kwargs)

    def cumsum(self, *args, **kwargs): # pragma: no cover
        return self.to_array().cumsum(*args, **kwargs)

    def dot(self, *args, **kwargs): # pragma: no cover
        return self.to_array().dot(*args, **kwargs)

    def prod(self, *args, **kwargs): # pragma: no cover
        return self.to_array().prod(*args, **kwargs)

    def ptp(self, *args, **kwargs): # pragma: no cover
        return self.to_array().ptp(*args, **kwargs)

    def put(self, *args, **kwargs): # pragma: no cover
        return self.to_array().put(*args, **kwargs)

    def round(self, *args, **kwargs): # pragma: no cover
        return self.to_array().round(*args, **kwargs)

    def std(self, *args, **kwargs): # pragma: no cover
        return self.to_array().std(*args, **kwargs)

    def trace(self, *args, **kwargs): # pragma: no cover
        return self.to_array().trace(*args, **kwargs)

    def var(self, *args, **kwargs): # pragma: no cover
        return self.to_array().var(*args, **kwargs)

    def __matmul__(self, other): # pragma: no cover
        return self.to_array() @ other

    def __floordiv__(self, other): # pragma: no cover
        return self.to_array() // other

    def __mod__(self, other): # pragma: no cover
        return self.to_array() % other

    def __pow__(self, other): # pragma: no cover
        return self.to_array() ** other

    def __lshift__(self, other): # pragma: no cover
        return self.to_array() << other

    def __rshift__(self, other): # pragma: no cover
        return self.to_array() >> other

    def __and__(self, other): # pragma: no cover 
        return self.to_array() & other

    def __xor__(self, other): # pragma: no cover 
        return self.to_array() ^ other

    def __or__(self, other): # pragma: no cover
        return self.to_array() | other

    def __rmatmul__(self, other): # pragma: no cover
        return other @ self.to_array()

    def __rfloordiv__(self, other): # pragma: no cover
        return other // self.to_array()

    def __rmod__(self, other): # pragma: no cover
        return other % self.to_array()

    def __rpow__(self, other): # pragma: no cover
        return other ** self.to_array()

    def __rlshift__(self, other): # pragma: no cover
        return other << self.to_array() 

    def __rrshift__(self, other): # pragma: no cover
        return other >> self.to_array()

    def __rand__(self, other): # pragma: no cover
        return other & self.to_array()

    def __rxor__(self, other): # pragma: no cover
        return other ^ self.to_array()

    def __ror__(self, other): # pragma: no cover
        return other | self.to_array()

    def __imatmul__(self, other): # pragma: no cover
        raise TypeError("in-place matrix multiplication is not (yet) supported")

    def __ifloordiv__(self, other): # pragma: no cover
        self[:] = self.to_array() // other
        return self

    def __imod__(self, other): # pragma: no cover 
        self[:] =self.to_array() % other
        return self

    def __ipow__(self, other): # pragma: no cover
        self[:] = self.to_array() ** other
        return self

    def __ilshift__(self, other): # pragma: no cover
        self[:] = self.to_array() << other
        return self

    def __irshift__(self, other): # pragma: no cover
        self[:] = self.to_array() >> other
        return self

    def __iand__(self, other): # pragma: no cover 
        self[:] = self.to_array() & other
        return self

    def __ixor__(self, other): # pragma: no cover 
        self[:] = self.to_array() ^ other
        return self

    def __ior__(self, other): # pragma: no cover
        self[:] = self.to_array() | other
        return self
    
    # Representation
    
    def __repr__(self):
        name = 'sparse'
        n_spaces = len(name) - 5
        nums = repr(self.to_array())[5:].replace('\n', '\n' + n_spaces * ' ')
        return f'{name}{nums}'
    
    def __str__(self):
        arr = self.to_array()
        return str(arr)
            

class SparseVector:
    __doc__ = sparse.__doc__
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
            self.dct = {i.__index__(): j.__float__() for i, j in obj.items() if j}
            self.size = size
            if size is None: raise ValueError('must pass size if object is a dictionary')
        elif isinstance(obj, SparseVector):
            self.dct = obj.dct.copy()
            self.size = obj.size if size is None else size
        elif hasattr(obj, '__iter__'):
            self.dct = dct = {}
            for i, j in enumerate(obj):
                if j: dct[i] = j.__float__()
            self.size = len(obj) if size is None else size
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse array')
    
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
        
    def setflags(self, flag):
        if flag == 0: self.read_only = True # For backwards compatibility with arrays
        else: raise NotImplementedError(f'flag {flag} not yet implemented')
    
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
    
    def negative_keys(self):
        dct = self.dct
        for i in dct:
            if dct[i] < 0.: yield i
    
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
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = bool(self.dct)
        if keepdims: arr = np.array([arr])
        return arr
    
    def all(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = len(self.dct) == self.size
        if keepdims: arr = np.array([arr])
        return arr
    
    def sum(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = sum(self.dct.values())
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def mean(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = sum(self.dct.values()) / self.size if self.dct else 0.
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def max(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        dct = self.dct
        if dct:
            arr = max(dct.values())
            if arr < 0 and len(dct) < self.size: arr = 0
        elif self.size:
            arr = 0.
        else:
            raise ValueError('zero-size array reduction has no identity')
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def min(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        dct = self.dct
        if dct:
            arr = min(dct.values())
            if arr > 0 and len(dct) < self.size: arr = 0
        elif self.size:
            arr = 0.
        else:
            raise ValueError('zero-size array reduction has no identity')
        if keepdims: arr = SparseVector({0: arr} if arr else {}, size=1)
        return arr
    
    def to_array(self, dtype=None):
        if dtype is None: dtype = self.dtype
        arr = np.zeros(self.size, dtype=dtype)
        for i, j in self.dct.items(): arr[i] = j
        return arr
    astype = to_array
    
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
            return self[np.where(index)]
        if ndim == 1:
            arr = np.zeros(len(index))
            for n, i in enumerate(index):
                i = i.__index__()
                if i in dct: arr[n] = dct[i]
            return arr
        elif index.__class__ is slice:
            if index == open_slice:
                return self
            else:
                value = np.array([dct.get(i, 0.) for i in default_range(index, self.size)])
                value.setflags(0)
                return value
        elif ndim:
            raise IndexError(f'index can be at most 1-d, not {ndim}-d')    
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
                    f'boolean index is {ndim}-d but sparse array is 1-d '
                )
            index, = np.where(index)
        if ndim == 1:
            vd = get_ndim(value)
            if vd == 1:
                for i, j in zip(index, value): 
                    i = i.__index__()
                    if j: dct[i] = j.__float__()
                    elif i in dct: del dct[i]
            elif vd > 1:
                raise IndexError(
                    f'cannot broadcast {vd}-d array on to 1-d sparse array'
                )
            elif value:
                value = value.__float__()
                for i in index:
                    i = i.__index__()
                    dct[i] = value
            else:
                for i in index:
                    i = i.__index__()
                    if i in dct: del dct[i]
        elif index.__class__ is slice:
            if index == open_slice:
                if value is self: return
                vd = get_ndim(value)
                dct.clear()
                if value.__class__ is SparseVector:
                    dct.update(value.dct)
                elif vd == 1:
                    for i, j in enumerate(value):
                        if j: dct[i] = j.__float__()
                        elif i in dct: del dct[i]  
                elif vd == 0:
                    if value != 0.:
                        value = value.__float__()
                        for i in range(self.size): dct[i] = value
                else:
                    raise IndexError(
                        f'cannot broadcast {vd}-d array on to 1-d sparse array'
                    )
            else:
                self[default_range(index, self.size)] = value
        elif ndim:
            raise IndexError(f'index can be at most 1-d, not {ndim}-d')    
        elif hasattr(value, '__iter__'):
            raise IndexError(
                'cannot set an array element with a sequence'
            )
        else:
            index = index.__index__()
            if value:
                dct[index] = value.__float__()
            elif index in dct:
                del dct[index]
    
    def __iadd__(self, other):
        dct = self.dct
        if other.__class__ is SparseVector:
            if other.size == 1:
                if 0 in other.dct: 
                    other = other.dct[0]
                    for i in range(self.size):
                        if i in dct: 
                            j = dct[i] + other
                            if j: dct[i] = j
                            else: del dct[i]
                        else:
                            dct[i] = other
                return self
            for i, j in other.dct.items():
                if i in dct:
                    j += dct[i]
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = j
        elif hasattr(other, '__iter__'):
            if other.__len__() == 1: 
                self.__iadd__(other[0])
                return self
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = j.__float__()
                    j += dct[i]
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = j
        elif other:
            other = other.__float__()
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
            if other.size == 1:
                if 0 in other.dct: 
                    other = other.dct[0]
                    for i in range(self.size):
                        if i in dct: 
                            j = dct[i] - other
                            if j: dct[i] = j
                            else: del dct[i]
                        else:
                            dct[i] = -other
                return self
            for i, j in other.dct.items():
                if i in dct:
                    j = dct[i] - j
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -j
        elif hasattr(other, '__iter__'):
            if other.__len__() == 1: 
                self.__isub__(other[0])
                return self
            for i, j in enumerate(other):
                if not j: continue
                j = j.__float__()
                if i in dct:
                    j = dct[i] - j
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -j
        elif other:
            other = other.__float__()
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
            if other.size == 1:
                if 0 in other.dct: 
                    other = other.dct[0]
                    for i in dct: dct[i] *= other
                else:
                    dct.clear()
                return self
            other = other.dct
            for i in tuple(dct):
                if i in other:
                    dct[i] *= other[i]
                else:
                    del dct[i]
        elif hasattr(other, '__iter__'):
            if other.__len__() == 1: 
                self.__imul__(other[0])
                return self
            for i in tuple(dct):
                j = other[i]
                if j: dct[i] *= j.__float__()
                else: del dct[i]
        elif other:
            other = other.__float__()
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
            if other.size == 1:
                if 0 in other.dct: 
                    other = other.dct[0]
                    for i in dct: dct[i] /= other
                elif dct:
                    raise FloatingPointError('division by zero')
                return self
            other = other.dct
            for i in dct: dct[i] /= other[i]
        elif hasattr(other, '__iter__'):
            if other.__len__() == 1: 
                self.__itruediv__(other[0])
                return self
            for i in dct: dct[i] /= other[i].__float__()
        elif other:
            other = other.__float__()
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
            if other.__len__() == 1: 
                return self.__rtruediv__(other[0])
            for i, j in enumerate(other):
                if j:
                    if i in dct:
                        dct[i] = j.__float__() / dct[i] 
                    else:
                        raise FloatingPointError('division by zero')
                else: del dct[i]
        elif other:
            if len(dct) != self.size: raise FloatingPointError('division by zero')
            other = other.__float__()
            for i in dct: dct[i] = other / dct[i]
        else:
            dct = {}
        return SparseVector.from_dict(dct, self.size)
    
    value = SparseArray.value
    
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