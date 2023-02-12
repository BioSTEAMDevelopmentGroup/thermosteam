# -*- coding: utf-8 -*-
"""
"""
import numpy as np
__all__ = (
    'nonzero_items',
    'sparse_vector',
    'sparse_array',
    'SparseVector',
    'SparseLogicalVector',
    'SparseArray',
    'sparse',
    'reduce_ndim',
    'get_ndim',
)

open_slice = slice(None)
bools = frozenset([bool, np.bool_, np.dtype('bool')])

# TODO: With python 3.10, use strict=True zip kwarg
sparse_array_imath = """
def __i{name}__(self, other):
    if other.__class__ is SparseArray:
        rows = self.rows
        other_rows = other.rows
        dtype = self.dtype
        other_dtype = other.dtype
        if dtype is bool:
            if other_dtype is float:
                raise ValueError('cannot cast boolean to float')
        elif other_dtype is bool:
            other_rows = [SparseVector.from_dict({{i: 1. for i in row.set}}, row.size) for row in other_rows]
        if len(other_rows) == 1:
            other = other_rows[0]
            for i in rows: i._i{name}_sparse(other)
        else:
            for i, j in zip(rows, other_rows): i._i{name}_sparse(j)
    elif other.__class__ in SparseVectorSet:
        dtype = self.dtype
        other_dtype = other.dtype
        if dtype is bool:
            if other_dtype is float:
                raise ValueError('cannot cast boolean to float')
        elif other_dtype is bool:
            other = SparseVector.from_dict({{i: 1. for i in other.set}}, other.size)
        for i in self.rows: i._i{name}_sparse(other) 
    else:
        other, ndim, _ = reduce_ndim(other)
        if ndim == 0:
            for i in self.rows: i._i{name}_scalar(other) 
        elif ndim == 1:
            for i in self.rows: i._i{name}_array(other) 
        elif ndim == 2:
            rows = self.rows
            for i, j in zip(rows, other): i._i{name}_array(j)
        else:
            raise ValueError('shape mismatch between arrays')
    return self
"""
sparse_array_math = """
def __{name}__(self, other):
    rows = self.rows
    if other.__class__ is SparseArray:
        other_rows = other.rows
        dtype = self.dtype
        other_dtype = other.dtype
        if dtype is bool:
            if other_dtype is float:
                rows = [SparseVector.from_dict({{i: 1. for i in row.set}}, row.size) for row in rows]
        elif other_dtype is bool:
            other_rows = [SparseVector.from_dict({{i: 1. for i in row.set}}, row.size) for row in other_rows]
        if len(rows) == 1:
            row = rows[0]
            new = SparseArray.from_rows(
                [row._{name}_sparse(i) for i in other_rows]
            )
        elif len(other_rows) == 1:
            other = other_rows[0]
            new = SparseArray.from_rows(
                [i._{name}_sparse(other) for i in rows]
            )
        else:
            new = SparseArray.from_rows(
                [i._{name}_sparse(j) for i, j in zip(rows, other_rows)]
            )
    elif other.__class__ in SparseVectorSet:
        dtype = self.dtype
        other_dtype = other.dtype
        if dtype is bool:
            if other_dtype is float:
                rows = [SparseVector.from_dict({{i: 1. for i in row.set}}, row.size) for row in rows]
        elif other_dtype is bool:
            other = SparseVector.from_dict({{i: 1. for i in other.set}}, other.size)
        new = SparseArray.from_rows([
            i._{name}_sparse(other) for i in rows
        ])
    else:
        other, ndim, _ = reduce_ndim(other)
        if ndim == 0:
            new = SparseArray.from_rows([
                i._{name}_scalar(other) for i in rows
            ])
        elif ndim == 1:
            new = SparseArray.from_rows([
                i._{name}_array(other) for i in rows
            ])
        elif ndim == 2:
            new = SparseArray.from_rows(
                [i._{name}_array(j) for i, j in zip(rows, other)]
            )
        else:
            new = self.to_array().__{name}__(other)
    return new
"""
read_only = "if self.read_only: raise ValueError('assignment destination is read-only')"
not_read_only = ''
sparse_vector_imath = """
def __i{name}__(self, other):
    {read_only}
    if other.__class__ in SparseVectorSet:
        self._i{name}_sparse(other)
    elif other.__class__ is SparseArray:
        if len(other.rows) == 1:
            self._i{name}_sparse(other.rows[0])
        else:
            raise ValueError('shape mismatch between arrays')
    else:
        other, ndim, size = reduce_ndim(other)
        if ndim == 0:
            self._i{name}_scalar(other)
        elif ndim == 1:
            self._i{name}_array(other)
        elif ndim == 2:
            for i in other: self._i{name}_array(other)
        else:
            raise ValueError('shape mismatch between arrays')
    return self
"""

sparse_vector_math = """
def __{name}__(self, other):
    cls = other.__class__
    if cls is self.__class__:
        new = self._{name}_sparse(other)
    elif cls is SparseLogicalVector:
        other = SparseVector.from_dict({{i: 1. for i in other.set}}, other.size)
        new = self._{name}_sparse(other)
    elif cls is SparseVector: 
        self = SparseVector.from_dict({{i: 1. for i in self.set}}, self.size)
        new = self._{name}_sparse(other)
    elif other.__class__ is SparseArray:
        dtype = self.dtype
        other_dtype = other.dtype
        other_rows = other.rows
        if dtype is bool:
            if other_dtype is float:
                self = SparseVector.from_dict({{i: 1. for i in self.set}}, self.size)
        elif other_dtype is bool:
            other_rows = [SparseVector.from_dict({{i: 1. for i in row.set}}, row.size) for row in other_rows]
        new = SparseArray.from_rows([
            self._{name}_sparse(i) for i in other_rows
        ])
    else:
        other, ndim, size = reduce_ndim(other)
        if ndim == 0:
            new = self._{name}_scalar(other)
        elif ndim == 1:
            new = self._{name}_array(other)
        elif ndim == 2:
            new = SparseArray.from_rows([
                self._{name}_array(i) for i in other
            ])
        else:
            new = self.to_array().__{name}__(other)
    return new
"""
sparse_vector_comparison_math = sparse_vector_math + """
def _{name}_sparse(self, other):
    dct = self.dct
    size = self.size
    other_size = other.size
    if size == other_size:
        other = other.dct
        default = 0. {sign} 0.
        if default:
            new = {{*range(size)}}
            new.difference_update([i for i in [*dct, *other] if not dct.get(i, 0.) {sign} other.get(i, 0.)])
        else:
            new = {{i for i in [*dct, *other] if dct.get(i, 0.) {sign} other.get(i, 0.)}}
    elif other_size == 1:
        other = other.dct.get(0, 0.)
        default = 0. {sign} other
        if default:
            new = {{i for i in range(size) if i not in dct or dct[i] {sign} other}}
        else:
            new = {{i for i in dct if dct[i] {sign} other}}
    elif size == 1 and other_size:  
        size = other_size
        value = dct.get(0, 0.)
        other = other.dct
        default = value {sign} 0.
        if default:
            new = {{i for i in range(size) if i not in other or value {sign} other[i]}}
        else:
            new = {{i for i in other if value {sign} other[i]}}
    else:
        raise ValueError('shape mismatch between arrays')
    return SparseLogicalVector.from_set(new, size)
    
def _{name}_scalar(self, other):
    dct = self.dct
    size = self.size
    other = float(other)
    default = 0. {sign} other
    if default:
        new = {{i for i in range(size) if i not in dct or dct[i] {sign} other}}
    else:
        new = {{i for i in dct if dct[i] {sign} other}}
    return SparseLogicalVector.from_set(new, size)
        
def _{name}_array(self, other):
    dct = self.dct
    size = self.size
    other_size = len(other)
    if size == other_size:
        new = {{i for i in range(size) if dct.get(i, 0.) {sign} float(other[i])}}
    elif size == 1 and other_size: 
        size = other_size
        value = dct.get(0, 0.)
        new = {{i for i in range(size) if value {sign} float(other[i])}}
    else:
        raise ValueError('shape mismatch between arrays')
    return SparseLogicalVector.from_set(new, size)
"""
sparse_vector_not_optimized = sparse_vector_math + sparse_vector_imath + """
def _{name}_sparse(self, other):
    return SparseVector.from_dict({{i: float(j) for i, j in enumerate(self.to_array().__{name}__(other.to_array())) if j}})
    
def _{name}_scalar(self, other):
    return SparseVector.from_dict({{i: float(j) for i, j in enumerate(self.to_array().__{name}__(other)) if j}})
        
def _{name}_array(self, other):
    return SparseVector.from_dict({{i: float(j) for i, j in enumerate(self.to_array().__{name}__(other)) if j}})

def _i{name}_sparse(self, other):
    self[:] = self.to_array().__{name}__(other.to_array())
    return self
    
def _i{name}_scalar(self, other):
    self[:] = self.to_array().__{name}__(other)
    return self
        
def _i{name}_array(self, other):
    self[:] = self.to_array().__{name}__(other)
    return self
"""

sparse_logical_vector_math_pseudo_optimized = sparse_vector_math + """
def _{name}_sparse(self, other):
    return self.copy()._i{name}_sparse(other)
    
def _{name}_scalar(self, other):
    if other.__class__ in bools:
        return self.copy()._i{name}_scalar(other)
    else:
        self = SparseVector.from_dict({{i: 1. for i in self.set}}, self.size)
        return self._{name}_scalar(other)

def _{name}_array(self, other):
    for i in other:
        has_bool = i.__class__ in bools
        break
    else:
        has_bool = False
    if has_bool:
        return self.copy()._i{name}_array(other)
    else:
        self = SparseVector.from_dict({{i: 1. for i in self.set}}, self.size)
        return self._{name}_array(other)

"""

def default_range(slice, max):
    return range(
        0 if slice.start is None else slice.start,
        max if slice.stop is None else slice.stop, 
        1 if slice.step is None else slice.step,
    )

def unpack_index(index, ndim):
    indim = len(index)
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
    Convert 1-d array to a SparseVector or SparseLogicalVector object.

    """
    if arr.__class__ in SparseVectorSet:
        return arr.copy() if copy else arr
    else:
        for i in arr:
            if i.__class__ in bools:
                return SparseLogicalVector(arr, size)
        return SparseVector(arr, size)

def nonzero_items(arr):
    if arr.__class__ in SparseSet:
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
    Create a sparse array from an array-like object:
    
    >>> from thermosteam.base import sparse
    >>> sa = sparse([[0, 1, 2], [3, 2, 0]])
    >>> sa
    sparse([[0., 1., 2.],
            [3., 2., 0.]])
    
    Create a sparse array from an a list of dictionaries of index-nonzero value pairs:
    
    >>> sa = sparse(
    ...     [{1: 1, 2: 2},
    ...      {0: 3, 1: 2}],
    ...     vector_size=3,
    ... )
    >>> sa
    sparse([[0., 1., 2.],
            [3., 2., 0.]])
    
    Sparse arrays support arithmetic operations just like dense arrays:
    
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
    
    Sparse arrays also support logical operations:
    
    >>> sa = sparse([[True, False, True, False],
    ...              [False, True, True, False]])
    >>> sa ^ True # XOR
    sparse([[False,  True, False,  True],
            [ True, False, False,  True]])
    
    """
    if arr.__class__ in SparseSet:
        return arr
    else:
        ndim = get_ndim(arr)
        if ndim == 1:
            for i in arr:
                if i.__class__ in bools:
                    return SparseLogicalVector(arr, vector_size)
            return SparseVector(arr, vector_size)
        elif ndim == 2:
            return SparseArray(arr, vector_size)
        else:
            raise ValueError(f'cannot convert {ndim}-d object to a sparse array or vector')
    
def get_ndim(value):
    if hasattr(value, 'ndim'): return value.ndim
    ndim = 0
    while hasattr(value, '__iter__'):
        ndim += 1
        try: value = value.__iter__().__next__()
        except: break
    return ndim

def reduce_ndim(value):
    if hasattr(value, 'ndim'): 
        ndim = value.ndim
        while ndim:
            size = len(value)
            if size != 1: break
            ndim -= 1
            value = value[0]
        else:
            size = 0
        return value, ndim, size
    else:
        ndim = size = 0
        while hasattr(value, '__len__'): 
            size = len(value)
            if size == 1:
                value = value[0]
            else: 
                dummy = value
                while hasattr(dummy, '__iter__'):
                    ndim += 1
                    try: dummy = dummy.__iter__().__next__()
                    except: break
                break
        return value, ndim, size

def get_array_properties(index):
    if hasattr(index, 'ndim'): return index.ndim, index.dtype in bools
    ndim = 0
    while hasattr(index, '__len__'):
        ndim += 1
        if index: index = index[0]
        else: break
    return ndim, index.__class__ in bools

def sum_sparse_vectors(svs):
    if svs: dtype = svs[0].dtype
    else: return {}
    if dtype is bool:
        sets = [i.set for i in svs]
        keys = {i for set in sets for i in set}
        dct = {i: sum([1. for j in sets if i in j]) for i in keys}
    elif dtype is float:
        dcts = [i.dct for i in svs]
        keys = {i for dct in dcts for i in dct}
        dct = {i: x for i in keys if (x:=sum([j[i] for j in dcts if i in j]))}
    else:
        raise RuntimeError('unexpected dtype')
    return dct

class SparseArray:
    __doc__ = sparse.__doc__
    __slots__ = ('rows',)
    ndim = 2
    
    def __init__(self, obj=None, vector_size=None):
        if obj is None:
            self.rows = []
        elif hasattr(obj, '__iter__'):
            self.rows = [sparse_vector(row, size=vector_size) for row in obj]
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse array')
    
    @property
    def dtype(self):
        for i in self.rows: return i.dtype
    
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
        for i in self.rows: i.set.clear()
    
    def copy(self):
        return SparseArray.from_rows([i.copy() for i in self.rows])
    
    def __int__(self):
        value, ndim, _ = reduce_ndim(self)
        if ndim == 0:
            return int(value)
        else:
            raise ValueError(
                'cannot cast array with more than one element to a scalar'
            )
    
    def __float__(self):
        value, ndim, _ = reduce_ndim(self)
        if ndim == 0:
            return float(value)
        else:
            raise ValueError(
                'cannot cast array with more than one element to a scalar'
            )
    
    def __bool__(self):
        value, ndim, _ = reduce_ndim(self)
        if ndim == 0:
            return bool(value)
        else:
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
        if self.dtype is float:
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
            for j in row.set:
                m.append(i); n.append(j)
        return m, n
    nonzero = nonzero_index
    
    def nonzero_values(self):
        for row in self.rows:
            yield from row.nonzero_values()
    
    def nonzero_items(self):
        dtype = self.dtype
        if dtype is bool: 
            for i, row in enumerate(self.rows):
                for j in row.set: yield ((i, j), True)
        elif dtype is float:
            for i, row in enumerate(self.rows):
                for j, k in row.dct.items(): yield ((i, j), k)
        else:
            raise RuntimeError('unexpected dtype')
    
    def nonzero_rows(self):
        return [i for i, j in enumerate(self.rows) if j.set]
    
    def nonzero_keys(self):
        keys = set()
        for i in self.rows: keys.update(i.set)
        return keys
    
    def positive_index(self):
        if self.dtype is bool: return self.nonzero_index()
        m = []; n = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value > 0.:
                    m.append(i); n.append(j)
        return m, n
    
    def negative_index(self):
        if self.dtype is bool: return [], []
        m = []; n = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value < 0.:
                    m.append(i); n.append(j)
        return m, n
    
    def negative_rows(self):
        if self.dtype is bool: return []
        m = []
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value < 0.: 
                    m.append(i)
                    break
        return m
    
    def negative_keys(self):
        if self.dtype is bool: return set()
        n = set()
        for i, row in enumerate(self.rows):
            for j, value in row.dct.items():
                if value < 0.: n.add(j)
        return n
    
    def has_negatives(self):
        if self.dtype is bool: return False
        for i, row in enumerate(self.rows):
            if row.has_negatives(): return True
        return False
    
    def remove_negatives(self):
        for i in self.rows: i.remove_negatives()
    
    def shares_data_with(self, other):
        if other.__class__ is SparseArray:
            for i in self.rows:
                for j in other.rows:
                    if i is j: return True
        else:
            for i in self.rows:
                if i is other: return True
        return False
    
    def to_flat_array(self, arr=None):
        vector_size = self.vector_size
        rows = self.rows
        N = rows.__len__()
        if arr is None: arr = np.zeros(N * vector_size)
        else: arr[:] = 0.
        for i in range(N):
            row = rows[i]
            for j, value in row.nonzero_items(): 
                arr[i * vector_size + j] = value
        return arr
        
    def from_flat_array(self, arr):
        rows = self.rows
        vector_size = self.vector_size
        dtype = self.dtype
        if dtype is float:
            dcts = [i.dct for i in rows]
            for i in dcts: i.clear()
            for j, value in enumerate(arr):
                if value: 
                    i = int(j / vector_size)
                    j -= i * vector_size
                    dcts[i][j] = value
        elif dtype is bool:
            sets = [i.set for i in rows]
            for i in sets: i.clear()
            for j, value in enumerate(arr):
                if value: 
                    i = int(j / vector_size)
                    j -= i * vector_size
                    sets[i].add(j)
        else:
            raise RuntimeError('unexpected dtype')
    
    def sparse_equal(self, other):
        return all([i.set == j.set for i, j in zip(self.rows, sparse_array(other).rows)])
    
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
                md, misbool = get_array_properties(m)
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
                    dtype = self.dtype
                    if dtype is float:
                        if nd == 0:
                            return np.array([rows[i].dct.get(n, 0.) for i in m])
                        elif nd == 1:
                            return np.array([rows[i].dct.get(j, 0.) for i, j in zip(m, n)])
                        else:
                            raise IndexError(f'column index can be at most 1-d, not {nd}-d')
                    elif dtype is bool:
                        if nd == 0:
                            return np.array([n in rows[i].set for i in m])
                        elif nd == 1:
                            return np.array([j in rows[i].set for i, j in zip(m, n)])
                        else:
                            raise IndexError(f'column index can be at most 1-d, not {nd}-d')
                    else:
                        raise RuntimeError('unexpected dtype')
                else:
                    raise IndexError(f'row index can be at most 1-d, not {md}-d')
        else:
            ndim, has_bool = get_array_properties(index)
            if has_bool:
                if ndim == 1: 
                    return SparseArray.from_rows([rows[i] for i, j in enumerate(index) if j])
                else:
                    return self[index.nonzero() if hasattr(index, 'nonzero') else np.nonzero(index)]
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
        value, vd, _ = reduce_ndim(value)
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
                if vd == 0.:
                    for i in rows: i[n] = value
                elif vd == 1:
                    nd, nisbool = get_array_properties(n)
                    if nd:
                        if nisbool: n, = n.nonzero() if hasattr(n, 'nonzero') else np.nonzero(n)
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
                md, misbool = get_array_properties(m)
                if md == 0:
                    rows[m][n] = value
                elif md == 1:
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
            else:
                md = get_ndim(m)
                if md == 0: 
                    rows[m][n] = value
                elif md == 1: 
                    dtype = self.dtype
                    if dtype is float:
                        nd = get_ndim(n)
                        if nd == 0:
                            if vd == 0:
                                if value:
                                    value = float(value)
                                    for i in m: 
                                        rows[i].dct[n] = value
                                else:
                                    for i in m: 
                                        dct = rows[i].dct
                                        if n in dct: del dct[n]
                            elif vd == 1:
                                for i, k in zip(m, value): 
                                    dct = rows[i].dct
                                    if k: dct[n] = float(k)
                                    elif n in dct: del dct[n]
                            else:
                                raise IndexError(
                                    'cannot set an array element with a sequence'
                                )
                        elif nd == 1:
                            if vd == 0:
                                if value:
                                    value = float(value)
                                    for i, j in zip(m, n): 
                                        rows[i].dct[j] = value
                                else:
                                    for i, j in zip(m, n): 
                                        dct = rows[i].dct
                                        if j in dct: del dct[j]
                            elif vd == 1:
                                for i, j, k in zip(m, n, value): 
                                    dct = rows[i].dct
                                    if k: dct[j] = float(k)
                                    elif j in dct: del dct[j]
                            else:
                                raise IndexError(
                                    'cannot set an array element with a sequence'
                                )
                        else:
                            raise IndexError(f'column index can be at most 1-d, not {nd}-d')
                    elif dtype is bool:
                        if vd == 0:
                            if value:
                                for i, j in zip(m, n): 
                                    rows[i].set.add(j)
                            else:
                                for i, j in zip(m, n): 
                                    rows[i].set.discard(j)
                        elif vd == 1:
                            for i, j, k in zip(m, n, value): 
                                set = rows[i].set
                                if k: set.add(j)
                                else: set.discard(j)
                        else:
                            raise IndexError(
                                'cannot set an array element with a sequence'
                            )
                else:
                    raise IndexError(f'row index can be at most 1-d, not {md}-d')
        else:
            ndim, has_bool = get_array_properties(index)
            if has_bool:
                if ndim == 1: 
                    if vd == 0:
                        for i, j in enumerate(index):
                            if j: rows[i][:] = value
                    else:
                        for i, j in enumerate(index):
                            if j: rows[i][:] = value[i]
                else:
                    self[index.nonzero() if hasattr(index, 'nonzero') else np.nonzero(index)] = value
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
        arr = np.zeros([N, self.vector_size], dtype=dtype or self.dtype)
        for i in range(N):
            row = rows[i]
            dtype = row.dtype 
            if dtype is bool:
                for j in row.set: arr[i, j] = True
            elif dtype is float:
                for j, value in row.dct.items(): arr[i, j] = value
            else:
                raise RuntimeError('unexpected dtype')
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
            if keepdims: arr = SparseArray.from_rows([SparseLogicalVector.from_set({0} if arr else set(), 1)])
        elif axis == 0:
            if rows: 
                keys = set(rows[0].set)
                for i in rows[1:]: keys.intersection_update(i.set)
                arr = SparseLogicalVector.from_set(keys, self.vector_size)
            else:
                arr = SparseLogicalVector.from_set(set(), 0)
            if keepdims: arr = SparseArray.from_rows([arr])
        elif axis == 1:
            if keepdims:
                arr = SparseArray.from_rows(
                    [SparseLogicalVector.from_set({0} if i.all() else {}, 1) for i in rows]
                )
            else:
                arr = SparseLogicalVector.from_set({i for i, j in enumerate(rows) if j.all()}, len(rows))
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def any(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = any([i.any() for i in rows])
            if keepdims: arr = SparseArray.from_rows([SparseLogicalVector.from_set({0} if arr else set(), 1)])
        elif axis == 0:
            keys = {j for i in rows for j in i.set}
            arr = SparseLogicalVector.from_set(keys, self.vector_size)
            if keepdims: arr = SparseArray.from_rows([arr])
        elif axis == 1:
            if keepdims:
                arr = SparseArray.from_rows(
                    [SparseLogicalVector.from_set({0} if i.any() else {}, 1) for i in rows]
                )
            else:
                arr = SparseLogicalVector.from_set({i for i, j in enumerate(rows) if j.any()}, len(rows))
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
            
    def sum(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = sum([i.sum() for i in rows])
            if keepdims: arr = SparseArray.from_rows([SparseVector.from_dict({0: arr} if arr else {}, 1)])
        elif axis == 0:
            arr = SparseVector.from_dict(sum_sparse_vectors(rows), size=self.vector_size)
            if keepdims: arr = SparseArray.from_rows([arr])
        elif axis == 1:
            if keepdims:
                svs = []
                for i in rows:
                    x = i.sum()
                    svs.append(SparseVector.from_dict({0: x} if x else {}, 1))
                arr = SparseArray.from_rows(svs)
            else:
                dct = {}
                for i, j in enumerate(rows):
                    x = j.sum()
                    if x: dct[i] = x
                arr = SparseVector.from_dict(dct, len(rows))
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def mean(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = sum([i.sum() for i in rows]) / sum([i.size for i in self.rows])
            if keepdims: arr = SparseArray.from_rows([SparseVector.from_dict({0: arr} if arr else {}, 1)])
            return arr
        elif axis == 0:
            arr = SparseVector.from_dict(sum_sparse_vectors(rows), size=self.vector_size)
            arr /= rows.__len__()
            if keepdims: arr = SparseArray.from_rows([arr])
            return arr
        elif axis == 1:
            if keepdims:
                svs = []
                for i in rows:
                    x = i.sum()
                    svs.append(SparseVector.from_dict({0: x / i.size} if x else {}, 1))
                arr = SparseArray.from_rows(svs)
            else:
                dct = {}
                for i, j in enumerate(rows):
                    x = j.sum()
                    if x: dct[i] = x / j.size
                arr = SparseVector.from_dict(dct, len(rows))
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def max(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = max([i.max() for i in rows])
            if keepdims: arr = SparseArray.from_rows([SparseVector.from_dict({0: arr}, 1)])
        elif axis == 0:
            keys = set()
            dtype = self.dtype
            if dtype is bool:
                sets = [i.set for i in rows]
                keys = {j for i in sets for j in i}
                arr = SparseVector.from_dict(
                    {i: 1. for i in keys if any([i in j for j in sets])},
                    self.vector_size
                )
            elif dtype is float:
                dcts = [i.dct for i in rows]
                keys = {j for i in dcts for j in i}
                dct = {}
                for i in keys:
                    x = max([j[i] if i in j else 0. for j in dcts])
                    if x: dct[i] = x
                arr = SparseVector.from_dict(dct, self.vector_size)
            else:
                raise RuntimeError('unexpected dtype')
            if keepdims: arr = SparseArray.from_rows([arr])
        elif axis == 1:
            if keepdims:
                svs = []
                for i in rows:
                    x = i.max()
                    svs.append(SparseVector.from_dict({0: x} if x else {}, 1))
                arr = SparseArray.from_rows(svs)
            else:
                dct = {}
                for i, j in enumerate(rows):
                    x = j.max()
                    if x: dct[i] = x
                arr = SparseVector.from_dict(dct, len(rows))
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    def min(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            arr = min([i.min() for i in rows])
            if keepdims: arr = SparseArray.from_rows([SparseVector.from_dict({0: arr}, 1)])
        elif axis == 0:
            dtype = self.dtype
            if dtype is bool:
                sets = [i.set for i in rows]
                keys = {j for i in sets for j in i}
                arr = SparseVector.from_dict(
                    {i: 1. for i in keys if all([i in j for j in sets])},
                    self.vector_size
                )
            elif dtype is float:
                dcts = [i.dct for i in rows]
                keys = {j for i in dcts for j in i}
                dct = {}
                for i in keys:
                    x = min([j[i] if i in j else 0. for j in dcts])
                    if x: dct[i] = x
                arr = SparseVector.from_dict(
                    dct, self.vector_size
                )
            else:
                raise RuntimeError('unexpected dtype')
            if keepdims: arr = SparseArray([arr])
        elif axis == 1:
            if keepdims:
                svs = []
                for i in rows:
                    x = i.min()
                    svs.append(SparseVector.from_dict({0: x} if x else {}, 1))
                arr = SparseArray.from_rows(svs)
            else:
                dct = {}
                for i, j in enumerate(rows):
                    x = j.min()
                    if x: dct[i] = x
                arr = SparseVector.from_dict(dct, len(rows))
        else:
            raise ValueError('axis is out of bounds for 2-d sparse array')
        return arr
    
    exec(sparse_array_math.format(name='add'))
    exec(sparse_array_math.format(name='sub'))
    exec(sparse_array_math.format(name='mul'))
    exec(sparse_array_math.format(name='truediv'))
    exec(sparse_array_math.format(name='and'))
    exec(sparse_array_math.format(name='xor'))
    exec(sparse_array_math.format(name='or'))
    exec(sparse_array_math.format(name='eq'))
    exec(sparse_array_math.format(name='ne'))
    exec(sparse_array_math.format(name='gt'))
    exec(sparse_array_math.format(name='lt'))
    exec(sparse_array_math.format(name='ge'))
    exec(sparse_array_math.format(name='le'))
    exec(sparse_array_imath.format(name='add'))
    exec(sparse_array_imath.format(name='sub'))
    exec(sparse_array_imath.format(name='mul'))
    exec(sparse_array_imath.format(name='truediv'))
    exec(sparse_array_imath.format(name='and'))
    exec(sparse_array_imath.format(name='xor'))
    exec(sparse_array_imath.format(name='or'))
    
    def __neg__(self):
        return SparseArray.from_rows([-i for i in self.rows])
    
    def __invert__(self):
        return SparseArray.from_rows([~i for i in self.rows])
    
    def __rtruediv__(self, other):
        if hasattr(other, '__len__'):
            return other / self.to_array()
        else:
            return SparseArray.from_rows([
                i.__rtruediv__(other) for i in self.rows
            ])
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return -self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rand__(self, other): # pragma: no cover
        return self & other

    def __rxor__(self, other): # pragma: no cover
        return self ^ other

    def __ror__(self, other): # pragma: no cover
        return self | other
    
    # Not yet optimized methods
    
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
    __slots__ = ('dct', 'read_only', 'size')
    ndim = 1
    dtype = float
    
    def __init__(self, obj=None, size=None):
        self.read_only = False
        if obj is None:
            self.dct = {}
            if size is None: raise ValueError('must pass size if no object given')
            self.size = size
        elif isinstance(obj, dict):
            self.dct = {i: float(j) for i, j in obj.items() if j}
            self.size = size
            if size is None: raise ValueError('must pass size if object is a dictionary')
        elif isinstance(obj, SparseVector):
            self.dct = obj.dct.copy()
            self.size = obj.size if size is None else size
        elif hasattr(obj, '__iter__'):
            self.dct = dct = {}
            for i, j in enumerate(obj):
                if j: dct[i] = float(j)
            self.size = len(obj) if size is None else size
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse array')
    
    def mix_from(self, others):
        if others: 
            other_dcts = [i.dct for i in others]
            dct = self.dct
            repeated = 0
            dcts = []
            for i in other_dcts:
                if i is dct: repeated += 1
                else: dcts.append(i)
            if repeated == 0:
                dct.clear()
            elif repeated == 1:
                dcts.append(dct.copy())
            elif repeated:
                dcts.append({i: j * repeated for i, j in dct.items()})
            keys = {i for dct in dcts for i in dct}
            for i in keys:
                x = sum([j[i] for j in dcts if i in j])
                if x: dct[i] = x
                elif x in dct: del dct[i]
        else:
            self.dct.clear()
    
    def shares_data_with(self, other):
        if other.__class__ is SparseArray:
            for i in other.rows:
                if self is i: return True
        else:
            return self is other
        return False
    
    @property
    def set(self):
        return self.dct
    
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
            arr[:] = 0.
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
        return sorted(self.dct)
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
        if keepdims: arr = SparseLogicalVector.from_set({0} if arr else set(), size=1)
        return arr
    
    def all(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = len(self.dct) == self.size
        if keepdims: arr = SparseLogicalVector.from_set({0} if arr else set(), size=1)
        return arr
    
    def sum(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = sum(self.dct.values())
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
        return arr
    
    def mean(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = sum(self.dct.values()) / self.size if self.dct else 0.
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
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
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
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
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
        return arr
    
    def to_array(self, dtype=None):
        arr = np.zeros(self.size, dtype=dtype or self.dtype)
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
        ndim, has_bool = get_array_properties(index)
        if has_bool:
            return self[index.nonzero() if hasattr(index, 'nonzero') else np.nonzero(index)]
        if ndim == 1:
            arr = np.zeros(len(index))
            for n, i in enumerate(index):
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
            return dct.get(index, 0.)

    def __setitem__(self, index, value):
        if self.read_only: raise ValueError('assignment destination is read-only')
        dct = self.dct
        if index.__class__ is tuple:
            index, = unpack_index(index, self.ndim)
        ndim, has_bool = get_array_properties(index)
        value, vd, _ = reduce_ndim(value)
        if has_bool: 
            if ndim != 1:
                raise IndexError(
                    f'boolean index is {ndim}-d but sparse array is 1-d '
                )
            index, = index.nonzero() if hasattr(index, 'nonzero') else np.nonzero(index)
        if ndim == 1:
            if vd == 1:
                for i, j in zip(index, value): 
                    if j: dct[i] = float(j)
                    elif i in dct: del dct[i]
            elif vd > 1:
                raise IndexError(
                    f'cannot broadcast {vd}-d array on to 1-d sparse array'
                )
            elif value:
                value = float(value)
                for i in index:
                    dct[i] = value
            else:
                for i in index:
                    if i in dct: del dct[i]
        elif index.__class__ is slice:
            if index == open_slice:
                if value is self: return
                dct.clear()
                if value.__class__ is SparseVector:
                    dct.update(value.dct)
                elif vd == 1:
                    for i, j in enumerate(value):
                        if j: dct[i] = float(j)
                        elif i in dct: del dct[i]  
                elif vd == 0:
                    if value:
                        value = float(value)
                        for i in range(self.size): dct[i] = value
                else:
                    raise IndexError(
                        f'cannot broadcast {vd}-d array on to 1-d sparse array'
                    )
            else:
                self[default_range(index, self.size)] = value
        elif ndim:
            raise IndexError(f'index can be at most 1-d, not {ndim}-d')    
        elif vd:
            raise IndexError(
                'cannot set an array element with a sequence'
            )
        elif value:
            dct[index] = float(value)
        elif index in dct:
            del dct[index]
    
    exec(sparse_vector_math.format(name='add'))
    exec(sparse_vector_math.format(name='sub'))
    exec(sparse_vector_math.format(name='mul'))
    exec(sparse_vector_math.format(name='truediv'))
    exec(sparse_vector_math.format(name='and'))
    exec(sparse_vector_math.format(name='xor'))
    exec(sparse_vector_math.format(name='or'))
    exec(sparse_vector_math.format(name='eq'))
    exec(sparse_vector_math.format(name='ne'))
    exec(sparse_vector_comparison_math.format(name='gt', sign='>'))
    exec(sparse_vector_comparison_math.format(name='lt', sign='<'))
    exec(sparse_vector_comparison_math.format(name='ge', sign='>='))
    exec(sparse_vector_comparison_math.format(name='le', sign='<='))
    exec(sparse_vector_imath.format(name='add', read_only=read_only))
    exec(sparse_vector_imath.format(name='sub', read_only=read_only))
    exec(sparse_vector_imath.format(name='mul', read_only=read_only))
    exec(sparse_vector_imath.format(name='truediv', read_only=read_only))
    exec(sparse_vector_not_optimized.format(name='and', read_only=not_read_only))
    exec(sparse_vector_not_optimized.format(name='xor', read_only=not_read_only))
    exec(sparse_vector_not_optimized.format(name='or', read_only=not_read_only))
    
    __float__ = SparseArray.__float__
    __bool__ = SparseArray.__bool__
    __int__ = SparseArray.__int__
    __radd__ = SparseArray.__radd__
    __rsub__ = SparseArray.__rsub__
    __rmul__ = SparseArray.__rmul__
    __rand__ = SparseArray.__rand__
    __rxor__ = SparseArray.__rxor__
    __ror__ = SparseArray.__ror__
    
    def _add_scalar(self, other):
        dct = self.dct
        size = self.size
        if other:
            new = {}
            other = float(other)
            for i in range(size):
                if i in dct: 
                    j = dct[i] + other
                    if j: new[i] = j
                else:
                    new[i] = other
        else:
            new = dct.copy()
        return SparseVector.from_dict(new, size)
    
    def _add_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            new = dct.copy()
            for i, j in other_dct.items():
                if i in dct:
                    j = dct[i] + j
                    if j: new[i] = j
                    else: del new[i]
                else:
                    new[i] = j
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {}
                for i in range(other_size):
                    if i in other_dct:
                        j = value + other_dct[i]
                        if j: new[i] = j
                    else:
                        new[i] = value
            else:
                new = other_dct.copy()
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                new = {}
                for i in range(size):
                    if i in dct: 
                        j = dct[i] + other
                        if j: new[i] = j
                    else:
                        new[i] = other
            else:
                new = dct.copy()
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _add_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            new = dct.copy()
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] + float(j)
                    if j: new[i] = j
                    else: del new[i]
                else:
                    new[i] = float(j)
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i: j for i in range(other_size) if (j:=value + float(other[i]))}
            else:
                new = {i: float(j) for i in range(other_size) if (j:=other[i])}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _iadd_scalar(self, other):
        dct = self.dct
        size = self.size
        if not other: return self
        other = float(other)
        for i in range(size):
            if i in dct: 
                j = dct[i] + other
                if j: dct[i] = j
                else: del dct[i]
            else:
                dct[i] = other
        return self
    
    def _iadd_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            for i, j in other_dct.items():
                if i in dct:
                    j += dct[i]
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = j
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i in range(other_size):
                    if i in other_dct:
                        j = value + other_dct[i]
                        if j: dct[i] = j
                    else:
                        dct[i] = value
            else:
                dct.update(other_dct)
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                for i in range(size):
                    if i in dct: 
                        j = dct[i] + other
                        if j: dct[i] = j
                        else: del dct[i]
                    else:
                        dct[i] = other
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _iadd_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] + float(j)
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = float(j)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i in range(other_size):
                    j = value + float(other[i])
                    if j: dct[i] = j
            else:
                for i in range(other_size):
                    j = other[i]
                    if j: dct[i] = float(j)
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _sub_scalar(self, other):
        dct = self.dct
        size = self.size
        if other:
            new = {}
            other = -float(other)
            for i in range(size):
                if i in dct: 
                    j = dct[i] + other
                    if j: new[i] = j
                else:
                    new[i] = other
        else:
            new = dct.copy()
        return SparseVector.from_dict(new, size)
    
    def _sub_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            new = dct.copy()
            for i, j in other_dct.items():
                if i in dct:
                    j = dct[i] - j
                    if j: new[i] = j
                    else: del new[i]
                else:
                    new[i] = -j
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {}
                for i in range(other_size):
                    if i in other_dct:
                        j = value - other_dct[i]
                        if j: new[i] = j
                    else:
                        new[i] = value
            else:
                new = {i: -j for i, j in other_dct.items()}
        elif other_size == 1:
            if 0 in other_dct: 
                other = -other_dct[0]
                new = {}
                for i in range(size):
                    if i in dct: 
                        j = dct[i] + other
                        if j: new[i] = j
                    else:
                        new[i] = other
            else:
                new = dct.copy()
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _sub_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            new = dct.copy()
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] - float(j)
                    if j: new[i] = j
                    else: del new[i]
                else:
                    new[i] = -float(j)
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i: j for i in range(other_size) if (j:=value - float(other[i]))}
            else:
                new = {i: -float(j) for i in range(other_size) if (j:=other[i])}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _isub_scalar(self, other):
        dct = self.dct
        size = self.size
        if not other: return self
        other = -float(other)
        for i in range(size):
            if i in dct: 
                j = dct[i] + other
                if j: dct[i] = j
                else: del dct[i]
            else:
                dct[i] = other
        return self
    
    def _isub_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            for i, j in other_dct.items():
                if i in dct:
                    j = dct[i] - j
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -j
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i in range(other_size):
                    if i in other_dct:
                        j = value - other_dct[i]
                        if j: dct[i] = j
                    else:
                        dct[i] = value
            else:
                for i, j in other_dct.items(): dct[i] = -j
        elif other_size == 1:
            if 0 in other_dct: 
                other = -other_dct[0]
                for i in range(size):
                    if i in dct: 
                        j = dct[i] + other
                        if j: dct[i] = j
                        else: del dct[i]
                    else:
                        dct[i] = other
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _isub_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] - float(j)
                    if j: dct[i] = j
                    else: del dct[i]
                else:
                    dct[i] = -float(j)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i in range(other_size):
                    j = value - float(other[i])
                    if j: dct[i] = j
            else:
                for i in range(other_size):
                    j = other[i]
                    if j: dct[i] = -float(j)
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _mul_scalar(self, other):
        dct = self.dct
        if other:
            other = float(other)
            new= {i: j * other for i, j in dct.items()}
        else:
            new = {}
        return SparseVector.from_dict(new, self.size)
    
    def _mul_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            new = {i: dct[i] * other_dct[i] for i in dct.keys() & other_dct.keys()}
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i: value * j for i, j in other_dct.items()}
            else:
                new = {}
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                new = {i: j * other for i, j in dct.items()}
            else:
                new = {}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _mul_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            new = {i: dct[i] * float(j) for i in dct if (j:=other[i])}
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i: value * float(j) for i in range(other_size) if (j:=other[i])}
            else:
                new = {}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _imul_scalar(self, other):
        dct = self.dct
        if other:
            other = float(other)
            for i in dct: dct[i] *= other
        else:
            dct.clear()
        return self
    
    def _imul_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            for i in tuple(dct):
                if i in other_dct:
                    dct[i] *= other_dct[i]
                else:
                    del dct[i]
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i, j in other_dct.items(): dct[i] = value * j
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                for i in dct: dct[i] *= other
            else:
                dct.clear()
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _imul_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i in tuple(dct):
                j = other[i]
                if j: dct[i] *= float(j)
                else: del dct[i]
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i in range(other_size):
                    j = other[i]
                    if j: dct[i] = value * float(j)
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _truediv_scalar(self, other):
        dct = self.dct
        other = float(other)
        new = {i: j / other for i, j in dct.items()}
        return SparseVector.from_dict(new, self.size)
    
    def _truediv_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            if len(dct) > len(other_dct): raise ZeroDivisionError('division by zero')
            new = {i: dct[i] / other_dct[i] for i in dct.keys() & other_dct.keys()}
        elif size == 1 and other_size: 
            if 0 in dct: 
                if len(other_dct) != other_size: raise ZeroDivisionError('division by zero')
                value = dct[0]
                new = {i: value / j for i, j in other_dct.items()}
            else:
                new = dct
            size = other_size
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                new = {i: j / other for i, j in dct.items()}
            elif dct:
                raise ZeroDivisionError('division by zero')
            else:
                new = {}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _truediv_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            new = {i: j / other[i] for i, j in dct.items()}
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i: value / float(other[i]) for i in range(other_size)}
            elif dct:
                raise ZeroDivisionError('division by zero')
            else:
                new = {}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseVector.from_dict(new, size)
    
    def _itruediv_scalar(self, other):
        dct = self.dct
        other = float(other)
        for i in dct: dct[i] /= other
        return self
    
    def _itruediv_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            for i in dct: 
                if i in other_dct:
                    dct[i] /= other_dct[i]
                else:
                    raise ZeroDivisionError('division by zero')
        elif size == 1 and other_size: 
            if 0 in dct: 
                if other_size != other_size: raise ZeroDivisionError('division by zero')
                value = dct.pop(0)
                for i, j in other_dct.items(): dct[i] = value / j
            self.size = other_size
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                for i in dct: dct[i] /= other
            elif dct:
                raise ZeroDivisionError('division by zero')
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _itruediv_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i in tuple(dct):
                j = other[i]
                dct[i] /= float(j)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in dct: 
                value = dct.pop(0)
                for i in range(other_size):
                    dct[i] = value / float(other[i])
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def __neg__(self):
        return SparseVector.from_dict({i: -j for i, j in self.dct.items()}, self.size)
    
    def __rtruediv__(self, other):
        if hasattr(other, '__len__'):
            return other / self.to_array()
        elif other:
            dct = self.dct.copy()
            size = self.size
            if len(dct) != size: raise ZeroDivisionError('division by zero')
            other = float(other)
            for i in dct: dct[i] = other / dct[i]
        else:
            size = self.size
            dct = {}
        return SparseVector.from_dict(dct, size)
    
    value = SparseArray.value
    
    def _eq_scalar(self, other):
        dct = self.dct
        size = self.size
        if other:
            new = {i for i in dct if dct[i] == other}
        else:
            new = {i for i in range(size) if i not in dct}
        return SparseLogicalVector.from_set(new, size)
    
    def _eq_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            new = {
                i for i in range(size) 
                if (i in other_dct and dct[i] == other_dct[i]
                    if i in dct
                    else i not in other_dct)
            }
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                new = {i for i in dct if dct[i] == other}
            else:
                new = {i for i in range(size) if i not in dct}
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i for i, j in other_dct.items() if j == value}
            else:
                new = {i for i in range(size) if i not in other_dct}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _eq_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            new = {
                i for i in range(size) 
                if (dct[i] == other[i] if i in dct else not other[i])
            }
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct: 
                value = dct[0]
                new = {i for i in range(size) if other[i] == value}
            else:
                new = {i for i in range(size) if not other[i]}
        elif other_size == 1:
            other = other[0]
            if other: 
                new = {i for i, j in dct.items() if other == j}
            else:
                new = {i for i in range(size) if i not in dct}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)

    def _ne_scalar(self, other):
        dct = self.dct
        size = self.size
        if other:
            new = {i for i in range(size) if i not in dct or dct[i] != other}
        else:
            new = {*dct}
        return SparseLogicalVector.from_set(new, size)
    
    def _ne_sparse(self, other):
        dct = self.dct
        size = self.size
        other_size = other.size
        other_dct = other.dct
        if size == other_size:
            try:
                new = {
                    i for i in [*dct, *other_dct]
                    if (i not in other_dct or i not in dct or dct[i] != other_dct[i])
                }
            except:
                breakpoint()
        elif other_size == 1:
            if 0 in other_dct: 
                other = other_dct[0]
                new = {i for i in range(size) if i not in dct or dct[i] != other}
            else:
                new = {*dct}
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct:
                value = dct[0]
                new = {i for i in range(size) if i not in other_dct or other_dct[i] != value}
            else:
                new = {*other_dct}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _ne_array(self, other):
        dct = self.dct
        size = self.size
        other_size = len(other)
        if size == other_size:
            new = {
                i for i in range(size) 
                if (dct[i] != other[i] if i in dct else other[i])
            }
        elif size == 1 and other_size: 
            size = other_size
            if 0 in dct:
                value = dct[0]
                new = {i for i in range(size) if other[i] != value}
            else:
                new = {i for i in range(size) if other[i]}
        elif other_size == 1:
            other = other[0]
            if other: 
                new = {i for i in range(size) if i not in dct or (other != dct[i])}
            else:
                new = {*dct}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    # Not yet optimized methods

    def __and__(self, other):
        return self.to_array() & other

    def __xor__(self, other):
        return self.to_array() ^ other

    def __or__(self, other): 
        return self.to_array() | other

    def __iand__(self, other):
        self[:] = self.to_array() & other
        return self

    def __ixor__(self, other):
        self[:] = self.to_array() ^ other
        return self

    def __ior__(self, other): 
        self[:] = self.to_array() | other
        return self
    
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
    __rmatmul__ = SparseArray.__rmatmul__
    __rfloordiv__ = SparseArray.__rfloordiv__
    __rmod__ = SparseArray.__rmod__
    __rpow__ = SparseArray.__rpow__
    __rlshift__ = SparseArray.__rlshift__
    __rrshift__ = SparseArray.__rrshift__
    __imatmul__ = SparseArray.__imatmul__
    __ifloordiv__ = SparseArray.__ifloordiv__
    __imod__ = SparseArray.__imod__ 
    __ipow__ = SparseArray.__ipow__
    __ilshift__ = SparseArray.__ilshift__
    __irshift__ = SparseArray.__irshift__
    
    # Representation
    
    __bool__ = SparseArray.__bool__
    __repr__ = SparseArray.__repr__
    __str__ = SparseArray.__str__

# %% For boolean sparse arrays
    
class SparseLogicalVector:
    __doc__ = sparse.__doc__
    __slots__ = ('set', 'size')
    ndim = 1
    dtype = bool
    
    def __init__(self, obj=None, size=None):
        if obj is None:
            self.set = set()
            if size is None: raise ValueError('must pass size if no object given')
            self.size = size
        elif isinstance(obj, set):
            self.set = obj
            self.size = size
            if size is None: raise ValueError('must pass size if object is a set')
        elif isinstance(obj, SparseLogicalVector):
            self.set = obj.set.copy()
            self.size = obj.size if size is None else size
        elif isinstance(obj, SparseVector):
            self.set = set(obj.dct)
            self.size = obj.size if size is None else size
        elif hasattr(obj, '__iter__'):
            self.set = set()
            for i, j in enumerate(obj):
                if j: self.set.add(i)
            self.size = len(obj) if size is None else size
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse array')
    
    @property
    def dct(self):
        return {i: 1. for i in self.set}
    
    def __abs__(self):
        return self.copy()
    
    def __iter__(self):
        set = self.set
        for i in range(self.size):
            yield i in set
    
    def __len__(self):
        return self.size
    
    def to_flat_array(self, arr=None):
        if arr is None:
            return self.to_array()
        else:
            arr[:] = False
            for i in self.set: arr[i] = True
            return arr
    
    def from_flat_array(self, arr=None):
        self[:] = arr
    
    @classmethod
    def from_size(cls, size):
        return cls.from_set(set(), size)
    
    @property
    def vector_size(self):
        return self.size
    
    @property
    def shape(self):
        return (self.size,)
    
    @classmethod
    def from_set(cls, set, size):
        new = cls.__new__(cls)
        new.set = set
        new.size = size
        return new
    
    def remove_negatives(self): pass
    
    def has_negatives(self):
        return False
    
    def negative_keys(self): ()
    
    def negative_index(self): [],
    
    def nonzero_index(self):
        return [*self.set],
    nonzero = positive_index = nonzero_index
    
    def nonzero_keys(self):
        return self.set
    
    def nonzero_values(self):
        for i in self.set: yield True
    
    def nonzero_items(self):
        for i in self.set: yield (i, True)
    
    def sparse_equal(self, other):
        other = sparse_vector(other)
        return self.set == other.set
    
    def sum_of(self, index):
        set = self.set
        if hasattr(index, '__iter__'):
            return sum([1 for i in index if i in set])
        else:
            return index in set
    
    def any(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = bool(self.set)
        if keepdims: arr = SparseLogicalVector({0} if arr else set(), size=1)
        return arr
    
    def all(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = len(self.set) == self.size
        if keepdims: arr = SparseLogicalVector({0} if arr else set(), size=1)
        return arr
    
    def sum(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = len(self.set)
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
        return arr
    
    def mean(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        arr = len(self.set) / self.size if self.set else 0.
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
        return arr
    
    def max(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        set = self.set
        if set:
            arr = 1.
        elif self.size:
            arr = 0.
        else:
            raise ValueError('zero-size array reduction has no identity')
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
        return arr
    
    def min(self, axis=None, keepdims=False):
        if axis: raise ValueError('axis is out of bounds for 1-d sparse array')
        set = self.set
        if set:
            arr = len(set) >= self.size
        elif self.size:
            arr = 0.
        else:
            raise ValueError('zero-size array reduction has no identity')
        if keepdims: arr = SparseVector.from_dict({0: arr} if arr else {}, size=1)
        return arr
    
    def to_array(self, dtype=None):
        arr = np.zeros(self.size, dtype=dtype or bool)
        for i in self.set: arr[i] = True
        return arr
    astype = to_array
    
    def copy(self):
        return SparseLogicalVector.from_set(self.set.copy(), self.size)
    
    def __getitem__(self, index):
        set = self.set
        if index.__class__ is tuple:
            index, = unpack_index(index, self.ndim)
        ndim, has_bool = get_array_properties(index)
        if has_bool:
            return self[index.nonzero() if hasattr(index, 'nonzero') else np.nonzero(index)]
        if ndim == 1:
            arr = np.zeros(len(index), dtype=bool)
            for n, i in enumerate(index):
                if i in set: arr[n] = True
            return arr
        elif index.__class__ is slice:
            if index == open_slice:
                return self
            else:
                value = np.array([True if i in set else False for i in default_range(index, self.size)])
                value.setflags(0)
                return value
        elif ndim:
            raise IndexError(f'index can be at most 1-d, not {ndim}-d')    
        else:
            return True if index in set else False

    def __setitem__(self, index, value):
        set = self.set
        if index.__class__ is tuple:
            index, = unpack_index(index, self.ndim)
        ndim, has_bool = get_array_properties(index)
        value, vd, _ = reduce_ndim(value)
        if has_bool: 
            if ndim != 1:
                raise IndexError(
                    f'boolean index is {ndim}-d but sparse array is 1-d '
                )
            index, = index.nonzero() if hasattr(index, 'nonzero') else np.nonzero(index)
        if ndim == 1:
            if vd == 1:
                for i, j in zip(index, value): 
                    if j: set.add(i)
                    else: set.discard(i)
            elif vd > 1:
                raise IndexError(
                    f'cannot broadcast {vd}-d array on to 1-d sparse array'
                )
            elif value:
                for i in index: set.add(i)
            else:
                for i in index:
                    set.discard(i)
        elif index.__class__ is slice:
            if index == open_slice:
                if value is self: return
                set.clear()
                if vd == 0:
                    if value:
                        for i in range(self.size): set.add(i)
                elif value.__class__ in SparseVectorSet:
                    set.update(value.set)
                elif vd == 1:
                    for i, j in enumerate(value):
                        if j: set.add(i)
                        else: set.discard(i)
                
                else:
                    raise IndexError(
                        f'cannot broadcast {vd}-d array on to 1-d sparse array'
                    )
            else:
                self[default_range(index, self.size)] = value
        elif ndim:
            raise IndexError(f'index can be at most 1-d, not {ndim}-d')    
        elif vd:
            raise IndexError(
                'cannot set an array element with a sequence'
            )
        elif value:
            set.add(index)
        else:
            set.discard(index)
    
    exec(sparse_logical_vector_math_pseudo_optimized.format(name='add'))
    exec(sparse_logical_vector_math_pseudo_optimized.format(name='mul'))
    exec(sparse_logical_vector_math_pseudo_optimized.format(name='truediv'))
    exec(sparse_logical_vector_math_pseudo_optimized.format(name='and'))
    exec(sparse_logical_vector_math_pseudo_optimized.format(name='xor'))
    exec(sparse_logical_vector_math_pseudo_optimized.format(name='or'))
    exec(sparse_vector_imath.format(name='add', read_only=not_read_only))
    exec(sparse_vector_imath.format(name='mul', read_only=not_read_only))
    exec(sparse_vector_imath.format(name='truediv', read_only=not_read_only))
    exec(sparse_vector_imath.format(name='and', read_only=not_read_only))
    exec(sparse_vector_imath.format(name='xor', read_only=not_read_only))
    exec(sparse_vector_imath.format(name='or', read_only=not_read_only))
    __eq__ = SparseVector.__eq__
    __ne__ = SparseVector.__ne__
    __gt__ = SparseVector.__gt__
    __lt__ = SparseVector.__lt__
    __ge__ = SparseVector.__ge__
    __le__ = SparseVector.__le__
    
    __float__ = SparseArray.__float__
    __bool__ = SparseArray.__bool__
    __int__ = SparseArray.__int__
    __radd__ = SparseArray.__radd__
    __rsub__ = SparseArray.__rsub__
    __rmul__ = SparseArray.__rmul__
    __rand__ = SparseArray.__rand__
    __rxor__ = SparseArray.__rxor__
    __ror__ = SparseArray.__ror__
    
    def _iadd_scalar(self, other):
        set = self.set
        size = self.size
        if other:
            for i in range(size):
                if i in set: 
                    j = True + other
                    if not j: set.discard(i)
                else:
                    set.add(i)
        return self
    
    def _iadd_sparse(self, other):
        set = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            set.update(other.set)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in set: set.update(range(1, other_size))
            else: self.set.update(other.set)
        elif other_size == 1:
            if 0 in other.set: self.set.update(range(size))
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _iadd_array(self, other):
        set = self.set
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i, j in enumerate(other):
                if not j: continue
                if i in set:
                    j = True + j
                    if not j: set.discard(i)
                else:
                    set.add(i)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in set: set.update(range(1, other_size))
            else: set.update([i for i, j in enumerate(other) if j])
        else:
            raise ValueError('shape mismatch between arrays')
        return self
            
    def __isub__(self, other):
        raise TypeError(
            "boolean subtract, the `-` operator, is not supported, use the "
            "bitwise_xor, the `^` operator instead"
        )
    _isub_scalar = _isub_sparse = _isub_array = __isub__
    def __sub__(self, other):
        return SparseVector.from_dict({i: 1. for i in self.set}, self.size) - other
    _sub_scalar = _sub_sparse = _sub_array = __sub__
    
    def _imul_scalar(self, other):
        set = self.set
        if not other: set.clear()
        return self
    
    def _imul_sparse(self, other):
        set = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            set.intersection_update(other.set)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in set: 
                set.clear()
                set.update(other.set)
        elif other_size == 1:
            if not other.set: set.clear()
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _imul_array(self, other):
        set = self.set
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i in tuple(set):
                j = other[i]
                if not j: set.remove(i)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in set: 
                set.clear()
                set.update([i for i, j in enumerate(other) if j])
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _itruediv_scalar(self, other):
        if not other and set: raise ZeroDivisionError('division by zero')
        return self
    
    def _itruediv_sparse(self, other):
        set = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            if set.difference(other.set): raise ZeroDivisionError('division by zero')
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in set: set.update(range(1, other_size))
            elif set.difference(other.set): raise ZeroDivisionError('division by zero')
        elif other_size == 1:
            if not other.set and set: raise ZeroDivisionError('division by zero')
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _itruediv_array(self, other):
        set = self.set
        size = self.size
        other_size = len(other)
        if size == other_size:
            if set.difference([i for i, j in enumerate(other) if j]): raise ZeroDivisionError('division by zero')
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in set: 
                set.update(range(1, other_size))
                if set.difference([i for i, j in enumerate(other) if j]): raise ZeroDivisionError('division by zero')
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def __neg__(self):
        return SparseVector.from_dict({i: -1. for i in self.set}, self.size)
    
    def __invert__(self):
        set = self.set
        return SparseLogicalVector.from_set({i for i in range(self.size) if i not in set}, self.size)
    
    def __rtruediv__(self, other):
        if hasattr(other, '__len__'):
            return other / self.to_array()
        elif self.dtype is bool and other.__class__ is not bool:
            return SparseVector(other / self.to_array())
        elif other:
            set_ = self.set.copy()
            if len(set_) != self.size: raise ZeroDivisionError('division by zero')
        else:
            set_ = set()
        return SparseLogicalVector.from_set(set_, self.size)
    
    value = SparseArray.value
    
    def _iand_scalar(self, other):
        data = self.set
        if not other: data.clear()
        return self
    
    def _iand_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            data.intersection_update(other.set)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in data: 
                data.clear()
                data.update(other.set)
        elif other_size == 1:
            if 0 not in other.set: data.clear()
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _iand_array(self, other):
        data = self.set
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i in tuple(data):
                j = other[i]
                if not j: data.discard(i)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in data: 
                data.clear()
                data.update([i for i, j in enumerate(other) if j])
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _ixor_scalar(self, other):
        data = self.set
        if other: data.symmetric_difference_update(range(self.size))
        return self
    
    def _ixor_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            data.symmetric_difference_update(other.set)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in data: 
                other_data = other.set
                data.clear()
                data.update([i for i in range(other_size) if i not in other_data])
            else:
                data.update(other.set)
        elif other_size == 1:
            if 0 in other.set: 
                data.symmetric_difference_update(range(size))
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _ixor_array(self, other):
        data = self.set
        size = self.size
        other_size = len(other)
        if size == other_size:
            for i, j in enumerate(other):
                if not j: continue
                if i in data: data.remove(i)
                else: data.add(i)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in data: 
                data.clear()
                data.update([i for i, j in enumerate(other) if not j])
            else:
                data.update([i for i, j in enumerate(other) if j])
        else:
            raise ValueError('shape mismatch between arrays')
        return self

    def _ior_scalar(self, other):
        data = self.set
        if other: data.update(range(self.size))
        return self
    
    def _ior_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            data.update(other.set)
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in data: data.update(range(1, other_size))
            else: data.update(other.set)
        elif other_size == 1:
            if 0 in other.set: data.update(range(size))
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _ior_array(self, other):
        data = self.set
        size = self.size
        other_size = len(other)
        if size == other_size:
            data.update([i for i, j in enumerate(other) if j])
        elif size == 1 and other_size: 
            self.size = other_size
            if 0 in data: data.update(range(1, other_size))
            else: data.update([i for i, j in enumerate(other) if j])
        else:
            raise ValueError('shape mismatch between arrays')
        return self
    
    def _eq_scalar(self, other):
        data = self.set
        size = self.size
        new = {i for i in range(size) if (i in data) == other}
        return SparseLogicalVector.from_set(new, size)
    
    def _eq_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            new = {*range(size)}
            new.difference_update(data.symmetric_difference(other.set))
        elif other_size == 1:
            if 0 in other.set: 
                new = data.copy()
            else:
                new = {*range(size)}
                new.difference_update(data)
        elif size == 1 and other_size: 
            size = other_size
            other = other.set
            if 0 in data:
                new = other.copy()
            else:
                new = {i for i in range(size) if i not in other}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _eq_array(self, other):
        new = self.to_array() == other
        if new.__class__ in bools: raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set({i for i, j in enumerate(new) if j}, new.size)

    def _ne_scalar(self, other):
        data = self.set
        size = self.size
        new = {i for i in range(size) if (i in data) != other}
        return SparseLogicalVector.from_set(new, size)
    
    def _ne_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            new = data.symmetric_difference(other.set)
        elif other_size == 1:
            if 0 in other.set: 
                new = {*range(size)}
                new.difference_update(data)
            else:
                new = data.copy()
        elif size == 1 and other_size: 
            size = other_size
            other = other.set
            if 0 in data:
                new = {i for i in range(size) if i not in other}
            else:
                new = other.copy()
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _ne_array(self, other):
        new = self.to_array() != other
        if new.__class__ in bools: raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set({i for i, j in enumerate(new) if j}, new.size)

    def _gt_scalar(self, other):
        data = self.set
        size = self.size
        new = {i for i in range(size) if (i in data) > other}
        return SparseLogicalVector.from_set(new, size)
    
    def _gt_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            new = data.difference(other.set)
        elif other_size == 1:
            if 0 in other.set:
                new = set()
            else:
                new = data.copy()
        elif size == 1 and other_size: 
            size = other_size
            if 0 in data:
                other = other.set
                new = {i for i in range(size) if i not in other}
            else:
                new = set()
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _gt_array(self, other):
        new = self.to_array() > other
        return SparseLogicalVector.from_set({i for i, j in enumerate(new) if j}, new.size)
    
    def _lt_scalar(self, other):
        data = self.set
        size = self.size
        new = {i for i in range(size) if (i in data) < other}
        return SparseLogicalVector.from_set(new, size)
    
    def _lt_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            new = other.set.difference(data)
        elif other_size == 1:
            if 0 in other.set:
                new = {*range(size)}
                new.difference_update(data)
            else:
                new = set()
        elif size == 1 and other_size: 
            size = other_size
            if 0 in data:
                new = set()
            else:
                other = other.set
                new = other.copy()
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _lt_array(self, other):
        new = self.to_array() < other
        return SparseLogicalVector.from_set({i for i, j in enumerate(new) if j}, new.size)

    def _ge_scalar(self, other):
        data = self.set
        size = self.size
        new = {i for i in range(size) if (i in data) >= other}
        return SparseLogicalVector.from_set(new, size)
    
    def _ge_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            new = {*range(size)}
            new.difference_update(other.set.difference(data))
        elif other_size == 1:
            if 0 in other.set:
                new = data.copy()
            else:
                new = {*range(size)}
        elif size == 1 and other_size: 
            size = other_size
            if 0 in data:
                new = {*range(size)}
            else:
                other = other.set
                new = {i for i in range(size) if i not in other}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _ge_array(self, other):
        new = self.to_array() >= other
        return SparseLogicalVector.from_set({i for i, j in enumerate(new) if j}, new.size)

    def _le_scalar(self, other):
        data = self.set
        size = self.size
        new = {i for i in range(size) if (i in data) <= other}
        return SparseLogicalVector.from_set(new, size)
    
    def _le_sparse(self, other):
        data = self.set
        size = self.size
        other_size = other.size
        if size == other_size:
            new = {*range(size)}
            new.difference_update(data.difference(other.set))
        elif other_size == 1:
            if 0 in other.set:
                new = {*range(size)}
            else:
                new = {*range(size)}
                new.difference_update(data)
        elif size == 1 and other_size: 
            size = other_size
            if 0 in data:
                other = other.set
                new = {i for i in range(size) if i in other}
            else:
                new = {*range(size)}
        else:
            raise ValueError('shape mismatch between arrays')
        return SparseLogicalVector.from_set(new, size)
    
    def _le_array(self, other):
        new = self.to_array() <= other
        return SparseLogicalVector.from_set({i for i, j in enumerate(new) if j}, new.size)
    
    # Not yet optimized methods
    
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
    __rmatmul__ = SparseArray.__rmatmul__
    __rfloordiv__ = SparseArray.__rfloordiv__
    __rmod__ = SparseArray.__rmod__
    __rpow__ = SparseArray.__rpow__
    __rlshift__ = SparseArray.__rlshift__
    __rrshift__ = SparseArray.__rrshift__
    __imatmul__ = SparseArray.__imatmul__
    __ifloordiv__ = SparseArray.__ifloordiv__
    __imod__ = SparseArray.__imod__ 
    __ipow__ = SparseArray.__ipow__
    __ilshift__ = SparseArray.__ilshift__
    __irshift__ = SparseArray.__irshift__
    
    # Representation
    
    __bool__ = SparseArray.__bool__
    __repr__ = SparseArray.__repr__
    __str__ = SparseArray.__str__

SparseSet = frozenset([SparseArray, SparseVector, SparseLogicalVector])
SparseVectorSet = frozenset([SparseVector, SparseLogicalVector])