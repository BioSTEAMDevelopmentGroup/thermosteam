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

def unpack_index(index, ndim):
    if (indim:=index.__len__()) == ndim:
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
        raise ValueError(f'cannot convert {ndim}-D object to a sparse array or vector')
    
def get_ndim(value, ndim=0):
    if hasattr(value, 'ndim'):
        ndim += value.ndim
    elif hasattr(value, '__iter__'):
        ndim += 1
        for i in value: return get_ndim(i, ndim)
    return ndim

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
    
    In contrast to Scipy's sparse 2-D arrays, sparse arrays do not have a strict row length 
    (but still have a strict column length). 
    
    """
    __slots__ = ('rows', '_base')
    ndim = 2
    
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
                
    def __float__(self):
        return self.to_array(self.vector_size, dtype=float)
    
    def __int__(self):
        return self.to_array(self.vector_size, dtype=int)
    
    def nonzero_index(self):
        return [(i, j) for i, row in enumerate(self.rows) for j in row.dct]
    
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
    
    def remove_negatives(self):
        for i in self.rows: i.remove_negatives
    
    def shares_data_with(self, other):
        return bool(self.base.intersection(other.base))
    
    def flat_array(self, arr=None, vector_size=None, dtype=None):
        rows = self.rows
        N = len(rows)
        if vector_size is None: vector_size = self.vector_size
        if dtype is None: dtype = self.dtype
        if arr is None:
            arr = np.zeros(N * vector_size, dtype=dtype)
            for i in range(N):
                row = rows[i]
                for j, value in row.dct.items(): 
                    arr[i * vector_size + j] = value
            return arr
        else:
            dcts = [i.dct for i in rows]
            for i in range(N): dcts[i].clear()
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
    
    def any(self, dim=None):
        rows = self.rows
        if dim is None:
            return any([i.any() for i in rows])
        elif dim == 0:
            keys = set()
            for i in rows: keys.update(i)
            dct = {i: True for i in keys}
            return SparseVector.from_dict(dct, self.size)
        elif dim == 1:
            return np.array([i.any() for i in rows])
    
    def sparse_equal(self, other):
        return all([i.dct == j.dct for i, j in zip(self.rows, sparse_array(other).rows)])
    
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
                    raise IndexError(f'row index can be at most 1-D, not {md}-D')
            elif md == 0: 
                return rows[m][n]
            elif md == 1: 
                if nd == 0:
                    n = n.__index__()
                    return np.array([rows[i].dct[n] for i in m])
                elif nd == 1:
                    return np.array([rows[i].dct[j.__int__()] for i, j in zip(m, n)])
                else:
                    raise IndexError(f'column index can be at most 1-D, not {nd}-D')
            else:
                raise IndexError(f'row index can be at most 1-D, not {md}-D')
        elif (ndim:=get_ndim(index)) == 1:
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
                                f'cannot broadcast {vd}-D array on to array column'
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
                            f'cannot broadcast {vd}-D array on to 2-D sparse array'
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
                            f'cannot broadcast {ndim}-D array on to 1-D '
                        )
                else:
                    raise IndexError(f'column index can be at most 1-D, not {md}-D')
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
                        f'cannot broadcast {vd}-D array on to 2-D sparse array'
                    )
            else:
                raise IndexError(f'row index can be at most 1-D, not {md}-D')
        elif (ndim:=get_ndim(index)) == 1:
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
                    f'cannot broadcast {vd}-D array on to 2-D sparse array'
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
    def _minimum_vector_size(self):
        rows = self.rows
        return max([i._minimum_vector_size for i in rows]) if rows else 0
    
    @property
    def dtype(self):
        for i in self.rows: return i.dtype
    
    @property
    def shape(self):
        return (len(self.rows), self.vector_size)
    
    @property
    def size(self):
        return len(self.rows) * self.vector_size
    
    def to_array(self, vector_size=None, dtype=None):
        rows = self.rows
        N = len(rows)
        if vector_size is None: vector_size = self.vector_size
        if dtype is None: dtype = self.dtype
        arr = np.zeros([N, vector_size], dtype=dtype)
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
        
    def sum(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            value = sum([sum(i.dct.values()) for i in rows])
            if keepdims: value = SparseArray([[value]], 1)
            return value
        elif axis == 0:
            value = SparseVector(sum_sparse_vectors(rows), size=self.vector_size)
            if keepdims: value = SparseArray([value])
            return value
        elif axis == 1:
            if keepdims:
                arr = np.zeros([len(rows), 1])
            else:
                arr = np.zeros(len(rows))
            for i, j in enumerate(rows): arr[i] = sum(j.dct.values())
            return arr
        else:
            raise ValueError('axis must be either 0, 1, or None')
    
    def mean(self, axis=None, keepdims=False):
        rows = self.rows
        if axis is None:
            value = sum([sum(i.dct.values()) for i in rows]) / sum([i.size for i in self.rows])
            if keepdims: value = SparseArray([[value]], 1)
            return value
        elif axis == 0:
            value = SparseVector(sum_sparse_vectors(rows), size=self.vector_size)
            value /= len(rows)
            if keepdims: value = SparseArray([value])
            return value
        elif axis == 1:
            if keepdims:
                arr = np.zeros([len(rows)])
            else:
                arr = np.zeros(len(rows))
            for i, j in enumerate(rows): arr[i] = sum(j.dct.values()) / j.size
            return arr
        else:
            raise ValueError('axis must be either 0, 1, or None')
    
    def max(self, axis=None):
        rows = self.rows
        if axis is None:
            return max(max([max(i.dct.values()) for i in rows]), 0.)
        elif axis == 0:
            keys = set()
            for i in rows: keys.update(i)
            dcts = [i.dct for i in rows]
            return SparseVector.from_dict({i: max([j[i] for j in dcts if i in j]) for i in keys})
        elif axis == 1:
            arr = np.zeros(len(rows))
            for i, j in enumerate(rows): arr[i] = j.max()
            return arr
        else:
            raise ValueError('axis must be either 0, 1, or None')
    
    def min(self, axis=None):
        rows = self.rows
        if axis is None:
            return min(min([min(i.dct.values()) for i in rows]), 0.)
        elif axis == 0:
            keys = set()
            for i in rows: keys.update(i)
            dcts = [i.dct for i in rows]
            return SparseVector.from_dict({i: min([j[i] for j in dcts if i in j]) for i in keys})
        elif axis == 1:
            arr = np.zeros(len(rows))
            for i, j in enumerate(rows): arr[i] = j.min()
            return arr
        else:
            raise ValueError('axis must be either 0, 1, or None')
            
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
    
    def __repr__(self):
        name = type(self).__name__
        if self.any():
            n_spaces = len(name) - 5
            nums = repr(self.to_array(vector_size=self._minimum_vector_size))[5:].replace('\n', '\n' + n_spaces * ' ')
            return f'{name}{nums}'
        elif self.ndim == 2:
            return f'{name}({[[]*len(self.rows)]})'
        elif self.ndim == 1:
            return f'{name}([])'
    
    def __str__(self):
        if self.any():
            arr = self.to_array(vector_size=self._minimum_vector_size)
            return str(arr)
        elif self.ndim == 2:
            return str([[]*len(self.rows)])
        elif self.ndim == 1:
            return '[]'
            

class SparseVector:
    """
    Create a SparseVector object that can be used for array-like arithmetic operations
    (i.e., +, -, *, /) of sparse 1-dimensional arrays. 
    
    In contrast to Scipy's sparse 2-D arrays, sparse vectors can only represent 1-D arrays
    and do not have a strict size. 
    
    """
    __slots__ = ('dct', 'read_only', 'size', '_base')
    ndim = 1
    
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
    
    __eq__ = SparseArray.__eq__
    __ne__ = SparseArray.__ne__
    __gt__ = SparseArray.__gt__
    __lt__ = SparseArray.__lt__
    __ge__ = SparseArray.__ge__
    __le__ = SparseArray.__le__
    
    def __abs__(self):
        positive = abs
        dct = self.dct.copy()
        for i in dct: dct[i] = positive(dct[i])
        return SparseVector.from_dict(dct, self.size)
    
    def __iter__(self):
        dct = self.dct
        for i in range(self.size):
            yield dct[i] if i in dct else 0.
    
    def __len__(self):
        return self.size
    
    def __float__(self):
        return self.to_array(self.size, dtype=float)
    
    def __int__(self):
        return self.to_array(self.size, dtype=int)
    
    def flat_array(self, arr=None, vector_size=None):
        if arr is None:
            return self.to_array(vector_size)
        else:
            self[:] = arr
    
    @classmethod
    def from_size(cls, size):
        return cls.from_dict({}, size)
    
    @property
    def vector_size(self):
        return self.size
    
    @property
    def _minimum_vector_size(self):
        dct = self.dct
        if dct:
            return max(self.dct) + 1
        else:
            return 0
    
    @property
    def dtype(self):
        for i in self.dct.values(): return type(i)
    
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
    
    def sparse_equal(self, other):
        other = sparse_vector(other)
        return self.dct == other.dct
    
    def any(self):
        return bool(self.dct)
    
    def argmax(self):
        max = 0
        argmax = None
        for i, j in self.dct.items():
            if j > max: argmax = i
        return argmax
    
    def sum_of(self, index):
        dct = self.dct
        if hasattr(index, '__iter__'):
            return sum([dct[i] for i in index if i in dct])
        elif index in dct:
            return dct[index]
        else:
            return 0.
    
    def remove_negatives(self):
        dct = self.dct
        for i in tuple(dct): 
            if dct[i] < 0.: del dct[i]
    
    def negative_index(self):
        dct = self.dct
        return [i for i in dct if dct[i] < 0.],
    
    def positive_index(self):
        dct = self.dct
        return [i for i in dct if dct[i] > 0.],
    
    def nonzero_index(self):
        return [*self.dct.keys()]
    
    def nonzero_keys(self):
        return self.dct.keys()
    
    def nonzero_values(self):
        return self.dct.values()
    
    def nonzero_items(self):
        return self.dct.items()
    
    def sum(self):
        return sum(self.dct.values())
    
    def mean(self):
        return sum(self.dct.values()) / self.size
    
    def max(self):
        return max(max(self.dct.values()), 0.)
    
    def min(self):
        return min(min(self.dct.values()), 0.)
    
    def to_array(self, vector_size=None, dtype=None):
        if vector_size is None: vector_size = self.size
        if dtype is None: dtype = self.dtype
        arr = np.zeros(vector_size, dtype=dtype)
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
        if hasattr(index, '__iter__'):
            if index.__class__ is tuple:
                index, = unpack_index(index, self.ndim)
                if not hasattr(index, '__iter__'):
                    index = index.__index__()
                    return dct[index] if index in dct else 0.
            arr = np.zeros(len(index))
            for n, i in enumerate(index):
                i = i.__index__()
                if i in dct: arr[n] = dct[i]
            return arr
        elif index == open_slice:
            return self
        else:
            index = index.__index__()
            if index in dct:
                return dct[index]
            else:
                return 0.

    def __setitem__(self, index, value):
        if self.read_only: raise ValueError('assignment destination is read-only')
        dct = self.dct
        if hasattr(index, '__iter__'):
            if index.__class__ is tuple:
                index, = unpack_index(index, self.ndim)
                if not hasattr(index, '__iter__'):
                    if hasattr(value, '__iter__'):
                        raise IndexError(
                            'cannot set an array element with a sequence'
                        )
                    index = index.__index__()
                    if value:
                        dct[index] = value
                    elif index in dct:
                        del dct[index]
                    return
            if (vd:=get_ndim(value)) == 1:
                for i, j in zip(index, value): 
                    i = i.__index__()
                    if j: dct[i] = j
                    elif i in dct: del dct[i]
            elif vd > 1:
                raise IndexError(
                    f'cannot broadcast {vd}-D array on to 1-D sparse vector'
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
    
    shares_data_with = SparseArray.shares_data_with
    __repr__ = SparseArray.__repr__
    __str__ = SparseArray.__str__