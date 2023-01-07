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
)

open_slice = slice(None)

def sparse_vector(arr, copy=False):
    """
    Convert 1-d array to a SparseVector object.

    """
    if isinstance(arr, SparseVector):
        return arr.copy() if copy else arr
    else:
        return SparseVector(arr)

def nonzero_items(arr):
    if isinstance(arr, SparseVector):
        return arr.dct.items()
    else:
        return [(i, j) for i, j in enumerate(arr) if j]

def sparse_array(arr, copy=False):
    """
    Convert 2-d array to a SparseArray object.

    """
    if isinstance(arr, SparseArray):
        return arr.copy() if copy else arr
    else:
        return SparseArray(arr)

def get_ndim(value, ndim=0):
    if hasattr(value, 'ndim'):
        ndim += value.ndim
    elif hasattr(value, '__iter__'):
        ndim += 1
        for i in value: return get_ndim(i, ndim)
    return ndim

class SparseArray:
    """
    Create a SparseArray object that can be used for array-like arithmetic operations
    (i.e., +, -, *, /) of sparse 2-dimensional arrays. 
    
    In contrast to Scipy's sparse 2-D arrays, sparse arrays do not have a defined row length 
    (but still have a defined column length). 
    
    """
    __slots__ = ('rows', '_base')
    ndim = 2
    
    def __init__(self, obj=None):
        if obj is None:
            self.rows = []
        elif hasattr(obj, '__iter__'):
            self.rows = [sparse_vector(row) for row in obj]
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse array')
    
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
    
    def shares_data_with(self, other):
        return self.base.intersection(other.base)
    
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
            return SparseVector.from_dict(dct, dtype=bool)
        elif dim == 1:
            return np.array([i.any() for i in rows])
    
    def __eq__(self, other):
        return all([i.dct == j.dct for i, j in zip(self.rows, sparse_array(other).rows)])
    
    def __getitem__(self, index):
        rows = self.rows
        if index.__class__ is tuple:
            i, j = index
            return rows[i][j]
        elif hasattr(index, '__iter__'):
            return SparseArray.from_rows([rows[i] for i in index])
        elif index == open_slice:
            return self
        else:
            return rows[index]
        
    def __setitem__(self, index, value):
        rows = self.rows
        if index.__class__ is tuple:
            m, n = index
            if m == open_slice:
                if n == open_slice:
                    self[:] = value
                else:
                    ndim = get_ndim(value)
                    if ndim == 0.:
                        for i in rows: i[n] = value
                    elif ndim == 1:
                        for i in rows: i[n] = value
                    elif ndim == 2:
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
                            f'cannot broadcast {ndim}-D array with onto 2-D sparse array'
                        )
            elif n == open_slice:
                md = get_ndim(m)
                if md == 0:
                    rows[m][:] = value
                elif md == 1:
                    ndim = get_ndim(value)
                    if ndim == 0:
                        if value != 0:
                            raise IndexError(
                                'cannot broadcast nonzero value onto sparse array; '
                                'sparse arrays do not have a defined shape'
                            )
                        else:
                            for i in m: rows[i].clear()
                    elif ndim == 1:
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
                            f'cannot broadcast {ndim}-D array with onto 1-D '
                        )
                else:
                    raise IndexError(f'column index can be at most 1-D, not {md}-D')
            else: 
                rows[m][n] = value
        elif hasattr(index, '__iter__'):
            SparseArray.from_rows([rows[i] for i in index])[:] = value
        elif index == open_slice:
            for i in rows: i.dct.clear()
            ndim = get_ndim(value)
            if ndim == 0.:
                if value != 0.:
                    raise IndexError(
                        'cannot broadcast nonzero value onto sparse array; '
                        'sparse arrays do not have a defined shape'
                    )
            elif ndim == 1:
                for i in rows: i[:] = value
            elif ndim == 2:
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
                    f'cannot broadcast {ndim}-D array with onto 2-D sparse array'
                )
        else:
            rows[index][:] = value
        
    def minimum_length(self):
        return max([i.minimum_length() for i in self.rows])
        
    def to_array(self, length):
        rows = self.rows
        N = len(rows)
        for i in rows:
            dtype = i.dtype
        else:
            dtype = None
        arr = np.zeros([N, length], dtype=dtype)
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
        
    def sum(self, axis=None):
        rows = self.rows
        if axis is None:
            return sum([sum(i.dct.values()) for i in rows])
        elif axis == 0:
            arr = np.zeros(len(rows))
            for i, j in enumerate(rows): arr[i] = sum(j.dct.values())
            return arr
        elif axis == 1:
            return sum(rows)
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
        if hasattr(value, '__iter__'):
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i += j
            elif value_length == 1:
                value = value[0]
                for i in rows: i += value
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        elif isinstance(value, SparseVector):
            for i in rows: i += value
        elif value != 0.:
            raise IndexError(
                'cannot broadcast nonzero value onto sparse array; '
                'sparse arrays do not have a defined shape'
            )
        return self
    
    def __add__(self, value):
        new = self.copy()
        new += value
        return new
            
    def __isub__(self, value):
        rows = self.rows
        if hasattr(value, '__iter__'):
            value_length = len(value)
            self_length = len(rows)
            if self_length == value_length:
                for i, j in zip(rows, value): i -= j
            else:
                raise IndexError(
                    f'cannot broadcast input array with length {value_length} to length {self_length}'
                )
        elif isinstance(value, SparseVector):
            for i in rows: i -= value
        elif value != 0.:
            raise IndexError(
                'cannot broadcast nonzero value onto sparse array; '
                'sparse arrays do not have a defined shape'
            )
        return self
    
    def __sub__(self, value):
        new = self.copy()
        new -= value
        return new
    
    def __imul__(self, value):
        rows = self.rows
        if hasattr(value, '__iter__'):
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
        if hasattr(value, '__iter__'):
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
        if hasattr(value, '__iter__'):
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
            nums = repr(self.to_array(length=self.minimum_length()))[5:].replace('\n', '\n' + n_spaces * ' ')
            return f'{name}{nums}'
        elif self.ndim == 2:
            return f'{name}({[[]*len(self.rows)]})'
        elif self.ndim == 1:
            return f'{name}([])'
    
    def __str__(self):
        if self.any():
            arr = self.to_array(length=self.minimum_length())
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
    and do not have a defined size. 
    
    """
    __slots__ = ('dct', 'dtype', '_base')
    ndim = 1
    
    def __init__(self, obj=None, dtype=float):
        self.dtype = dtype
        if obj is None:
            self.dct = {}
        elif isinstance(obj, dict):
            self.dct = obj
        elif isinstance(obj, SparseVector):
            self.dct = obj.dct.copy()
        elif hasattr(obj, '__iter__'):
            self.dct = dct = {}
            for i, j in enumerate(obj):
                if j: dct[int(i)] = self.dtype(j)
        else:
            raise TypeError(f'cannot convert {type(obj).__name__} object to a sparse vector')
    
    @property
    def base(self):
        try:
            base = self._base
        except:
            self._base = base = frozenset([id(self.dct)])
        return base
    
    @classmethod
    def from_dict(cls, dct, dtype=float):
        new = cls.__new__(cls)
        new.dct = dct
        new.dtype = dtype
        return new
    
    def minimum_length(self):
        dct = self.dct
        if dct:
            return max(self.dct) + 1
        else:
            return 0
    
    def __eq__(self, other):
        other = sparse_vector(other)
        return self.dct == other.dct
    
    def any(self):
        return bool(self.dct)
    
    def sum_of(self, index):
        if hasattr(index, '__iter__'):
            dct = self.dct
            return sum([dct[i] for i in index if i in dct])
        elif (index:=int(index)) in dct:
            return dct[index]
    
    def nonzero_index(self):
        return [*self.dct.keys()]
    
    def nonzero_values(self):
        return [*self.dct.values()]
    
    def nonzero_items(self):
        return [*self.dct.items()]
    
    def sum(self):
        return sum(self.dct.values())
    
    def max(self):
        return max(max(self.dct.values()), 0.)
    
    def min(self):
        return min(min(self.dct.values()), 0.)
    
    def to_array(self, length):
        arr = np.zeros(length, dtype=self.dtype)
        for i, j in self.dct.items(): arr[i] = j
        return arr
    
    def copy(self):
        new = object.__new__(SparseVector)
        new.dct = self.dct.copy()
        new.dtype = self.dtype
        return new
    
    def copy_like(self, other):
        dct = self.dct
        dct.clear()
        dct.update(other.dct)
    
    def clear(self):
        self.dct.clear()
    
    def __getitem__(self, index):
        dct = self.dct
        if hasattr(index, '__iter__'):
            dct = self.dct
            arr = np.zeros(len(index))
            for n, i in enumerate(index):
                if (i:=int(i)) in dct: arr[n] = dct[i]
            return arr
        elif index == open_slice:
            return self
        elif (index:=int(index)) in dct:
            return dct[index]
        else:
            return 0.

    def __setitem__(self, index, value):
        if hasattr(index, '__iter__'):
            dct = self.dct
            if hasattr(value, '__iter__'):
                for i, j in zip(index, value): 
                    if j: dct[int(i)] = self.dtype(j)
                    elif (i:=int(i)) in dct: del dct[i]
            else:
                if value:=self.dtype(value):
                    for i in index: dct[int(i)] = value
                else:
                    for i in index: del dct[i]
        elif index == open_slice:
            dct = self.dct
            dct.clear()
            if hasattr(value, '__iter__'):
                for i, j in enumerate(value): 
                    if j: dct[i] = self.dtype(j)
                    elif i in dct: del dct[i]
            elif isinstance(value, SparseVector):
                dct.update(value.dct)
            elif value != 0.:
                raise IndexError(
                    'cannot broadcast nonzero value onto sparse vector; '
                    'sparse vectors do not have a defined length'
                )
        elif (value:= self.dtype(value)):
            self.dct[index] = value
    
    def __iadd__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if j: 
                    if i in dct:
                        dct[i] += self.dtype(j)
                    else:
                        dct[i] = self.dtype(j)
        elif isinstance(other, SparseVector):
            for i, j in other.dct.items():
                if i in dct:
                    dct[i] += j
                else:
                    dct[i] = j
        elif other != 0.:
            raise IndexError(
                'cannot broadcast nonzero value onto sparse vector; '
                'sparse arrays do not have a defined length'
            )
        return self
    
    def __add__(self, other):
        new = self.copy()
        new += other
        return new
            
    def __isub__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] - self.dtype(j)
                    if j:
                        dct[i] = -j
                    else: 
                        del dct[i]
                else:
                    dct[i] = -self.dtype(j)
        elif isinstance(other, SparseVector):
            for i, j in other.dct.items():
                if i in dct:
                    j = dct[i] - j
                    if j:
                        dct[i] = -j
                    else: 
                        del dct[i]
                else:
                    dct[i] = -j
        elif other != 0.:
            raise IndexError(
                'cannot broadcast nonzero value onto sparse vector; '
                'sparse arrays do not have a defined length'
            )
        return self
    
    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new
    
    def __imul__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i in dct:
                j = other[i]
                if j: dct[i] *= self.dtype(j)
                else: del dct[i]
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in dct:
                if i in other:
                    dct[i] *= other[i]
                else:
                    del dct[i]
        else:
            other = self.dtype(other)
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
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if i in dct: dct[i] /= self.dtype(j)
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in dct: dct[i] /= other[i]
        else:
            other = self.dtype(other)
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
        return SparseVector.from_dict({i: -j for i, j in self.dct.items()})
    
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
                if i in dct: dct[i] = j / dct[i] 
                elif j: raise FloatingPointError('division by zero')
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in other: 
                if i in dct: dct[i] = other[i] / dct[i]
                else: raise FloatingPointError('division by zero')
        else:
            other = self.dtype(other)
            if other:
                for i in dct: dct[i] = other / dct[i]
            else:
                dct = {}
        return SparseVector.from_dict(dct)
    
    shares_data_with = SparseArray.shares_data_with
    __repr__ = SparseArray.__repr__
    __str__ = SparseArray.__str__