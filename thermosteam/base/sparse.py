# -*- coding: utf-8 -*-
"""
"""
import numpy as np
__all__ = (
    'sparse_vector',
    'sparse_array',
    'SparseVector',
    'SparseArray',
)

def sparse_vector(arr, copy=False):
    """
    Convert array to a SparseVector object.

    """
    if isinstance(arr, SparseVector):
        return arr.copy() if copy else arr
    elif hasattr(arr, '__iter__'):
        dct = {}
        for i, j in enumerate(arr):
            if j: dct[int(i)] = float(j)
        return SparseVector(dct)
    else:
        raise TypeError(f'cannot convert {type(arr).__name__} object to a sparse vector')

def sparse_array(arr, copy=False):
    """
    Convert array to a SparseVector object.

    """
    if isinstance(arr, SparseArray):
        return arr.copy() if copy else arr
    elif hasattr(arr, '__iter__'):
        return SparseArray([sparse_vector(row) for row in arr])
    else:
        raise TypeError(f'cannot convert {type(arr).__name__} object to a sparse array')

class SparseArray:
    """
    Create a SparseArray object that can be used for array-like arrithmetic operations
    (i.e., +, -, *, /) of sparse 2-dimensional arrays. 
    
    In contrast to Scipy's sparse 2-D arrays, sparse arrays do not have a defined row length 
    (but still have a defined column length). 
    
    """
    __slots__ = ('rows',)
    def __init__(self, rows):
        self.rows = rows
        
    def __iter__(self):
        return iter(self.rows)
        
    def any(self):
        return any([i.any() for i in self.rows])
    
    def __eq__(self, other):
        return all([i.dct == j.dct for i, j in zip(self.rows, sparse_array(other).rows)])
    
    def __getitem__(self, index):
        rows = self.rows
        if hasattr(index, '__iter__'):
            i, j = index
            return rows[i][j]
        else:
            return rows[index]
        
    def __setitem__(self, index, value):
        rows = self.rows
        if hasattr(index, '__iter__'):
            i, j = index
            return rows[i][j]
        else:
            rows[index][:] = value
        

class SparseVector:
    """
    Create a SparseVector object that can be used for array-like arrithmetic operations
    (i.e., +, -, *, /) of sparse 1-dimensional arrays. 
    
    In contrast to Scipy's sparse 2-D arrays, sparse vectors can only represent 1-D arrays
    and do not have a defined size. 
    
    """
    __slots__ = ('dct',)
    
    def __init__(self, dct):
        self.dct = dct
    
    def __eq__(self, other):
        other = sparse_vector(other)
        return self.dct == other.dct
    
    def any(self):
        return self.dct.__bool__()
    
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
    
    def array_of(self, index):
        if hasattr(index, '__iter__'):
            dct = self.dct
            arr = np.zeros(len(index))
            for i in index:
                if (i:=int(i)) in dct: arr[i] = dct[i]
            return arr
        elif (index:=int(index)) in dct:
            return dct[index]
    
    def sum(self):
        return sum(self.dct.values())
    
    def max(self):
        return max(max(self.dct.values()), 0.)
    
    def min(self):
        return min(min(self.dct.values()), 0.)
    
    def to_array(self, length):
        arr = np.zeros(length)
        for i, j in self.dct.items(): arr[i] = j
        return arr
    
    def to_list(self, length):
        lst = [0] * length
        for i, j in self.dct.items(): lst[i] = j
        return lst
    
    def copy(self):
        return SparseVector(self.dct.copy())
    
    def copy_like(self, other):
        dct = self.dct
        dct.clear()
        dct.update(other.dct)
    
    def clear(self):
        self.dct.clear()
    
    def __getitem__(self, index):
        dct = self.dct
        if hasattr(index, '__iter__'):
            return SparseVector({j: dct[j] for i in index if (j:=int(i)) in dct})
        elif (index:=int(index)) in dct:
            return dct[index]
        else:
            return 0.

    def __setitem__(self, index, value):
        if hasattr(index, '__iter__'):
            dct = self.dct
            for i, j in zip(index, value): 
                if j: dct[int(i)] = float(j)
                elif (i:=int(i)) in dct: del dct[i]
        elif index == slice(None):
            dct = self.dct
            dct.clear()
            if hasattr(value, '__iter__'):
                for i, j in enumerate(value): 
                    if j: dct[i] = float(j)
                    elif i in dct: del dct[i]
            elif value != 0.:
                raise IndexError(
                    'cannot broadcast nonzero value onto sparse vector; '
                    'sparse vectors do not have a defined length'
                )
        elif (value:= float(value)):
            self.dct[index] = value
    
    def __iadd__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if j: 
                    if i in dct:
                        dct[i] += float(j)
                    else:
                        dct[i] = float(j)
        elif isinstance(other, SparseVector):
            for i, j in other.dct.items():
                if i in dct:
                    dct[i] += j
                else:
                    dct[i] = j
        else:
            raise TypeError('cannot convert {type(other).__name__} object to a sparse vector')
        return self
    
    def __add__(self, other):
        dct = self.dct.copy()
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] + float(j)
                    if j:
                        dct[i] = j
                    else: 
                        del dct[i]
                else:
                    dct[i] = float(j)
                
        elif isinstance(other, SparseVector):
            for i, j in other.dct.items():
                if i in dct:
                    j = dct[i] + j
                    if j:
                        dct[i] = j
                    else: 
                        del dct[i]
                else:
                    dct[i] = j
        else:
            raise TypeError('cannot convert {type(other).__name__} object to a sparse vector')
        return SparseVector(dct)
            
    def __isub__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] - float(j)
                    if j:
                        dct[i] = -j
                    else: 
                        del dct[i]
                else:
                    dct[i] = -float(j)
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
        else:
            raise TypeError('cannot convert {type(other).__name__} object to a sparse vector')
        return self
    
    def __sub__(self, other):
        dct = self.dct.copy()
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if not j: continue
                if i in dct:
                    j = dct[i] - float(j)
                    if j:
                        dct[i] = -j
                    else: 
                        del dct[i]
                else:
                    dct[i] = -float(j)
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
        else:
            raise TypeError('cannot convert {type(other).__name__} object to a sparse vector')
        return SparseVector(dct)
    
    def __imul__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i in dct:
                j = other[i]
                if j: dct[i] *= float(j)
                else: del dct[i]
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in dct:
                if i in other:
                    dct[i] *= other[i]
                else:
                    del dct[i]
        else:
            other = float(other)
            if other:
                for i in dct: dct[i] *= other
            else:
                dct.clear()
        return self
    
    def __mul__(self, other):
        dct = self.dct.copy()
        if hasattr(other, '__iter__'):
            for i in dct:
                j = other[i]
                if j: dct[i] *= float(j)
                else: del dct[i]
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in dct:
                if i in other:
                    dct[i] *= other[i]
                else:
                    del dct[i]
        else:
            other = float(other)
            if other:
                for i in dct: dct[i] *= other
            else:
                dct.clear()
        return SparseVector(dct)
        
    def __itruediv__(self, other):
        dct = self.dct
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if i in dct: dct[i] /= float(j)
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in dct: dct[i] /= other[i]
        else:
            other = float(other)
            if other:
                for i in dct: dct[i] /= other
            elif dct:
                raise FloatingPointError('division by zero')
        return self
    
    def __truediv__(self, other):
        dct = self.dct.copy()
        if hasattr(other, '__iter__'):
            for i, j in enumerate(other):
                if i in dct: dct[i] /= float(j)
        elif isinstance(other, SparseVector):
            other = other.dct
            for i in dct: dct[i] /= other[i]
        else:
            other = float(other)
            if other:
                for i in dct: dct[i] /= other
            elif dct:
                raise FloatingPointError('division by zero')
        return SparseVector(dct)
    
    def __repr__(self):
        if self.dct:
            lst = self.to_list(length=max(self.dct) + 1)
            lst = str(lst)
            return lst[:-1] + ', ...]'
        else:
            return '[...]'
    