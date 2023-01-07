# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentVector/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
import numpy as np
from thermosteam.base import SparseVector, SparseArray, sparse_vector, sparse_array
from numpy.testing import assert_allclose

def test_sparse_vector_creation():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    assert (sv.to_array(4) == arr).all()
    assert repr(sv) == 'SparseVector([1. , 2. , 0. , 4.5])'
    assert str(sv) == '[1.  2.  0.  4.5]'
    
    arr = np.array([1., 2., 0., 4.5, 0., 0.])
    sv = sparse_vector(arr)
    assert (sv.to_array(6) == arr).all()
    assert repr(sv) == 'SparseVector([1. , 2. , 0. , 4.5])'
    assert str(sv) == '[1.  2.  0.  4.5]'
    
def test_sparse_array_creation():
    arr = np.array([[1., 2., 0., 4.5]])
    sa = sparse_array(arr)
    assert (sa.to_array(4) == arr).all()
    assert repr(sa) == 'SparseArray([[1. , 2. , 0. , 4.5]])'
    assert str(sa) == '[[1.  2.  0.  4.5]]'
    
    arr = np.array([[1., 2., 0., 4.5, 0., 0.]])
    sa = sparse_array(arr)
    assert (sa.to_array(6) == arr).all()
    assert repr(sa) == 'SparseArray([[1. , 2. , 0. , 4.5]])'
    assert str(sa) == '[[1.  2.  0.  4.5]]'

def test_sparse_vector_math():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    assert sv * 2 == sv + sv == 2. * arr
    assert sv * 0 == sv - sv == [0.]
    assert sv / sv == np.array([1., 1., 0., 1.])
    
    sv *= 0.
    assert sv == [0.]
    
    sv += arr
    assert sv == arr
    
    sv *= 2
    assert sv == 2. * arr

    sv /= 2.
    assert sv == arr
    
    sv /= sv
    assert sv == np.array([1., 1., 0., 1.])
    
    assert 2. / sv == [2., 2., 0., 2.]
    assert [2, 1, 0, 3] / sv == [2, 1, 0, 3]
    
    with pytest.raises(FloatingPointError):
        [2, 1, 0.1, 3] / sv
    
def test_sparse_array_math():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse_array(arr)
    assert sa * 2 == sa + sa == 2. * arr
    assert sa * 0 == sa - sa == [[0.], [0.]]
    assert sa / sa == np.array([[1., 1., 0., 1.],
                                [0., 0., 1., 1.]])
    
    sa *= 0.
    assert sa == [[0.], [0.]]
    
    sa += arr
    assert sa == arr
    
    sa *= 2
    assert sa == 2. * arr

    sa /= 2.
    assert sa == arr
    
    sa /= sa
    assert sa == np.array([[1., 1., 0., 1.],
                           [0., 0., 1., 1.]])
    
def test_sparse_vector_indexing():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    old_dct = sv.dct.copy()
    sv[20] = 0.
    assert sv.dct == old_dct
    sv[5] = 3.
    assert sv == np.array([1., 2., 0., 4.5, 0., 3.])
    sv[[1, 3]] = [0., 0.]
    assert sv == np.array([1., 0., 0., 0., 0., 3.])
    assert (sv[[0, 1, 5]] == np.array([1., 0., 3.])).all()
    assert sv[0] == 1.
    assert sv[4] == 0.
    assert sv[5] == 3.
    assert sv[10] == 0.
    sv[[0, 1, 5]] = 1.
    assert sv == np.array([1., 1., 0., 0., 0., 1.])
    with pytest.raises(IndexError):
        sv[:] = 2.
        

def test_sparse_array_indexing():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse_array(arr)
    old_rows = [i.copy() for i in sa.rows]
    sa[0, 20] = 0.
    assert sa.rows == old_rows
    sa[0, 5] = 3.
    assert sa[0] == np.array([1., 2., 0., 4.5, 0., 3.])
    assert sa == np.array([[1., 2., 0., 4.5, 0., 3.],
                           [0., 0., 1., 1.5, 0,  0.]])
    assert sa[[0, 1]] == sa
    sa[0, [0, 1]] = [0., 0.]
    assert sa[0] == np.array([0. , 0. , 0. , 4.5, 0. , 3. ])
    assert (sa[0, [0, 1, 5]] == np.array([0., 0., 3.])).all()
    assert sa[0, 0] == 0.
    assert sa[0, 4] == 0.
    assert sa[0, 5] == 3.
    assert sa[0, 10] == 0.
    sa[1] = 0
    assert sa[1] == []
    assert sa[0] == np.array([0. , 0. , 0. , 4.5, 0. , 3. ])
    sa[:, 1] = 2
    assert sa == np.array([[0., 2., 0., 4.5, 0., 3.],
                           [0., 2., 0., 0. , 0,  0.]])
    sa[:, [1, 2]] = 1.
    assert sa == np.array([[0., 1., 1., 4.5, 0., 3.],
                           [0., 1., 1., 0. , 0,  0.]])
    sa[:] = 0
    assert sa == []
    
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    sa[0, :] = sv
    assert sa[0, :] == arr
    sa[0, :] = 0.
    assert sa[0, :] == []
    sa[[0, 1], :] = sv
    assert sa == [[1., 2., 0., 4.5],
                  [1., 2., 0., 4.5]]
    sa[[0, 1], :] = [0, 1, 0]
    assert sa == [[0, 1],
                  [0, 1]]
    sa[[0, 1], :] = [[0, 1], [1, 0]]
    assert sa == [[0, 1],
                  [1, 0]]
    sa[:] = 0
    sa[:, [1, 3]] = [[0, 1],
                     [1, 0]]
    assert sa == [[0, 0, 0, 1],
                  [0, 1, 0, 0]]
    
    with pytest.raises(IndexError):
        sa[:] = 2.
        
    with pytest.raises(IndexError):
        sa[[0, 1], :] = 2.

if __name__ == '__main__':
    test_sparse_vector_creation()
    test_sparse_array_creation()
    test_sparse_vector_math()
    test_sparse_array_math()
    test_sparse_vector_indexing()
    test_sparse_array_indexing()