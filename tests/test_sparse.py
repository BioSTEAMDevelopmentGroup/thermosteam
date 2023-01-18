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
from thermosteam.base import (
    SparseVector, SparseArray, sparse_vector, sparse_array,
    nonzero_items, sparse
)
from numpy.testing import assert_allclose

def assert_no_zero_data(arr):
    if isinstance(arr, SparseVector):
        assert 0 not in arr.dct.values()
    elif isinstance(arr, SparseArray):
        assert all([0 not in i.dct.values() for i in arr])

def test_sparse_vector_creation():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    assert_no_zero_data(sv)
    assert (sv.to_array() == arr).all()
    assert repr(sv) == 'sparse([1. , 2. , 0. , 4.5])'
    assert str(sv) == '[1.  2.  0.  4.5]'
    
    arr = np.array([1., 2., 0., 4.5, 0., 0.])
    sv = sparse_vector(arr)
    assert_no_zero_data(sv)
    assert (sv.to_array() == arr).all()
    assert repr(sv) == 'sparse([1. , 2. , 0. , 4.5, 0. , 0. ])'
    assert str(sv) == '[1.  2.  0.  4.5 0.  0. ]'
    
    assert (sparse(arr) == arr).all()
    
    
def test_sparse_array_creation():
    arr = np.array([[1., 2., 0., 4.5]])
    sa = sparse_array(arr)
    assert_no_zero_data(sa)
    assert (sa.to_array() == arr).all()
    assert repr(sa) == 'sparse([[1. , 2. , 0. , 4.5]])'
    assert str(sa) == '[[1.  2.  0.  4.5]]'
    
    arr = np.array([[1., 2., 0., 4.5, 0., 0.]])
    sa = sparse_array(arr)
    assert_no_zero_data(sa)
    assert (sa.to_array() == arr).all()
    assert repr(sa) == 'sparse([[1. , 2. , 0. , 4.5, 0. , 0. ]])'
    assert str(sa) == '[[1.  2.  0.  4.5 0.  0. ]]'
    
    assert (sparse(arr) == arr).all()
    
    arr = np.ones([2, 3, 1])
    with pytest.raises(ValueError):
        sparse(arr)
        
    sa = SparseArray()
    assert (sa == np.ones([0, 0])).all()
    
    with pytest.raises(TypeError):
        SparseArray(0)
        
    with pytest.raises(ValueError):
        bool(sa)
        
def test_sparse_vector_easy_methods():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    assert [*sv.nonzero_items()] == nonzero_items(arr) == [(0, 1.), (1, 2.), (3, 4.5)]
    assert [*sv.nonzero_values()] == [1, 2, 4.5]
    nonzero_index, = sv.nonzero_index()
    assert [*sv.nonzero_keys()] == nonzero_index == [0, 1, 3]
    n_sv = -sv
    assert (n_sv == -arr).all()
    assert set(n_sv.negative_keys()) == n_sv.nonzero_keys()
    sv_index, = n_sv.negative_index()
    arr_index, = np.where(-arr < 0)
    (sv_index == arr_index).all()
    
    sv_index, = sv.positive_index()
    arr_index, = np.where(arr > 0)
    (sv_index == arr_index).all()
    
    n_sv.remove_negatives()
    assert (n_sv == np.zeros_like(sv)).all()

def test_sparse_array_easy_methods():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse_array(arr)
    sa_left_index, sa_right_index = sa.nonzero_index()
    arr_left_index, arr_right_index = arr.nonzero()
    assert (sa_left_index == arr_left_index).all()
    assert (sa_right_index == arr_right_index).all()
    arr_nonzero_values = arr[arr_left_index, arr_right_index]
    assert (np.array([*sa.nonzero_values()]) == arr_nonzero_values).all()
    for ((i, j), k), ak in zip(sa.nonzero_items(), arr_nonzero_values):
        assert sa[i, j] == k == ak

    assert [*sa.negative_keys()] == []
    assert [*sa.negative_rows()] == []
    n_sa = -sa
    assert (n_sa == -arr).all()
    assert set(n_sa.negative_keys()) == sa.nonzero_keys()
    assert set(n_sa.negative_rows()) == {0, 1}
    n_sa.remove_negatives()
    assert (n_sa == np.zeros_like(sa)).all()
    

def test_sparse_vector_math():
    arr = np.array([1., -2., 0., 4.5])
    sv = sparse_vector(arr)
    arr = abs(arr)
    sv = abs(sv)
    assert (sv == arr).all()
    assert (sv * 2).sparse_equal(2 * arr)
    assert (sv + sv).sparse_equal(2 * arr)
    assert (sv * 0).sparse_equal([])
    assert (sv - sv).sparse_equal([])
    assert (sv / sv).sparse_equal(np.array([1., 1., 0., 1.]))
    
    assert_no_zero_data(sv * 2)
    assert_no_zero_data(sv + sv)
    assert_no_zero_data(sv * 0)
    assert_no_zero_data(sv - sv)
    assert_no_zero_data(sv / sv)
    
    sv *= 0.
    assert sv.sparse_equal([0.])
    assert_no_zero_data(sv)
    
    sv += arr
    assert sv.sparse_equal(arr)
    assert_no_zero_data(sv)
    
    sv *= 2
    assert sv.sparse_equal(2. * arr)
    assert_no_zero_data(sv)
    
    sv /= 2.
    assert sv.sparse_equal(arr)
    assert_no_zero_data(sv)
    
    sv /= sv
    assert sv.sparse_equal(np.array([1., 1., 0., 1.]))
    assert_no_zero_data(sv)
    
    sv[:] *= 0.
    assert sv.sparse_equal([0.])
    assert_no_zero_data(sv)
    
    sv[:] += arr
    assert sv.sparse_equal(arr)
    assert_no_zero_data(sv)
    
    sv[:] *= 2
    assert sv.sparse_equal(2. * arr)
    assert_no_zero_data(sv)
    
    sv[:] /= 2.
    assert sv.sparse_equal(arr)
    assert_no_zero_data(sv)
    
    sv[:] /= sv
    assert sv.sparse_equal(np.array([1., 1., 0., 1.]))
    assert_no_zero_data(sv)
    sv[:] = 1
    assert_no_zero_data(sv)
    assert (2. / sv).sparse_equal([2., 2., 2., 2.])
    assert ([2, 1, 0, 3] / sv).sparse_equal([2, 1, 0, 3])
    
    sv[:] = [0., 1]
    assert_no_zero_data(sv)
    
    assert ((0 / sv) == np.zeros_like(sv)).all()
    
    with pytest.raises(FloatingPointError):
        [2, 1, 0.1, 3] / sv
        
    with pytest.raises(FloatingPointError):
        sv /= 0

    

def test_sparse_array_math():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse_array(-arr)
    assert_no_zero_data(sa)
    sa = abs(sa)
    assert (sa == arr).all()
    assert (sa * 2).sparse_equal(2. * arr)
    assert (sa + sa).sparse_equal(2. * arr)
    assert (sa + arr).sparse_equal(2. * arr)
    assert (sa * sa).sparse_equal(arr * arr)
    assert (sa + sa).sparse_equal(2. * arr)
    assert (sa * 0).sparse_equal([[0.], [0.]])
    assert (sa - sa).sparse_equal([[0.], [0.]])
    assert (sa - 1).sparse_equal(arr - 1)
    assert (1 - sa).sparse_equal(1 - arr)
    assert (1 + sa).sparse_equal(1 + arr)
    assert (sa / sa).sparse_equal(np.array([[1., 1., 0., 1.], [0., 0., 1., 1.]]))
    assert (sa / 1 == arr / 1).all()
    assert (sa / 2 == arr / 2).all()
    sparse_ones = sparse(np.ones([3, 2]))
    assert (2 / (2. * sparse_ones) == np.ones([3, 2])).all()
    assert (sparse_ones * 2 / sparse_ones == 2 * np.ones([3, 2])).all()
    
    assert_no_zero_data(sa * 2)
    assert_no_zero_data(sa + sa)
    assert_no_zero_data(sa * 0)
    assert_no_zero_data(sa - sa)
    assert_no_zero_data(sa / sa)
    
    sa *= 0.
    assert sa.sparse_equal([[0.], [0.]])
    assert_no_zero_data(sa)
    
    sa += arr
    assert sa.sparse_equal(arr)
    assert_no_zero_data(sa)
    
    sa -= arr
    assert sa.sparse_equal([[0.], [0.]])
    assert_no_zero_data(sa)
    
    sa += arr
    assert sa.sparse_equal(arr)
    assert_no_zero_data(sa)
    
    sa *= 2
    assert sa.sparse_equal(2. * arr)
    assert_no_zero_data(sa)
    
    sa /= 2.
    assert sa.sparse_equal(arr)
    assert_no_zero_data(sa)
    
    sa /= sa
    assert sa.sparse_equal(np.array([[1., 1., 0., 1.], [0., 0., 1., 1.]]))
    assert_no_zero_data(sa)
    
    sa[:] *= 0.
    assert sa.sparse_equal([[0.], [0.]])
    assert_no_zero_data(sa)
    
    sa[:] += arr
    assert sa.sparse_equal(arr)
    assert_no_zero_data(sa)
    
    sa[:] -= arr
    assert sa.sparse_equal([[0.], [0.]])
    assert_no_zero_data(sa)
    
    sa[:] += arr
    assert sa.sparse_equal(arr)
    assert_no_zero_data(sa)
    
    sa[:] *= 2
    assert sa.sparse_equal(2. * arr)
    assert_no_zero_data(sa)
    
    sa[:] /= 2.
    assert sa.sparse_equal(arr)
    assert_no_zero_data(sa)
    
    sa[:] /= sa
    assert sa.sparse_equal(np.array([[1., 1., 0., 1.], [0., 0., 1., 1.]]))
    assert_no_zero_data(sa)
    
def test_sparse_vector_indexing():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    assert sv.size == 4 # Size is not strict
    old_dct = sv.dct.copy()
    sv[20] = 0.
    assert_no_zero_data(sv)
    assert sv.dct == old_dct
    sv[5] = 3.
    assert_no_zero_data(sv)
    assert sv.sparse_equal(np.array([1., 2., 0., 4.5, 0., 3.]))
    sv[[1, 3]] = [0., 0.]
    assert_no_zero_data(sv)
    assert sv.sparse_equal(np.array([1., 0., 0., 0., 0., 3.]))
    assert (sv[[0, 1, 5]] == np.array([1., 0., 3.])).all()
    assert sv[0] == 1.
    assert sv[4] == 0.
    assert sv[5] == 3.
    assert sv[10] == 0.
    sv[[0, 1, 5]] = 1.
    assert_no_zero_data(sv)
    assert sv.sparse_equal(np.array([1., 1., 0., 0., 0., 1.]))
    sv[:] = 2.
    assert_no_zero_data(sv)
    assert sv.size == 4
    assert sv.sparse_equal([2., 2., 2., 2.])
    sv[(1,)] = 3
    assert_no_zero_data(sv)
    assert sv[(1,)] == sv[1] == 3
    assert (sv[([1, 2],)] == np.array([3, 2])).all()
    assert (sv[[1, 2]] == np.array([3, 2])).all()
    
    arr = np.ones(10)
    sv = sparse(arr)
    sv[:2] = 3
    arr[:2] = 3
    assert (sv == arr).all()
    sv[0::2] = -2
    arr[0::2] = -2
    assert (sv == arr).all()
    sv[0:6:2] = -2
    arr[0:6:2] = -2
    assert (sv == arr).all()
    
    with pytest.raises(IndexError):
        sv[(1, 2)]
    with pytest.raises(IndexError):
        sv[(1, 2)] = 0.
    with pytest.raises(IndexError):
        sv[(1,)] = [0., 1]
    with pytest.raises(IndexError):
        sv[1] = [0., 1]

def test_sparse_array_indexing():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse_array(arr)
    
    with pytest.raises(IndexError):
        sa[(1, 1, 0)]
    
    with pytest.raises(IndexError):
        sa[(1, 1, 0)] = 2
    
    old_rows = [i.copy() for i in sa.rows]
    sa[0, 20] = 0.
    assert_no_zero_data(sa)
    assert (sa == old_rows).all()
    sa[0, 5] = 3.
    assert_no_zero_data(sa)
    assert sa[0].sparse_equal(np.array([1., 2., 0., 4.5, 0., 3.]))
    assert sa.sparse_equal(
        np.array([[1., 2., 0., 4.5, 0., 3.],
                  [0., 0., 1., 1.5, 0,  0.]])
    )
    assert sa[[0, 1]].sparse_equal(sa)
    sa[0, [0, 1]] = [0., 0.]
    assert_no_zero_data(sa)
    assert sa[0].sparse_equal(np.array([0. , 0. , 0. , 4.5, 0. , 3. ]))
    assert (sa[0, [0, 1, 5]] == np.array([0., 0., 3.])).all()
    assert sa[0, 0] == 0.
    assert sa[0, 4] == 0.
    assert sa[0, 5] == 3.
    assert sa[0, 10] == 0.
    sa[1] = 0
    assert_no_zero_data(sa)
    assert sa[1].sparse_equal([])
    assert sa[0].sparse_equal(np.array([0. , 0. , 0. , 4.5, 0. , 3. ]))
    sa[:, 1] = 2
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        np.array([[0., 2., 0., 4.5, 0., 3.],
                  [0., 2., 0., 0. , 0,  0.]])
    )
    sa[:, [1, 2]] = 1.
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        np.array([[0., 1., 1., 4.5, 0., 3.],
                  [0., 1., 1., 0. , 0,  0.]])
    )
    sa[:] = 0
    assert_no_zero_data(sa)
    assert sa.sparse_equal([])
    
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    sa[0, :] = sv
    assert_no_zero_data(sa)
    assert sa[0, :].sparse_equal(arr)
    sa[0, :] = 0.
    assert_no_zero_data(sa)
    assert sa[0, :].sparse_equal([])
    sa[[0, 1], :] = sv
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        [[1., 2., 0., 4.5],
         [1., 2., 0., 4.5]]
    )
    sa[[0, 1], :] = [0, 1, 0]
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        [[0, 1],
         [0, 1]]
    )
    sa[[0, 1], :] = [[0, 1], [1, 0]]
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        [[0, 1],
         [1, 0]]
    )
    sa[:] = 0
    assert_no_zero_data(sa)
    sa[:, [1, 3]] = [[0, 1],
                     [1, 0]]
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        [[0, 0, 0, 1],
         [0, 1, 0, 0]]
    )
    sa[:] = 2.
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        [[2., 2., 2., 2.],
         [2., 2., 2., 2.]]
    )
    sa[[0, 1], :] = 3.
    assert_no_zero_data(sa)
    assert sa.sparse_equal(
        [[3., 3., 3., 3.],
         [3., 3., 3., 3.]]
    )
    with pytest.raises(IndexError):
        sa[[[0, 1], [2, 3]]] = 2
    
    sa[[0, 1], [2, 3]] = 2
    assert_no_zero_data(sa)
    assert (sa[[0, 1], [2, 3]] == np.array([2., 2.])).all()
    assert sa.sparse_equal(
        [[3., 3., 2., 3.],
         [3., 3., 3., 2.]]
    )

    sa_not_a_view = sa[:, [1, 2]]
    with pytest.raises(ValueError):
        sa_not_a_view[:] = 20
    
    assert (sa_not_a_view ==
        [[3., 2.],
         [3., 3.]]
    ).all()
    
    arr = sa.to_array()
    assert (sa[[0, 1], :] == arr[[0, 1], :]).all()
    
    with pytest.raises(IndexError):
        sa[[[0, 1]], :]
    
    with pytest.raises(IndexError):
        sa[[[0, 1]], [[0]]]    
    
    assert (sa[[0, 1], 0] == arr[[0, 1], 0]).all()
    
    with pytest.raises(IndexError):
        sa[[[1, 2], 3]]
    
    sa[:, [0, 1]] = [5, 6]
    arr[:, [0, 1]] = [5, 6]
    assert (sa[:, [0, 1]] == [5, 6]).all()
    assert (arr[:, [0, 1]] == [5, 6]).all()

    sa[[0, 1], :] = 0
    arr[[0, 1], :] = 0
    assert (arr[[0, 1], :] == sa[[0, 1], :]).all()
    
    sa[0:2, :] = 2
    arr[0:2, :] = 2
    assert (sa[0:2, :] == arr[0:2, :]).all()
    sa[:, :1] = 2
    arr[:, :1] = 2
    assert (sa[:, :1] == arr[:, :1]).all()
    sa[:2] = 2 
    arr[:2] = 2
    assert (sa[:3] == arr[:3]).all()
    
    arr = np.ones([3,4])
    sa = sparse(arr)
    sa[[True, False, True]] = 2
    arr[[True, False, True]] = 2
    assert (sa == arr).all() and (sa[[True, False, True]] == arr[[True, False, True]]).all()
    sa[[True, False, True], :2] = 3
    arr[[True, False, True], :2] = 3
    assert (sa == arr).all() and (sa[[True, False, True], :2] == arr[[True, False, True], :2]).all()
    sa[0, :2] = 3
    arr[0, :2] = 3
    assert (sa == arr).all() and (sa[0, :2] == arr[0, :2]).all()
    sa[0, 2:4] = 10
    arr[0, 2:4] = 10
    assert (sa == arr).all() and (sa[0, 2:4] == arr[0, 2:4]).all()
    sa[[0, 2]] = 0
    arr[[0, 2]] = 0
    assert (sa == arr).all() and (sa[[0, 2]] == arr[[0, 2]]).all()
    sa[[0, 1], [2, 0]] = 0
    arr[[0, 1], [2, 0]] = 0
    assert (sa == arr).all() and (sa[[0, 1], [2, 0]] == arr[[0, 1], [2, 0]]).all()

    with pytest.raises(IndexError):
        sa[(0,)] # Too few indices, array is 2 dimensional but only 1 dimension was indexed

def test_sparse_array_boolean_indexing():
    arr = np.array([[1., 0.], [0., 1.]])
    sa = sparse_array(arr)
    assert (sa[sa == 0] == arr[arr == 0]).all()
    sa[sa == 0] = 1.
    arr[arr == 0] = 1.
    assert (sa == arr).all()
    sa[sa == 1] = [0., 1, 1, 0.]
    arr[arr == 1] = [0., 1, 1, 0.]
    assert (sa == arr).all()
    mask = (arr == 0).any(0)
    assert (sa[mask] == arr[mask]).all()
    arr[mask] *= 2
    sa[mask] *= 2
    assert (sa[mask] == arr[mask]).all()
    sa[mask] = 2
    arr[mask] = 2
    assert (sa[mask] == arr[mask]).all()
    arr = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    sa = SparseArray(arr)
    assert (arr[[True, False, True], :] == sa[[True, False, True], :]).all()
    assert (arr[:, [True, False, True]] == sa[:, [True, False, True]]).all()


def test_sparse_vector_boolean_indexing():
    arr = np.array([1., 0., 0., 1.])
    sv = sparse_vector(arr)
    assert (sv[sv == 0] == arr[arr == 0]).all()
    sv[sv == 0] = 1.
    arr[arr == 0] = 1.
    assert (sv == arr).all()
    sv[sv == 1] = [0., 1, 1, 0.]
    arr[arr == 1] = [0., 1, 1, 0.]
    assert (sv == arr).all()

def test_sparse_vector_special_methods():
    arr = np.array([1., 2., 0., 4.5])
    sv = sparse_vector(arr)
    assert (sv.to_flat_array() == np.array([1., 2., 0., 4.5])).all()
    sv.from_flat_array(np.array([1., 2., 0., 2]))
    assert (sv.to_flat_array() == np.array([1., 2., 0., 2])).all()
    
    with pytest.raises(IndexError):
        sv[[[True, False]]] # boolean index dimensions is off
    
    with pytest.raises(IndexError):
        sv[[[True, False]]] = [2, 2] # boolean index dimensions is off
    
    with pytest.raises(IndexError):
        sv[[[1, 2]]] # index dimensions is off

    with pytest.raises(IndexError):
        sv[[[1, 2]]] = [2, 3] # index dimensions is off
        
    with pytest.raises(IndexError):
        sv[[1, 2]] = [[2, 3]] # value dimensions is off
    
    with pytest.raises(IndexError):
        sv[:] = [[2]] # value dimensions is off

def test_sparse_array_special_methods():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse_array(arr)
    assert (sa.to_flat_array() == np.array([1., 2., 0., 4.5, 0., 0., 1., 1.5])).all()
    sa.from_flat_array(np.array([1., 2., 0., 0, 0., 0., 1., 2]))
    assert (sa.to_flat_array() == np.array([1., 2., 0., 0, 0., 0., 1., 2])).all()
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse(arr)
    assert (sa.sum_of([3, 1], axis=0) == [6., 2.]).all()
    assert (sa.sum_of([0, 2], axis=1) == [1, 1]).all()
    assert set(zip(*sa.positive_index())) == set([(int(i), int(j)) for i,j in zip(*np.where(arr > 0))])
    assert set(zip(*(-sa).negative_index())) == set([(int(i), int(j)) for i,j in zip(*np.where(arr > 0))])
    assert sa.nonzero_rows() == [0, 1]
    sa[:] = 0
    assert sa.nonzero_rows() == []
    assert [*nonzero_items(sa)] == [*sa.nonzero_items()] == [*nonzero_items(sa.to_array())]
    assert [*nonzero_items(sa[0])] == [*sa[0].nonzero_items()]
    
    sa = sparse([[1, 2], [3, 4]])
    sa_other = sa.copy()
    sa.copy_like(sa + 1)
    assert (sa == sa_other + 1).all()
    
    arr = np.array([[1., 2., 0., 4.5], [1., 2., 0., 4.5]])
    sa = sparse_array(arr)
    assert (sa.to_flat_array() == np.array([1., 2., 0., 4.5, 1., 2., 0., 4.5])).all()
    sa.from_flat_array(np.array([1., 2., 0., 2, 1., 2., 0., 2]))
    assert (sa == np.array([[1., 2., 0., 2], [1., 2., 0., 2]])).all()
    
def test_descriptive_methods():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse(arr)
    assert (sa.value == sa.to_array()).all()
    assert sa.vector_size == sa[0].size == sa[0].vector_size == arr[0].size
    assert sa.shape == arr.shape
    assert sa[0].shape == arr[0].shape

def test_read_only_flag():
    arr = np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]])
    sa = sparse(arr)
    sa.setflag(0)
    with pytest.raises(ValueError):
        sa[:] = 1
   
    arr = np.ones(3)
    sv = sparse(arr)
    sv.read_only = True
    with pytest.raises(ValueError):
        sv[:] = 1

def test_sparse_vector_methods_vs_numpy():
    arrays = [
        np.array([1., 2., 0., 4.5]),
        np.zeros(3),
        np.ones(3),
    ]
    for arr in arrays:
        sv = sparse_vector(arr)
        for method in ('min', 'max', 'argmin', 'argmax', 'mean', 'sum', 'any', 'all'):
            sv_method = getattr(sv, method)
            arr_method = getattr(arr, method)
            for axis in (0, None, 1):
                for keepdims in (False, True):
                    if axis == 1:
                        with pytest.raises(ValueError):
                            np.asarray(sv_method(axis=axis, keepdims=keepdims))
                        with pytest.raises(ValueError): # For good measure
                            arr_method(axis=axis, keepdims=keepdims)
                        continue
                    sv_result = np.asarray(sv_method(axis=axis, keepdims=keepdims))
                    arr_result = arr_method(axis=axis, keepdims=keepdims)
                    assert sv_result.shape == arr_result.shape, f"wrong shape in SparseVector.{method} with axis {axis} and keepdims {keepdims}"
                    assert (sv_result == arr_result).all(), f"wrong value in SparseVector.{method} with axis {axis} and keepdims {keepdims}"

def test_sparse_array_methods_vs_numpy():
    arrays = [
        np.array([[1., 2., 0., 4.5], [0., 0., 1., 1.5]]),
        np.zeros([2, 3]),
        np.ones([2, 3]),
    ]
    for arr in arrays:
        sa = sparse_array(arr)
        for method in ('min', 'max', 'argmin', 'argmax', 'mean', 'sum', 'any', 'all'):
            sa_method = getattr(sa, method)
            arr_method = getattr(arr, method)
            for axis in (0, 1, None, 2):
                for keepdims in (False, True):
                    if axis == 2:
                        with pytest.raises(ValueError):
                            np.asarray(sa_method(axis=axis, keepdims=keepdims))
                        with pytest.raises(ValueError): # For good measure
                            arr_method(axis=axis, keepdims=keepdims)
                        continue
                    sa_result = np.asarray(sa_method(axis=axis, keepdims=keepdims))
                    arr_result = arr_method(axis=axis, keepdims=keepdims)
                    assert sa_result.shape == arr_result.shape, f"wrong shape in SparseArray.{method} with axis {axis} and keepdims {keepdims}"
                    assert (sa_result == arr_result).all(), f"wrong value in SparseArray.{method} with axis {axis} and keepdims {keepdims}"

    sa = sparse([[]])
    assert sa.vector_size == 0
    with pytest.raises(ValueError):
        sa.min()
    with pytest.raises(ValueError):
        sa.max()

if __name__ == '__main__':
    test_sparse_vector_creation()
    test_sparse_array_creation()
    test_sparse_vector_easy_methods()
    test_sparse_array_easy_methods()
    test_sparse_vector_math()
    test_sparse_array_math()
    test_sparse_vector_indexing()
    test_sparse_array_indexing()
    test_sparse_vector_boolean_indexing()
    test_sparse_array_boolean_indexing()
    test_sparse_vector_special_methods()
    test_sparse_array_special_methods()
    test_descriptive_methods()
    test_sparse_vector_methods_vs_numpy()
    test_sparse_array_methods_vs_numpy()