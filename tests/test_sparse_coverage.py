# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
import operator as op
import numpy as np
from thermosteam.base import (
    SparseVector, SparseLogicalVector, SparseArray, sparse_vector, sparse_array,
    nonzero_items, sparse
)
from thermosteam.base.sparse import (
    get_ndim, reduce_ndim, sum_sparse_vectors, unpack_index,
)
from numpy.testing import assert_allclose

def dense(x):
    return x.to_array() if hasattr(x, 'to_array') else np.asarray(x)

def assert_same(got, expected):
    got = dense(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape: got = got.reshape(expected.shape)
    assert_allclose(got.astype(float), expected.astype(float))

float_ops = [op.add, op.sub, op.mul, op.eq, op.ne, op.gt, op.lt, op.ge, op.le]
float_iops = [op.iadd, op.isub, op.imul]
bool_ops = [op.and_, op.xor, op.or_, op.eq, op.ne, op.gt, op.lt, op.ge, op.le, op.add, op.mul]
bool_iops = [op.iand, op.ixor, op.ior, op.iadd, op.imul]

def test_float_math_against_numpy():
    a = np.array([0., 1., 2., -3.])
    b = np.array([1., 0., 2., 0.])
    makers = {
        'sv': SparseVector,
        'sa1': lambda x: SparseArray([x]),
        'sa2': lambda x: SparseArray([x, x]),
    }
    others = {
        'sv': SparseVector,
        'arr': lambda x: x.copy(),
        'sa1': lambda x: SparseArray([x]),
        'sa2': lambda x: SparseArray([x, x]),
        'zero': lambda x: 0.,
        'two': lambda x: 2.,
        'size1': lambda x: SparseVector([2.]),
        'arr1': lambda x: np.array([2.]),
    }
    for xname, mx in makers.items():
        for yname, my in others.items():
            for f in float_ops:
                x = mx(a); y = my(b)
                assert_same(f(x, y), f(dense(x), dense(y)))
            if yname in ('size1', 'arr1'): continue
            for f in float_iops:
                if xname == 'sv' and yname in ('sa1', 'sa2'): continue
                if xname == 'sa1' and yname == 'sa2': continue
                x = mx(a); y = my(b)
                expected = f(dense(x).copy(), dense(y))
                assert_same(f(x, y), expected)
    # Division with nonzero divisors
    c = np.array([1., 4., 2., -1.])
    for xname, mx in makers.items():
        for y in (SparseVector(c), c.copy(), SparseArray([c]), 2., SparseVector([2.])):
            x = mx(a)
            assert_same(x / y, dense(x) / dense(y))
            if not (xname == 'sv' and y.__class__ is SparseArray) and not (y.__class__ is SparseVector and y.size == 1 and xname != 'sv'):
                x = mx(a)
                expected = dense(x) / dense(y)
                x /= y
                assert_same(x, expected)
    # Reverse operations
    sv = SparseVector(c)
    assert_same(1. + sv, 1. + c)
    assert_same(1. - sv, 1. - c)
    assert_same(2. * sv, 2. * c)
    assert_same(2. / sv, 2. / c)
    assert_same(c / sv, np.ones(4))
    sa = SparseArray([c, c])
    assert_same(1. + sa, 1. + np.array([c, c]))
    assert_same(1. - sa, 1. - np.array([c, c]))
    assert_same(2. * sa, 2. * np.array([c, c]))
    assert_same(2. / sa, 2. / np.array([c, c]))
    assert_same(np.array([c, c]) / sa, np.ones([2, 4]))
    assert_same(-sa, -np.array([c, c]))
    assert_same(abs(sa), abs(np.array([c, c])))
    assert_same(abs(sv), abs(c))

def test_logical_math_against_numpy():
    la = np.array([True, False, True, False])
    lb = np.array([False, False, True, True])
    makers = {
        'slv': SparseLogicalVector,
        'sa1': lambda x: SparseArray([x]),
    }
    others = {
        'slv': SparseLogicalVector,
        'arr': lambda x: x.copy(),
        'sa1': lambda x: SparseArray([x]),
        'true': lambda x: True,
        'false': lambda x: False,
        'size1': lambda x: SparseLogicalVector([True]),
        'arr1': lambda x: np.array([True]),
    }
    for xname, mx in makers.items():
        for yname, my in others.items():
            for f in bool_ops:
                x = mx(la); y = my(lb)
                assert_same(f(x, y), f(dense(x), dense(y)))
            if yname in ('size1', 'arr1'): continue
            for f in bool_iops:
                if xname == 'slv' and yname == 'sa1': continue
                x = mx(la); y = my(lb)
                expected = f(dense(x).copy(), dense(y))
                assert_same(f(x, y), expected)
    # Mixed logical and float operations
    fb = np.array([1., 0., 2., 0.])
    for x in (SparseLogicalVector(la), SparseArray([la])):
        for y in (SparseVector(fb), fb.copy(), 2., SparseArray([fb])):
            for f in (op.add, op.mul, op.eq, op.ne, op.gt, op.lt, op.ge, op.le):
                assert_same(f(x, y), f(dense(x).astype(float), dense(y)))
    slv = SparseLogicalVector(la)
    assert_same(~slv, ~la)
    assert_same(~SparseArray([la]), ~np.array([la]))
    assert_same(-slv, -la.astype(float))
    assert_same(abs(slv), la)
    with pytest.raises(ValueError):
        SparseArray([la]).__iadd__(SparseArray([fb]))
    with pytest.raises(ValueError):
        SparseArray([la]).__iadd__(SparseVector(fb))

def test_sparse_array_reductions_against_numpy():
    arr = np.array([[0., 1., 2.], [3., 0., -1.], [0., 0., 0.]])
    sa = SparseArray(arr)
    for name in ('sum', 'mean', 'max', 'min', 'any', 'all'):
        for axis in (None, 0, 1):
            for keepdims in (False, True):
                got = getattr(sa, name)(axis=axis, keepdims=keepdims)
                expected = getattr(arr, name)(axis=axis, keepdims=keepdims)
                assert_same(got, expected)
        with pytest.raises(ValueError):
            getattr(sa, name)(axis=2)
    larr = np.array([[True, False, True], [True, True, True]])
    sla = SparseArray(larr)
    for name in ('any', 'all', 'sum', 'max', 'min'):
        for axis in (None, 0, 1):
            for keepdims in (False, True):
                got = getattr(sla, name)(axis=axis, keepdims=keepdims)
                expected = getattr(larr, name)(axis=axis, keepdims=keepdims)
                assert_same(got, expected)
    assert_same(sa.sum_of([0, 2], axis=0), arr[:, [0, 2]].sum(axis=0))
    assert_same(sa.sum_of([0, 2], axis=1), arr[:, [0, 2]].sum(axis=1))
    with pytest.raises(ValueError): sa.sum_of(0, axis=2)
    empty = SparseArray.from_rows([])
    assert empty.all(axis=0).size == 0

def test_sparse_vector_reductions_against_numpy():
    for arr in (np.array([0., 1., 2.]), np.array([0., -1., -2.]),
                np.array([1., 2., 3.]), np.array([-1., -2., -3.]), np.zeros(3)):
        sv = SparseVector(arr)
        for name in ('sum', 'mean', 'max', 'min', 'any', 'all'):
            for keepdims in (False, True):
                assert_same(getattr(sv, name)(keepdims=keepdims),
                            getattr(arr, name)(keepdims=keepdims))
            with pytest.raises(ValueError):
                getattr(sv, name)(axis=1)
    empty = SparseVector.from_size(0)
    with pytest.raises(ValueError): empty.max()
    with pytest.raises(ValueError): empty.min()
    for larr in (np.array([True, False]), np.array([True, True]), np.array([False, False])):
        slv = SparseLogicalVector(larr)
        for name in ('sum', 'mean', 'max', 'min', 'any', 'all'):
            for keepdims in (False, True):
                assert_same(getattr(slv, name)(keepdims=keepdims),
                            getattr(larr, name)(keepdims=keepdims))
            with pytest.raises(ValueError):
                getattr(slv, name)(axis=1)
        assert slv.sum_of([0, 1]) == larr.sum()
        assert slv.sum_of(0) == larr[0]
    with pytest.raises(ValueError): SparseLogicalVector.from_size(0).max()
    with pytest.raises(ValueError): SparseLogicalVector.from_size(0).min()

def test_sparse_array_methods():
    arr = np.array([[0., 1., -2.], [3., 0., 0.]])
    sa = SparseArray(arr)
    assert sa.shape == (2, 3) and sa.size == 6 and sa.vector_size == 3
    assert_same(sa.value, arr)
    assert_same(sa.astype(float), arr)
    assert sa.tolist() == arr.tolist()
    assert sa.nonzero_rows() == [0, 1]
    assert sa.nonzero_keys() == {0, 1, 2}
    assert sorted(sa.nonzero_values()) == [-2., 1., 3.]
    assert dict(sa.nonzero_items()) == {(0, 1): 1., (0, 2): -2., (1, 0): 3.}
    assert sa.positive_index() == ([0, 1], [1, 0])
    assert sa.negative_index() == ([0], [2])
    assert sa.negative_rows() == [0]
    assert sa.negative_keys() == {2}
    assert sa.has_negatives()
    flat = sa.to_flat_array()
    assert_same(flat, arr.flatten())
    out = np.ones(6)
    sa.to_flat_array(out)
    assert_same(out, arr.flatten())
    sa2 = SparseArray.from_shape((2, 3))
    sa2.from_flat_array(flat)
    assert sa2.sparse_equal(sa)
    assert_same(sa.flatten(), arr.flatten())
    sa.remove_negatives()
    assert not sa.has_negatives()
    assert sa.shares_data_with(sa.rows[0])
    assert sa.shares_data_with(SparseArray.from_rows([sa.rows[1]]))
    assert not sa.shares_data_with(sa.copy())
    assert not sa.shares_data_with(SparseVector.from_size(3))
    c = sa.copy()
    c.clear()
    assert not c.any()
    c.copy_like(sa)
    assert c.sparse_equal(sa)
    sa.setflags(0)
    assert sa.rows[0].read_only
    with pytest.raises(NotImplementedError): sa.setflags(1)
    with pytest.raises(TypeError): SparseArray(1)
    assert len(SparseArray()) == 0
    # Scalars
    one = SparseArray([[2.]])
    assert float(one) == 2. and int(one) == 2 and bool(one)
    for f in (float, int, bool):
        with pytest.raises(ValueError): f(SparseArray(arr))
    # Boolean arrays
    larr = np.array([[True, False], [False, True]])
    sla = SparseArray(larr)
    assert sla.dtype is bool
    assert sla.positive_index() == sla.nonzero_index()
    assert sla.negative_index() == ([], [])
    assert sla.negative_rows() == []
    assert sla.negative_keys() == set()
    assert not sla.has_negatives()
    assert dict(sla.nonzero_items()) == {(0, 0): True, (1, 1): True}
    assert_same(sla.flatten(), larr.flatten())
    sla2 = SparseArray([[False, False], [False, False]])
    sla2.from_flat_array(larr.flatten())
    assert sla2.sparse_equal(sla)
    assert_same(abs(sla), larr)
    # Flat views
    sa = SparseArray(np.array([[0., 1., 2.], [3., 0., 4.]]))
    flat = sa.flat
    assert flat is sa.flat
    assert list(flat) == [0., 1., 2., 3., 0., 4.]
    assert len(flat) == 6
    assert flat[1] == 1. and flat[-1] == 4.
    assert_same(flat[[0, 3, -1]], [0., 3., 4.])
    assert_same(flat[np.array([True, False, False, True, False, False])], [0., 3.])
    assert_same(flat[1:4], [1., 2., 3.])
    assert flat[:] is flat
    assert flat[(2,)] == 2.
    with pytest.raises(IndexError): flat[[[0]]]
    with pytest.raises(IndexError): flat[np.array([[True]])]
    assert_same(flat == 0., np.array([1, 0, 0, 0, 1, 0]))
    assert_same(flat != 0., np.array([0, 1, 1, 1, 0, 1]))
    assert_same(flat > 1., np.array([0, 0, 1, 1, 0, 1]))
    assert_same(flat < 1., np.array([1, 0, 0, 0, 1, 0]))
    assert_same(flat >= 1., np.array([0, 1, 1, 1, 0, 1]))
    assert_same(flat <= 1., np.array([1, 1, 0, 0, 1, 0]))
    assert 'sparse' in repr(flat) and str(flat)
    flat[0] = 5.
    flat[-1] = 0.
    flat[(1,)] = 0.
    assert sa.to_array().tolist() == [[5., 0., 2.], [3., 0., 0.]]
    flat[[0, 1]] = [1., 0.]
    flat[[2, -1]] = 7.
    flat[[2]] = 0.
    flat[np.array([False, False, False, False, True, False])] = 9.
    assert sa.to_array().tolist() == [[1., 0., 0.], [3., 9., 7.]]
    flat[:] = 1.
    assert sa.to_array().sum() == 6.
    flat[:] = np.arange(6.)
    assert_same(sa, np.arange(6.).reshape(2, 3))
    flat[:] = flat
    flat[1:3] = 0.
    assert_same(sa, [[0., 0., 0.], [3., 4., 5.]])
    with pytest.raises(IndexError): flat[[0]] = [[1.], [2.]]
    with pytest.raises(IndexError): flat[:] = [[1., 2.], [3., 4.]]
    with pytest.raises(IndexError): flat[[[0]]] = 1.
    with pytest.raises(IndexError): flat[0] = [1., 2.]
    with pytest.raises(IndexError): flat[np.array([[True]])] = 1.
    # Logical flat view
    sla = SparseArray(np.array([[True, False], [False, True]]))
    lflat = sla.flat
    assert list(lflat) == [True, False, False, True]
    assert_same(lflat == True, [True, False, False, True])
    lflat[[1]] = [True]
    lflat[[1]] = [False]
    lflat[[2]] = True
    lflat[[2]] = False
    lflat[np.array([False, True, False, False])] = True
    lflat[:] = [True, False, False, False]
    lflat[:] = True
    lflat[:] = lflat
    lflat[0:2] = [False, False]
    assert_same(sla, [[False, False], [True, True]])
    with pytest.raises(IndexError): lflat[[0]] = [[1.], [2.]]
    with pytest.raises(IndexError): lflat[:] = [[True, False], [False, True]]
    with pytest.raises(IndexError): lflat[[[0]]] = True
    with pytest.raises(IndexError): lflat[0] = [True, False]
    with pytest.raises(IndexError): lflat[np.array([[True]])] = True

def test_sparse_vector_methods():
    sv = SparseVector([0., 1., -2.])
    assert sv.vector_size == 3 and sv.shape == (3,) and len(sv) == 3
    assert sv.tolist() == [0., 1., -2.]
    out = np.ones(3)
    assert_same(sv.to_flat_array(out), [0., 1., -2.])
    assert_same(sv.to_flat_array(), [0., 1., -2.])
    other = SparseVector.from_size(3)
    other.from_flat_array(np.array([0., 1., -2.]))
    assert other.sparse_equal(sv)
    assert sv.negative_keys() == {2}
    assert sv.negative_index() == ([2],)
    assert sv.positive_index() == ([1],)
    assert sv.nonzero_index() == ([1, 2],)
    assert set(sv.nonzero_keys()) == {1, 2}
    assert sum(sv.nonzero_values()) == -1.
    assert sv.sum_of([1, 2]) == -1. and sv.sum_of(1) == 1.
    sa = SparseArray.from_rows([sv])
    assert sv.shares_data_with(sa)
    assert not sv.shares_data_with(SparseArray([[1.]]))
    assert sv.shares_data_with(sv)
    c = sv.copy()
    c.copy_like(sv)
    c.copy_like(c)
    assert c.sparse_equal(sv)
    c.remove_negatives()
    assert not c.has_negatives() and sv.has_negatives()
    # Mixing
    x = SparseVector([1., 2., 0.])
    y = SparseVector([0., 1., 3.])
    x.mix_from([x, y])
    assert x.tolist() == [1., 3., 3.]
    x.mix_from([x, x])
    assert x.tolist() == [2., 6., 6.]
    x.mix_from([y, y])
    assert x.tolist() == [0., 2., 6.]
    x.mix_from([SparseVector([0., -2., 0.]), SparseVector([0., 2., 0.])])
    assert x.tolist() == [0., 0., 0.]
    x[1] = 1.
    x.mix_from([])
    assert not x.dct
    # Constructors
    with pytest.raises(ValueError): SparseVector()
    with pytest.raises(ValueError): SparseVector({0: 1.})
    assert SparseVector({0: 1., 1: 0.}, size=2).tolist() == [1., 0.]
    assert SparseVector(sv, size=5).size == 5
    with pytest.raises(TypeError): SparseVector(1.)
    # Read only
    ro = sv.copy()
    ro.setflags(0)
    with pytest.raises(ValueError): ro.clear()
    with pytest.raises(ValueError): ro[0] = 1.
    with pytest.raises(NotImplementedError): ro.setflags(1)
    # Scalars
    assert float(SparseVector([3.])) == 3.
    assert int(SparseVector([3.])) == 3
    # Indexing
    sv = SparseVector([0., 1., 2., 3.])
    assert sv[(1,)] == 1.
    assert_same(sv[[0, 3]], [0., 3.])
    assert_same(sv[np.array([True, False, False, True])], [0., 3.])
    assert_same(sv[1:3], [1., 2.])
    with pytest.raises(IndexError): sv[[[0]]]
    with pytest.raises(IndexError): sv[0, 1]
    with pytest.raises(IndexError): sv[()]
    sv[[0, 1]] = [5., 0.]
    sv[[0, 1]] = 1.
    sv[[0]] = 0.
    sv[np.array([False, False, True, False])] = 4.
    assert sv.tolist() == [0., 1., 4., 3.]
    sv[:] = sv
    sv[:] = SparseVector([1., 0., 0., 0.])
    sv[:] = [0., 1., 0., 2.]
    sv[:] = 0.
    sv[:] = 2.
    sv[1:3] = 0.
    assert sv.tolist() == [2., 0., 0., 2.]
    sv[0] = 0.
    assert sv.tolist() == [0., 0., 0., 2.]
    with pytest.raises(IndexError): sv[[0]] = [[1.], [2.]]
    with pytest.raises(IndexError): sv[:] = [[1., 2.], [3., 4.]]
    with pytest.raises(IndexError): sv[[[0]]] = 1.
    with pytest.raises(IndexError): sv[0] = [1., 2.]
    with pytest.raises(IndexError): sv[np.array([[True]])] = 1.

def test_sparse_logical_vector_methods():
    slv = SparseLogicalVector([True, False, True])
    assert slv.dct == {0: 1., 2: 1.}
    assert slv.vector_size == 3 and slv.shape == (3,)
    assert list(slv) == [True, False, True]
    assert slv.tolist() == [True, False, True]
    out = np.zeros(3, bool)
    assert_same(slv.to_flat_array(out), [True, False, True])
    assert_same(slv.to_flat_array(), [True, False, True])
    other = SparseLogicalVector.from_size(3)
    other.from_flat_array(np.array([True, False, True]))
    assert other.sparse_equal(slv)
    slv.remove_negatives()
    assert not slv.has_negatives()
    assert slv.nonzero_index() == ([0, 2],)
    assert set(slv.nonzero_keys()) == {0, 2}
    assert list(slv.nonzero_values()) == [True, True]
    assert dict(slv.nonzero_items()) == {0: True, 2: True}
    assert slv.copy().sparse_equal(slv)
    with pytest.raises(ValueError): SparseLogicalVector()
    assert SparseLogicalVector({1}, size=3).tolist() == [False, True, False]
    with pytest.raises(ValueError): SparseLogicalVector({1})
    assert SparseLogicalVector(slv, size=4).size == 4
    with pytest.raises(TypeError): SparseLogicalVector(1)
    # Indexing
    assert slv[(0,)] == True
    assert_same(slv[[0, 1]], [True, False])
    assert_same(slv[np.array([True, False, True])], [True, True])
    assert_same(slv[1:3], [False, True])
    assert slv[:] is slv
    with pytest.raises(IndexError): slv[[[0]]]
    slv[[0, 1]] = [False, True]
    slv[[0, 1]] = True
    slv[[1]] = False
    slv[np.array([False, True, False])] = True
    assert slv.tolist() == [True, True, True]
    slv[:] = slv
    slv[:] = SparseLogicalVector([False, True, False])
    slv[:] = [True, False, False]
    slv[:] = False
    slv[:] = True
    slv[0:2] = False
    slv[2] = False
    assert slv.tolist() == [False, False, False]
    with pytest.raises(IndexError): slv[[0]] = [[True], [False]]
    with pytest.raises(IndexError): slv[:] = [[True, False], [False, True]]
    with pytest.raises(IndexError): slv[[[0]]] = True
    with pytest.raises(IndexError): slv[0] = [True, False]

def test_sparse_array_indexing():
    arr = np.array([[0., 1., 2.], [3., 0., 4.], [0., 5., 0.]])
    sa = SparseArray(arr)
    m = [0, 2]; n = [1, 2]
    assert_same(sa[1:, 1], arr[1:, 1])
    assert_same(sa[:, 1], arr[:, 1])
    assert sa[:, :] is sa
    assert_same(sa[m, :], arr[m, :])
    assert_same(sa[np.array([True, False, True]), :], arr[[0, 2], :])
    assert_same(sa[[True, False, True], :], arr[[0, 2], :])
    assert_same(sa[0, 1:], arr[0, 1:])
    assert_same(sa[m, 1:], arr[m, 1:])
    assert_same(sa[[True, False, True], 1:], arr[[0, 2], 1:])
    with pytest.raises(IndexError): sa[[[0]], :]
    with pytest.raises(IndexError): sa[[[0]], 1:]
    assert sa[1, 2] == 4.
    assert_same(sa[m, 1], arr[m, 1])
    assert_same(sa[m, n], arr[m, n])
    with pytest.raises(IndexError): sa[m, [[0]]]
    with pytest.raises(IndexError): sa[[[0]], 0]
    assert_same(sa[[True, False, True]], arr[[0, 2]])
    assert_same(sa[np.array([[True, False, False], [False, False, True], [False, False, False]])], [0., 4.])
    assert_same(sa[[0, 1]], arr[[0, 1]])
    with pytest.raises(IndexError): sa[[[0]]]
    assert sa[:] is sa
    assert_same(sa[1], arr[1])
    assert_same(sa[0:2], arr[0:2])
    larr = arr.astype(bool)
    sla = SparseArray(larr)
    assert_same(sla[m, 1], larr[m, 1])
    assert_same(sla[m, n], larr[m, n])
    with pytest.raises(IndexError): sla[m, [[0]]]

def test_sparse_array_setitem():
    def check(index, value):
        arr = np.array([[0., 1., 2.], [3., 0., 4.], [0., 5., 0.]])
        sa = SparseArray(arr)
        arr[index] = value
        sa[index] = value
        assert_same(sa, arr)
    check((slice(None), slice(None)), 1.)
    check((slice(None), slice(None)), [1., 2., 3.])
    check((slice(None), slice(None)), np.ones([3, 3]))
    check((slice(None), slice(1, 3)), 7.)
    check((slice(None), slice(1, 3)), [7., 8.])
    check((slice(1, 3), slice(None)), 7.)
    check((slice(1, 3), 0), 7.)
    check((slice(1, 3), 0), [7., 8.])
    check((slice(1, 3), [0, 1]), [7., 8.])
    check((slice(1, 3), np.array([True, False, True])), [7., 8.])
    check((slice(1, 3), [0, 1]), np.ones([2, 2]))
    check((0, slice(None)), 9.)
    check(([0, 2], slice(None)), 9.)
    check(([0, 2], slice(None)), np.ones([2, 3]))
    check((np.array([True, False, True]), slice(None)), 9.)
    check((1, 1), 9.)
    check(([0, 1], 0), 9.)
    check(([0, 1], 0), 0.)
    check(([0, 1], 0), [9., 0.])
    check(([0, 1], [0, 2]), 9.)
    check(([0, 1], [0, 2]), 0.)
    check(([0, 1], [0, 2]), [9., 0.])
    check(np.array([True, False, True]), 1.)
    sa = SparseArray(np.zeros([3, 3]))
    sa[np.array([True, False, True])] = np.ones([3, 3])
    assert_same(sa, [[1., 1., 1.], [0., 0., 0.], [1., 1., 1.]])
    check(np.array([[True, False, False], [False, False, True], [False, False, False]]), 1.)
    check([0, 2], 1.)
    check([0, 2], np.ones([2, 3]))
    check(slice(1, 3), 1.)
    check(slice(None), 2.)
    check(1, [1., 2., 3.])
    sa = SparseArray(np.zeros([2, 2]))
    with pytest.raises(IndexError): sa[:, :] = np.ones([2, 2, 2])
    with pytest.raises(IndexError): sa[:, 0] = np.ones([2, 2, 2])
    with pytest.raises(IndexError): sa[[0, 1], :] = np.ones([2, 2, 2])
    with pytest.raises(IndexError): sa[[[0]], :] = 1.
    with pytest.raises(IndexError): sa[[[0]], 0] = 1.
    with pytest.raises(IndexError): sa[[0], 0] = np.ones([2, 2])
    with pytest.raises(IndexError): sa[[0], [0]] = np.ones([2, 2])
    with pytest.raises(IndexError): sa[[0], [[0]]] = 1.
    with pytest.raises(IndexError): sa[[0, 1]] = np.ones([2, 2, 2])
    # Boolean arrays
    larr = np.array([[True, False], [False, True]])
    def check_bool(index, value):
        arr = larr.copy()
        sla = SparseArray(arr)
        arr[index] = value
        sla[index] = value
        assert_same(sla, arr)
    check_bool(([0, 1], [1, 0]), True)
    check_bool(([0, 1], [0, 1]), False)
    check_bool(([0, 1], [0, 1]), [False, True])
    sla = SparseArray(larr)
    with pytest.raises(IndexError): sla[[0], [0]] = np.ones([2, 2], bool)

def test_sparse_constructors_and_utilities():
    assert sparse(SparseVector([1.])).__class__ is SparseVector
    assert sparse([True, False]).__class__ is SparseLogicalVector
    assert sparse([1., 0.]).__class__ is SparseVector
    assert sparse([[1., 0.]]).__class__ is SparseArray
    with pytest.raises(ValueError): sparse(np.ones([2, 2, 2]))
    assert isinstance(SparseVector([1.]), sparse)
    assert not isinstance(np.ones(2), sparse)
    sa = SparseArray([[1., 0.]])
    assert sparse_array(sa) is sa
    assert sparse_array(sa, copy=True) is not sa
    assert sparse_array([[1., 0.]]).sparse_equal(sa)
    sv = SparseVector([1., 0.])
    assert sparse_vector(sv, copy=True) is not sv
    assert sparse_vector([True, False]).__class__ is SparseLogicalVector
    assert nonzero_items(np.array([0., 2.])) == [(1, 2.)]
    assert list(nonzero_items(sv)) == [(0, 1.)]
    assert get_ndim([[1., 2.]]) == 2
    assert get_ndim(np.ones(3)) == 1
    with pytest.raises(ValueError): get_ndim([[[[1.]]]])
    assert reduce_ndim([[1.]])[1] == 0
    assert reduce_ndim(np.array([[1.]]))[1] == 0
    assert reduce_ndim([[1., 2.], [3., 4.]])[1] == 2
    assert sum_sparse_vectors([]) == {}
    assert sum_sparse_vectors([SparseVector([1., 0.]), SparseVector([1., 2.])]) == {0: 2., 1: 2.}
    assert sum_sparse_vectors([SparseLogicalVector([True, False]), SparseLogicalVector([True, True])]) == {0: 2., 1: 1.}
    with pytest.raises(IndexError): unpack_index((1,), 2)
    with pytest.raises(IndexError): unpack_index((1, 2, 3), 2)

if __name__ == '__main__':
    test_float_math_against_numpy()
    test_logical_math_against_numpy()
    test_sparse_array_reductions_against_numpy()
    test_sparse_vector_reductions_against_numpy()
    test_sparse_array_methods()
    test_sparse_vector_methods()
    test_sparse_logical_vector_methods()
    test_sparse_array_indexing()
    test_sparse_array_setitem()
    test_sparse_constructors_and_utilities()
