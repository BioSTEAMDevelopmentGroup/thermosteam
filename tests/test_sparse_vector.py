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
from thermosteam.base import SparseVector, sparse_vector
from numpy.testing import assert_allclose

def test_sparse_vector_creation():
    arr = np.array([1., 2., 0., 4.5])
    sg = sparse_vector(arr)
    assert (sg.to_array(4) == arr).all()
    assert (sg.to_list(4) == arr).all()
    assert repr(sg) == '[1.0, 2.0, 0, 4.5, ...]'
    
    arr = np.array([1., 2., 0., 4.5, 0., 0.])
    sg = sparse_vector(arr)
    assert (sg.to_array(6) == arr).all()
    assert (arr == sg.to_list(6)).all()
    assert repr(sg) == '[1.0, 2.0, 0, 4.5, ...]'
    

def test_sparse_vector_math():
    arr = np.array([1., 2., 0., 4.5])
    sg = sparse_vector(arr)
    assert sg * 2 == sg + sg == 2. * arr
    assert sg * 0 == sg - sg == [0.]
    assert sg / sg == np.array([1., 1., 0., 1.])
    
    sg *= 0.
    assert sg == [0.]
    
    sg += arr
    assert sg == arr
    
    sg *= 2
    assert sg == 2. * arr

    sg /= 2.
    assert sg == arr
    
    sg /= sg
    assert sg == np.array([1., 1., 0., 1.])
    
def test_sparse_vector_indexing():
    arr = np.array([1., 2., 0., 4.5])
    sg = sparse_vector(arr)
    old_data = sg.data.copy()
    sg[20] = 0.
    assert sg.data == old_data
    sg[5] = 3.
    assert sg == np.array([1., 2., 0., 4.5, 0., 3.])
    sg[[1, 3]] = [0., 0.]
    assert sg == np.array([1., 0., 0., 0., 0., 3.])
    assert (sg[[0, 1, 5]] == np.array([1., 0., 3.])).all()
    assert sg[0] == 1.
    assert sg[4] == 0.
    assert sg[5] == 3.
    assert sg[10] == 0.

if __name__ == '__main__':
    test_sparse_vector_creation()
    test_sparse_vector_math()
    test_sparse_vector_indexing()