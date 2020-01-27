# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 04:50:05 2020

@author: yoelr
"""
import thermosteam as tmo
import doctest

def test_stream_docs():
    doctest.testmod(tmo._stream)