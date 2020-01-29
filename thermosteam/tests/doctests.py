# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 04:50:05 2020

@author: yoelr
"""
import thermosteam as tmo
import doctest

def test_stream_docs():
    doctest.testmod(tmo._stream)
    
def test_multi_stream_docs():
    doctest.testmod(tmo._multi_stream)
    
def test_chemicals_docs():
    doctest.testmod(tmo._chemicals)
    
def test_chemical_docs():
    doctest.testmod(tmo._chemical)
    
if __name__ == '__main__':
    test_stream_docs()
    test_multi_stream_docs()
    test_chemicals_docs()
    test_chemical_docs()