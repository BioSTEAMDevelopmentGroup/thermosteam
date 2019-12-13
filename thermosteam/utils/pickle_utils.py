# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import pickle

__all__ = ('save', 'load')

def save(object, file):
    with open(file, 'wb') as f: pickle.dump(object, f)
    
def load(file):
    with open(file, "rb") as f: return pickle.load(f)