# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 03:53:15 2019

@author: yoelr
"""

__all__ = ('assert_same_chemicals',)

def assert_same_chemicals(obj, others):
    chemicals = obj.chemicals
    assert all([chemicals is i.chemicals for i in others]), "chemicals must match to mix streams"