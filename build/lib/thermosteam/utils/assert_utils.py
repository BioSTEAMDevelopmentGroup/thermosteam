# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 03:53:15 2019

@author: yoelr
"""

__all__ = ('assert_same_chemicals',)

def chemicals_match(chemical_users, chemicals):
    return all([chemicals is i.chemicals for i in chemical_users if i])

def assert_same_chemicals(obj, others):
    chemicals = obj.chemicals
    assert chemicals_match(others, chemicals), (
           "chemicals must match to mix streams")