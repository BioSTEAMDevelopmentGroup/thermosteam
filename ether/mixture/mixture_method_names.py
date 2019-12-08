# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 06:24:40 2019

@author: yoelr
"""

__all__ = ('mixture_phaseT_methods', 
           'mixture_phaseTP_methods', 
           'mixture_T_methods', 
           'mixture_methods')

mixture_phaseT_methods = ('H', 'Cp')
mixture_phaseTP_methods = ('H_excess', 'S_excess', 'mu', 'V', 'k', 'S')
mixture_T_methods  = ('Hvap', 'sigma', 'epsilon')
mixture_methods = (*mixture_phaseT_methods,
                   *mixture_phaseTP_methods,
                   *mixture_T_methods)