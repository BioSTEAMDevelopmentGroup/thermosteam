# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 06:24:40 2019

@author: yoelr
"""

__all__ = ('mixture_phaseT_methods', 
           'mixture_phaseTP_methods',
           'mixture_hidden_T_methods',
           'mixture_hidden_phaseTP_methods',
           'mixture_T_methods', 
           'mixture_methods')

mixture_phaseT_methods = ('Cp',)
mixture_hidden_T_methods = ('_H',)
mixture_phaseTP_methods = ('mu', 'V', 'k', 'S')
mixture_hidden_phaseTP_methods = ('_H_excess', '_S_excess', '_H')
mixture_T_methods  = ('Hvap', 'sigma', 'epsilon')
mixture_methods = (*mixture_phaseT_methods,
                   *mixture_phaseTP_methods,
                   *mixture_hidden_T_methods,
                   *mixture_hidden_phaseTP_methods,
                   *mixture_T_methods)