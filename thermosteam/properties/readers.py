# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:03:14 2020

@author: yoelr
"""
import os
import numpy as np
from typing import Dict
import pandas as pd
import json

__all__ = ('load_json',
           'CASDataSource', 
           'get_from_retrievers',
           'get_from_data_sources',
           'to_nums',
)

# %% Readers

class CASDataReader:
    __slots__ = ('folder',)
    def __init__(self, folder, data_folder=os.path.join(os.path.dirname(__file__), 'Data')):
        self.folder = os.path.join(data_folder, folder)
        
    def __call__(self, file, sep='\t', index_col=0,  **kwargs):
        path = os.path.join(self.folder, file)
        df = pd.read_csv(path,
                         sep=sep, index_col=index_col,
                         engine='python', **kwargs)
        return CASDataSource(df)

class CASDataSource:
    __slots__ = ('df', 'index', 'values')
    
    def __init__(self, df):
        self.df = df
        self.index = df.index
        self.values = df.values
    
    def retriever(self, var):
        return CASDataRetriever(self.df, var)
    
    def retrieve(self, CASRN, var):
        df = self.df
        if CASRN in df.index:
            value = df.at[CASRN, var]
            return None if np.isnan(value) else float(value)
        else:
            return None    
    
    @property
    def at(self):
        return self.df.at
    
    def __getitem__(self, CAS):
        return self.values[self.index.get_loc(CAS)]
    
    def __contains__(self, CAS):
        return CAS in self.index
    

class CASDataRetriever:
    __slots__ = ('df', 'var')
    
    def __init__(self, df, var):
        self.df = df
        self.var = var
        
    def __call__(self, CASRN):
        df = self.df
        if CASRN in df.index:
            value = df.at[CASRN, self.var]
            return None if np.isnan(value) else float(value)
        else:
            return None

CASDataSources = Dict[str, CASDataSource]
CASDataRetrievers = Dict[str, CASDataRetriever]

def retrievers_from_data_sources(sources: CASDataSources, var: str) -> dict:
    return {name: source.retriever(var)
            for name, source in sources.items()}

def get_from_retrievers(retrievers: CASDataRetrievers, CASRN: str, method:str):
    if method == 'All':
        value = {i: j(CASRN) for i,j in retrievers.items()}
    elif method == 'Any':
        for retriever in retrievers.values():
            value = retriever(CASRN)
            if value: break
    else:
        try:
            retriever = retrievers[method]
        except:
            raise ValueError("invalid method; method must be one of the following: "
                            f"{', '.join(retrievers)}.")
        value = retriever(CASRN)
    return value

def get_from_data_sources(sources: CASDataSources, CASRN: str, var: str, method:str):
    if method == 'All':
        value = {i: j.retrieve(CASRN, var) for i,j in sources.items()}
    elif method == 'Any':
        for source in sources.values():
            value = source.retrieve(CASRN, var)
            if value: break
    else:
        try:
            source = sources[method]
        except:
            raise ValueError("invalid method; method must be one of the following: "
                            f"{', '.join(sources)}.")
        value = source.retrieve(CASRN, var)
    return value

def to_nums(values):
    r'''Legacy function to turn a list of strings into either floats
    (if numeric), stripped strings (if not) or None if the string is empty.
    Accepts any numeric formatting the float function does.

    Parameters
    ----------
    values : list
        list of strings

    Returns
    -------
    values : list
        list of floats, strings, and None values [-]

    Examples
    --------
    >>> to_num(['1', '1.1', '1E5', '0xB4', ''])
    [1.0, 1.1, 100000.0, '0xB4', None]
    '''
    return [to_num(i) for i in values]

def to_num(value):
    try:
        return float(value)
    except:
        if value == '':
            return None
        else:
            return value.strip()

def load_json(folder, json_file, hook=None):
    with open(os.path.join(folder, json_file)) as f:
        return json.loads(f.read(), object_hook=hook)