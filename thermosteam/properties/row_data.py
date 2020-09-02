# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:03:14 2020

@author: yoelr
"""
import os
import numpy as np
import json

__all__ = (
    'load_json',
    'RowData',
)

# %% Readers
as_value = lambda value: None if np.isnan(value) else float(value)

class RowData:
    """
    Create a RowData object for fast data frame lookups by row.
    
    """
    __slots__ = ('df', 'row_index', 'col_index', 'values')
    
    def __init__(self, df):
        self.row_index = {key: index for index, key in enumerate(df.index)}
        self.col_index = {key: index for index, key in enumerate(df)}
        self.values = df.values
    
    def get_values(self, row, cols):
        if row in self.row_index: 
            return [as_value(self.get_value(row, col)) for col in cols]
        else:
            raise ValueError(f'row {row} not in {type(self).__name__} object')
    
    def get_value(self, row, col):
        row_index = self.row_index
        col_index = self.col_index
        if row in row_index:
            row = row_index[row]
        else:
            raise LookupError(f'row {row} not in {type(self).__name__} object')
        if col in col_index:
            col = col_index[col]
        else:
            raise LookupError(f'column {col} not in {type(self).__name__} object')
        return self.values[row, col]
    
    def __getitem__(self, row):
        return self.values[self.row_index[row]]
    
    def __contains__(self, row):
        return row in self.index
   
    
def load_json(folder, json_file, hook=None):
    with open(os.path.join(folder, json_file)) as f:
        return json.loads(f.read(), object_hook=hook)