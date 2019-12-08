# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:24:05 2019

@author: yoelr
"""
from ..exceptions import DimensionError
from .units_of_measure import ureg

__all__ = ('DisplayUnits',)

class DisplayUnits:
    """Create a DisplayUnits object where default units for representation are stored."""
    def __init__(self, **display_units):
        dct = self.__dict__
        dct.update(display_units)
        dct['dims'] = {}
        list_keys = []
        for k, v in display_units.items():
            try: # Assume units is one string
                dims = getattr(ureg, v).dimensionality
            except:
                try: # Assume units are a list of possible units
                    dims = [getattr(ureg, i).dimensionality for i in v]
                    list_keys.append(k)
                except: # Assume the user uses value as an option, and ignores units
                    dims = v
            self.dims[k] = dims
        for k in list_keys:
            dct[k] = dct[k][0] # Default units is first in list
    
    def get_units(self, **overide):
        default = self.__dict__
        units = []
        for i in overide:
            unit = overide[i]
            if unit is not None:
                units.append(unit)
            else:
                units.append(default[i])
        return units
    
    def __setattr__(self, name, unit):
        if name not in self.__dict__:
            raise AttributeError(f"can't set display units for '{name}'")
        if isinstance(unit, str):
            name_dim = self.dims[name]
            unit_dim = getattr(ureg, unit).dimensionality
            if isinstance(name_dim, list):
                if unit_dim not in name_dim:
                    name_dim = [f"({i})" for i in name_dim]
                    raise DimensionError(f"dimensions for '{name}' must be either {', '.join(name_dim[:-1])} or {name_dim[-1]}; not ({unit_dim})")    
            else:
                if name_dim != unit_dim:
                    raise DimensionError(f"dimensions for '{name}' must be in ({name_dim}), not ({unit_dim})")
        object.__setattr__(self, name, unit)
            
    def __repr__(self):
        sig = ', '.join((f"{i}='{j}'" if isinstance(j, str) else f'{i}={j}') for i,j in self.__dict__.items() if i != 'dims')
        return f'{type(self).__name__}({sig})'
