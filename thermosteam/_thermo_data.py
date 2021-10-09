# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from collections.abc import Iterable
import thermosteam as tmo

__all__ = ('ThermoData',)

# %% Utilities

def as_IDs(IDs):
    isa = isinstance
    if isa(IDs, Iterable):
        for i in IDs: 
            if not isa(i, str):
                raise ValueError('IDs must be an iterable of strings')
    else:
        raise ValueError('IDs must be an iterable of strings')
    return IDs

def chemicals_from_data(data, all_data):
    new_chemicals = dict(data)
    chemical_copies = {}
    for ID, kwargs in data.items():
        if kwargs:
            if 'Copy of' in kwargs:
                chemical_copies[ID] = kwargs
            else:
                new_chemicals[ID] = tmo.Chemical(ID, **kwargs)
        else:
            new_chemicals[ID] = tmo.Chemical(ID)
    for ID, kwargs in chemical_copies.items():
        copied_ID = kwargs.pop('Copy of')
        try:
            copied_chemical = new_chemicals[copied_ID]
        except KeyError:
            new_chemicals[copied_ID] = copied_chemical = tmo.Chemical(copied_ID, **all_data[copied_ID])
        new_chemicals[ID] = copied_chemical.copy(ID, **kwargs)
    return tmo.Chemicals([new_chemicals[i] for i in data])


# %% 

class ThermoData:
    """
    Create a ThermoData object for creating thermodynamic property packages
    and streams.
    
    Parameters
    ----------
    data : dict
        
    Examples
    --------
    >>> import thermosteam as tmo
    >>> data = {
    ... 'Chemicals': {
    ...     'Water': {},
    ...     'Ethanol': {},
    ...     'O2': {'phase': 'gas'},
    ...     'Cellulose': {
    ...          'search_db': False,
    ...          'phase': 'solid',
    ...          'formula': 'C6H10O5',
    ...          'Hf': -975708.8,
    ...          'default': True
    ...          },
    ...     'Octane': {}
    ...     },
    ... 'Streams': {
    ...     'process_water': {
    ...         'Water': 500,
    ...         'units': 'kg/hr',
    ...         'price': 0.00035,
    ...         },
    ...     'gasoline': {
    ...         'Octane': 400,
    ...         'units': 'kg/hr',
    ...         'price': 0.756,
    ...         },
    ...     }
    ... }
    >>> thermo_data = tmo.ThermoData(data)
    >>> chemicals = thermo_data.create_chemicals()
    >>> chemicals
    Chemicals([Water, Ethanol, O2, Cellulose, Octane])
    >>> tmo.settings.set_thermo(chemicals)
    >>> thermo_data.create_streams()
    [<Stream: process_water>, <Stream: gasoline>]
    
    It is also possible to create a ThermoData object from json or yaml files
    For example, lets say we have a yaml file that looks like this:
    
    .. code-block:: yaml
    
        # File name: example_chemicals.yaml
        Chemicals:
          Water:
          Ethanol:
          O2:
            phase: gas
          Cellulose:
            search_db: False
            phase: solid
            formula: C6H10O5
            Hf: -975708.8
            default: True
          Octane:
    
    Then we could create the chemicals in just a few lines:
    
    >>> # thermo_data = tmo.ThermoData.from_yaml('example_chemicals.yaml')
    >>> # thermo_data.create_chemicals()
    >>> # Chemicals([Water, Ethanol, O2, Cellulose, Octane])
    
    """
    __slots__ = ('data',)
    
    def __init__(self, data: dict):
        data_fields = ('Chemicals', 'Streams', 'Synonyms')
        self.data = {i: data.get(i, {})
                    for i in data_fields}
        
    @classmethod
    def from_yaml(cls, file):
        """Create a ThermoData object from a yaml file given its directory."""
        import yaml
        with open(file, 'r') as stream: 
            data = yaml.full_load(stream)
            assert isinstance(data, dict), 'yaml file must return a dict' 
            return ThermoData(data)
    
    @classmethod
    def from_json(cls, file):
        """Create a ThermoData object from a json file given its directory."""
        import json
        with open(file, 'r') as stream: 
            data = json.load(stream)
            assert isinstance(data, dict), 'json file must return a dict' 
            return ThermoData(data)
    
    @property
    def chemical_data(self):
        try: data = self.data['Chemicals']
        except KeyError: raise AttributeError('no chemical data available')
        return data
    
    @property
    def stream_data(self):
        try: data = self.data['Streams']
        except KeyError: raise AttributeError('no stream data available')
        return data
    
    @property
    def synonym_data(self):
        try: data = self.data['Synonyms']
        except KeyError: raise AttributeError('no synonym data available')
        return data
    
    def create_stream(self, ID):
        """
        Create stream from data. 
        
        Parameters
        ----------
        ID=None : str
            ID of stream to create.
        
        """
        data = self.stream_data[ID]
        return tmo.Stream(ID, **data) if data else tmo.Stream(ID)
    
    def create_streams(self, IDs=None):
        """
        Create streams from data. 
        
        Parameters
        ----------
        IDs=None : Iterable[str], optional
            IDs of streams to create. Defaults to all streams.
        
        """
        data = self.stream_data
        if IDs: data = {i:data[i] for i in as_IDs(IDs)}
        return [tmo.Stream(i, **(j or {})) for i, j in data.items()]
    
    def create_chemicals(self, IDs=None):
        """
        Create chemicals from data. 
        
        Parameters
        ----------
        IDs=None : Iterable[str] or str, optional
            IDs of chemicals to create. Defaults to all chemicals.
        
        """
        all_data = self.chemical_data
        data = all_data if IDs is None else {i: all_data[i] for i in as_IDs(IDs)}
        return chemicals_from_data(data, all_data)

        