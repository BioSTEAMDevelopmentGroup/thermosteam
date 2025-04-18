# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2024, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo

__all__ = (
    'streams_from_units',
    'process_streams',
    'streams_from_path', 
    'feeds',
    'products',
    'inlets',
    'outlets',
    'filter_out_missing_streams',
    'feeds_from_units',
    'products_from_units',
    'get_inlet_origin',
    'get_streams_from_context_level',
    'get_fresh_process_water_streams',
    'FreeProductStreams',
)

def inlets(units):
    return set(sum([i._ins for i in units], []))

def outlets(units):
    return set(sum([i._outs for i in units], []))

def streams_from_units(units):
    return set(sum([i._ins + i._outs for i in units], []))

def streams_from_path(path):
    isa = isinstance
    streams = set()
    Unit = tmo.AbstractUnit
    for i in path:
        if isa(i, Unit):
            streams.update(i._ins + i._outs)
        else:
            streams.update(i.streams)
    return streams

def get_inlet_origin(inlet):
    source = inlet.source
    while source:
        if len(source.ins) == len(source.outs) == 1 and 'processing' not in source.line.lower():
            inlet = source.ins[0]
        elif source._interaction:
            index = source.outs.index(inlet)
            inlet = source.ins[index]
        else:
            break
        source = inlet.source
    return inlet
        

def process_streams(streams):
    return {i for i in streams if i._source and i._sink}

def feeds(streams):
    return [s for s in streams if not s._source]

def products(streams):
    return [s for s in streams if not s._sink]

def filter_out_missing_streams(streams):
    streams.intersection_update([i for i in streams if i])

def feeds_from_units(units):
    unit_set = set(units)
    return sum([[i for i in u._ins if i._source not in unit_set]
                 for u in units], [])

def products_from_units(units):
    unit_set = set(units)
    return sum([[i for i in u._outs if i._sink not in unit_set]
                 for u in units], [])

def get_streams_from_context_level(level=None):
    """
    Return streams created in given context level.
    
    Parameters
    ----------
    level : int, optional
        If given, only filter through streams created in the given context 
        level. For example, use:
        * 0: to filter within the outer-most context level.
        * -1: to filter within the current context level.
        * -2: to filter within the outer context level.
        
    
    Examples
    --------
    Get all waste streams created (as registerd in the flowsheet):
        
    >>> import biosteam as bst
    >>> bst.main_flowsheet.clear()
    >>> bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> waste = bst.Stream('waste')
    >>> mixer = bst.Mixer('mixer', outs=waste)
    >>> streams = bst.get_streams_from_context_level()
    >>> assert len(streams) == 1 and streams[0] == waste
    
    Get waste streams created within a context:
        
    >>> with bst.System('sys') as sys:
    ...     new_waste = bst.Stream('new_waste')
    ...     tank = bst.MixTank('tank', outs=new_waste)
    ...     streams = bst.get_streams_from_context_level(level=0)
    ...     assert len(streams) == 1 and streams[0] == new_waste
    
    """
    if level is None:
        streams = list(tmo.AbstractStream.registry)
    else:
        units = tmo.AbstractUnit.registry
        context_levels = units.context_levels
        N_levels = len(context_levels)
        if (level >= 0 and level >= N_levels or level < 0 and -level > N_levels):
            streams = list(tmo.AbstractStream.registry)
        else:
            units = context_levels[level]
            streams = [i for i in streams_from_units(units) if i]
    return streams

def get_fresh_process_water_streams(streams=None):
    """
    Return all feed water streams without a price.
    
    Parameters
    ----------
    streams : Iterable[Stream], optional
        Stream to filter through.
        
    Examples
    --------
    Get all fresh process water streams created (as registerd in the flowsheet):
        
    >>> import biosteam as bst
    >>> bst.main_flowsheet.clear()
    >>> bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> water = bst.Stream('water', Water=1)
    >>> mixer = bst.Mixer('mixer', ins=water)
    >>> streams = bst.get_fresh_process_water_streams()
    >>> assert len(streams) == 1 and streams[0] == water
    
    """    
    if streams is None: streams = tmo.AbstractStream.registry
    return [
        i for i in streams
        if not i.price and i.isfeed() and not i.sink._interaction and has_only_water(i)
    ]

def has_only_water(stream):
    chemicals = stream.available_chemicals
    return len(chemicals) == 1 and chemicals[0].CAS == '7732-18-5'

class FreeProductStreams:
    __slots__ = ('all_streams', 'cache', 'LHV_combustible', 'wastewater_keys', 'combustible_keys')
    LHV_combustible_default = 800.
    wastewater_keys_default = {'wastewater', 'brine'}
    combustible_keys_default = {'to_boiler',}
    
    def __init__(self, all_streams, LHV_combustible=None, wastewater_keys=None, combustible_keys=None):
        self.all_streams = all_streams
        self.cache = {}
        self.wastewater_keys = self.wastewater_keys_default if wastewater_keys is None else wastewater_keys
        self.combustible_keys = self.combustible_keys_default if combustible_keys is None else combustible_keys
        self.LHV_combustible = self.LHV_combustible_default if LHV_combustible is None else LHV_combustible
       
    @property
    def streams(self):
        cache = self.cache
        if 'streams' in cache:
            return cache['streams']
        else:
            cache['streams'] = streams = frozenset([
                i for i in self.all_streams if 
                not i.price and i.isproduct() and 'recycle' not in i._ID and not i.isempty()
                and not i.source._universal
            ])
        return streams
       
    @property
    def combustibles(self):
        cache = self.cache
        if 'cumbustibles' in cache: return cache['combustibles']
        LHV_combustible = self.LHV_combustible
        wastewater_keys = self.wastewater_keys
        combustible_keys = self.combustible_keys
        cache['combustibles'] = combustibles = frozenset([
            i for i in self.streams 
            if not any([j in i.ID.lower() for j in wastewater_keys])
            and (i.LHV / i.F_mass >= LHV_combustible)
        ] + [i for i in self.all_streams if any([j in i.ID.lower() for j in combustible_keys])])
        return combustibles
    
    @property
    def combustible_gases(self):
        cache = self.cache
        if 'combustible_gases' in cache: return cache['combustible_gases']
        combustibles = self.combustibles
        cache['combustible_gases'] = gas_combustibles = frozenset([i for i in combustibles if i.phase == 'g'])
        return gas_combustibles
    
    @property
    def combustible_slurries(self):
        cache = self.cache
        if 'combustible_slurries' in cache: return cache['combustible_slurries']
        combustible_gases = self.combustible_gases
        combustibles = self.combustibles
        cache['combustible_slurries'] = combustible_slurries = frozenset([i for i in combustibles if i not in combustible_gases])
        return combustible_slurries
    
    @property
    def noncombustibles(self):
        cache = self.cache
        if 'noncombustibles' in cache: return cache['noncombustibles']
        combustibles = self.combustibles
        cache['noncombustibles'] = noncombustibles = frozenset([i for i in self.streams if i not in combustibles])
        return noncombustibles
    
    @property
    def noncombustible_slurries(self):
        cache = self.cache
        if 'noncombustible_slurries' in cache: return cache['noncombustible_slurries']
        noncombustibles = self.noncombustibles
        cache['noncombustible_slurries'] = noncombustible_slurries = frozenset([i for i in noncombustibles if i.phase != 'g'])
        return noncombustible_slurries
        
    def __repr__(self):
        return f"{type(self).__name__}(all_streams={self.all_streams})"
