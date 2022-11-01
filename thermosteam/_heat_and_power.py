# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .utils.decorators import registered_franchise
from ._stream import Stream

__all__ = ('Power',)

@registered_franchise(Stream)
class Heat:
    __slots__ = ('_ID', '_source', '_sink', 'heat')
    
    @property
    def source(self):
        return self._source
    
    @property
    def sink(self):
        return self._sink
    
    def isempty(self):
        return self.heat == 0.
    
    def empty(self):
        self.heat = 0.
    
    @classmethod
    def sum(cls, streams, ID=None):
        new = Power(ID)
        new.mix_from(streams)
        return new
    
    def mix_from(self, streams):
        isa = isinstance
        heat = 0.
        HeatOrPower = (Power, Heat)
        for i in streams:
            if isa(i, HeatOrPower):
                heat += i.heat
            else:
                raise TypeError(f"can only mix from Heat or Power objects, not '{type(i).__name__}'")
        self.heat = heat
    
    def __init__(self, ID=None, heat=0., source=None, sink=None):
        self._register(ID)
        self.heat = heat #: heat [kJ / hr]
        self._source = source
        self._sink = sink
        
    def __repr__(self):
        if self._source and self._sink:
            return f"{type(self).__name__}(heat={self.heat:.2f}, source={self.source}, sink={self.sink})"
        elif self._source:
            return f"{type(self).__name__}(heat={self.heat:.2f}, source={self.source})"
        elif self._sink:
            return f"{type(self).__name__}(heat={self.heat:.2f}, sink={self.sink})"
        else:
            return f"{type(self).__name__}(heat={self.heat:.2f})"


@registered_franchise(Stream)
class Power:
    __slots__ = ('_ID', '_source', '_sink', 'power')
    
    @property
    def heat(self):
        return self.power # Power can be directly converted to heat without any loses.
    
    @property
    def source(self):
        return self._source
    
    @property
    def sink(self):
        return self._sink
    
    def isempty(self):
        return self.power == 0.
    
    def empty(self):
        self.power = 0.
    
    @classmethod
    def sum(cls, streams, ID=None):
        new = Power(ID)
        new.mix_from(streams)
        return new
    
    def mix_from(self, streams):
        isa = isinstance
        power = 0.
        for i in streams:
            if isa(i, Power):
                power += i.power
            else:
                raise TypeError(f"can only mix from Power objects, not '{type(i).__name__}'")
        self.power = power
    
    def __init__(self, ID=None, power=0., source=None, sink=None):
        self._register(ID)
        self.power = power #: Electric power [kJ / hr]
        self._source = source
        self._sink = sink
        
    def __repr__(self):
        if self._source and self._sink:
            return f"{type(self).__name__}(power={self.power:.2f}, source={self.source}, sink={self.sink})"
        elif self._source:
            return f"{type(self).__name__}(power={self.power:.2f}, source={self.source})"
        elif self._sink:
            return f"{type(self).__name__}(power={self.power:.2f}, sink={self.sink})"
        else:
            return f"{type(self).__name__}(power={self.power:.2f})"
        