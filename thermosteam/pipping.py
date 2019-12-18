# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:47:33 2018

This module includes classes and functions relating Stream objects.

@author: Yoel Cortes-Pena
"""
__all__ = ('MissingStream', 'Ins', 'Outs',
           'Sink', 'Source')

isa = isinstance

# %% Dummy Stream object

class MissingStreamType:
    """Create a MissingStream object that acts as a dummy in Ins and Outs objects until replaced by an actual Stream object."""
    _sink = None
    _source = None
    
    def __new__(cls):
        return MissingStream
    
    def __bool__(self):
        return False
    
    def __setattr__(self, key, value): pass
    
    def __getattr__(self, key):
        raise AttributeError(type(self).__name__)

    def __repr__(self):
        return f'MissingStream'
    
    def __str__(self):
        return 'missing stream'

MissingStream = object.__new__(MissingStreamType)

# %% Utilities

def resize(list, N_now, N_total, N_missing):
    list[N_now: N_total] = (MissingStream,) * N_missing

# %% List objects for input and output streams

class StreamSequence:
    """Create a StreamSequence object which serves as a sequence of streams for a Unit object."""
    __slots__ = ('_unit', '_size', '_streams')

    def __init__(self, unit, size):
        self._unit = unit #: Unit object where Ins object is attached
        self._size = size #: Number of streams in Ins object
        self._streams = [MissingStream] * size #: All input streams
        
    @property
    def size(self):
        return self._size
    
    def __len__(self):
        return self._size
    
    def __iter__(self):
        yield from self._streams
    
    def __getitem__(self, index):
        return self._streams[index]
    
    def __setitem__(self, index, item):
        unit = self._unit
        streams = self._streams
        if isa(index, int):
            s_old = streams[index]
            s_old._unit = None
            item._unit = unit
        elif isa(index, slice):
            N = len(item)
            size = self._size
            N_missing = size - N
            if N_missing < 0:
                raise IndexError(f"size of streams ({N}) cannot be bigger than "
                                 "size of '{type(self).__name__}' object ({size})")
            for s in streams[index]:
                if s._unit is unit:
                    s._unit = None
            for s in item:
                s._unit = unit
            streams[index] = item
            if N_missing:
                resize(streams, N, size, N_missing)
        else:
            raise TypeError(f"Only intergers and slices are valid indices for '{type(self).__name__}' objects")
            
    def __repr__(self):
        return repr(self._streams)


class Ins(StreamSequence):
    """Create a Ins object which serves as input streams for a Unit object."""
    @property
    def sink(self): return self._unit
           
    
class Outs(StreamSequence):
    """Create a Outs object which serves as output streams for a Unit object."""
    @property
    def source(self): return self._unit


# %% Sink and Source object for piping notation

class Sink:
    """Create a Sink object that connects a stream to a unit using piping notation:
    
    Parameters
    ----------
    stream : Stream
    index : int
        
    Examples
    --------
    First create a stream and a Mixer:
    
    .. code-block:: python
    
        >>> stream = Stream('s1')
        >>> unit = Mixer('M1')
    
    Sink objects are created using -pipe- notation:
        
    .. code-block:: python
    
        >>> stream-1
        <Sink: s1-1>
    
    Use pipe notation to create a sink and connect the stream:
    
    .. code-block:: python
    
        >>> stream-1-unit
        >>> M1.show()
        
        Mixer: M1
        ins...
        [0] Missing stream
        [1] s1
            phase: 'l', T: 298.15 K, P: 101325 Pa
            flow:  0
        outs...
        [0] d27
            phase: 'l', T: 298.15 K, P: 101325 Pa
            flow:  0
    
    """
    __slots__ = ('stream', 'index')
    def __init__(self, stream, index):
        self.stream = stream
        self.index = index

    # Forward pipping
    def __sub__(self, unit):
        unit.ins[self.index] = self.stream
        return unit
    
    # Backward pipping
    __pow__ = __sub__
    
    def __repr__(self):
        return '<' + type(self).__name__ + ': ' + self.stream.ID + '-' + str(self.index) + '>'


class Source:
    """Create a Source object that connects a stream to a unit using piping notation:
    
    Parameters
    ----------
    stream : Stream
    index : int
    
    Examples
    --------
    First create a stream and a Mixer:
    
    .. code-block:: python
    
        >>> stream = Stream('s1')
        >>> unit = Mixer('M1')
    
    Source objects are created using -pipe- notation:
        
    .. code-block:: python
    
        >>> 1**stream
        <Source: 1-s1>
    
    Use -pipe- notation to create a source and connect the stream:
    
    .. code-block:: python
    
        >>> unit**0**stream
        >>> M1.show()
        
        Mixer: M1
        ins...
        [0] Missing stream
        [1] Missing stream
        outs...
        [0] s1
            phase: 'l', T: 298.15 K, P: 101325 Pa
            flow:  0
    
    """
    __slots__ = ('stream', 'index')
    def __init__(self, stream, index):
        self.stream = stream
        self.index = index

    # Forward pipping
    def __rsub__(self, unit):
        unit.outs[self.index] = self.stream
        return unit
    
    # Backward pipping
    __rpow__ = __rsub__
    
    def __repr__(self):
        return '<' + type(self).__name__ + ': ' + str(self.index) + '-' + self.stream.ID + '>'




