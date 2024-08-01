# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2024, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from __future__ import annotations
from warnings import warn
from typing import NamedTuple, Optional, Sequence, Callable, Tuple, Any, Iterable
import thermosteam as tmo
from flexsolve import IQ_interpolation
from .utils import AbstractMethod
from .utils.decorators import registered, thermo_user
from .base import display_asfunctor
from ._graphics import UnitGraphics, box_graphics
from thermosteam.utils import extended_signature
import numpy as np

int_types = (int, np.int32)

__all__ = ('stream_info', 'AbstractStream', 'AbstractUnit',
           'Connection', 'InletPipe', 'OutletPipe', 'AbstractMissingStream',
           'temporary_connection', 'TemporaryUnit',
           'BoundedNumericalSpecification', 'ProcessSpecification',
           'Network', 'mark_disjunction', 'unmark_disjunction',)

# %% Path utilities

def add_path_segment(start, end, path, ignored):
    fill_path_segment(start, path, end, set(), ignored)

def fill_path_segment(start, path, end, previous_units, ignored):
    if start is end: return path
    if start not in previous_units: 
        if start not in ignored: path.append(start)
        previous_units.add(start)
        success = False
        for outlet in start._outs:
            start = outlet._sink
            if not start: continue
            path_segment = fill_path_segment(start, [], end, previous_units, ignored)
            if path_segment is not None: 
                path.extend(path_segment)
                success = True
        if success: return path
    

# %% Streams

def stream_info(source, sink):
    """Return stream information header."""
    # First line
    if source is None:
        source = ''
    else:
        source = f' from {repr(source)}'
    if sink is None:
        sink = ''
    else:
        sink = f' to {repr(sink)}'
    return f"{source}{sink}"

@thermo_user
@registered(ticket_name='s')
class AbstractStream:
    __slots__ = ('_ID', '_source', '_sink', '_thermo', 'port')
    line = 'Stream'
    feed_priorities = {}
    price = F_mass = 0 # Required for filtering streams and sorting in network
    
    def __init__(self, ID=None, source=None, sink=None, thermo=None):
        self._register(ID)
        self._source = source
        self._sink = sink
        self._load_thermo(thermo)

    def get_feed_priority(self):
        if self.isfeed():
            return self.feed_priorities.get(self)
        else:
            raise RuntimeError(f"stream '{self}' is not a feed")

    def set_feed_priority(self, value):
        if self.isfeed():
            self.feed_priorities[self] = value
        else:
            raise RuntimeError(f"stream '{self}' is not a feed")

    @property
    def ID(self):
        return self._ID
    @property
    def source(self) -> AbstractUnit:
        """Outlet location."""
        return self._source
    @property
    def sink(self) -> AbstractUnit:
        """Inlet location."""
        return self._sink

    def isfeed(self):
        """Return whether stream has a sink but no source."""
        return bool(self._sink and not self._source)

    def isproduct(self):
        """Return whether stream has a source but no sink."""
        return bool(self._source and not self._sink)

    def disconnect_source(self):
        """Disconnect stream from source."""
        source = self._source
        if source: source.outs.remove(self)

    def disconnect_sink(self):
        """Disconnect stream from sink."""
        sink = self._sink
        if sink: sink.ins.remove(self)

    def disconnect(self):
        """Disconnect stream from unit."""
        self.disconnect_source()
        self.disconnect_sink()

    def __sub__(self, index):
        if isinstance(index, int):
            return InletPipe(self, index)
        elif hasattr(index, 'separate_out'):
            new = self.copy()
            new.separate_out(index)   
            return new
        else:
            return NotImplemented
    
    def __rsub__(self, index):
        if isinstance(index, int):
            return OutletPipe(self, index)
        else:
            return NotImplemented
        
    def __pow__(self, index):
        if isinstance(index, int):
            return InletPipe(self, index)
        else:
            return NotImplemented
    
    def __rpow__(self, index):
        if isinstance(index, int):
            return OutletPipe(self, index)
        else:
            return NotImplemented
    
    def get_connection(self, junction=None):
        if junction is None: junction = True
        source = self._source
        sink = self._sink
        if not junction and getattr(source, 'line', None) == 'Junction':
            stream = source._ins[0]
            source = stream._source
        else:
            stream = self
        try:
            source_index = source._outs.index(stream) if source else None
        except ValueError: # Auxiliary streams are not in inlets nor outlets  
            try:
                auxname = sink.ID
                source_index = source._auxout_index[auxname]
            except:
                source_index = -1
        if not junction and getattr(source, 'line', None) == 'Junction':
            stream = sink._outs[0]
            sink = stream._sink
        else:
            stream = self
        try:
            sink_index = sink._ins.index(stream) if sink else None
        except ValueError: # Auxiliary streams are not in inlets nor outlets
            try:
                auxname = source.ID
                sink_index = sink._auxin_index[auxname]
            except:
                sink_index = -1
        return Connection(source, source_index, self, sink_index, sink)

    def _basic_info(self):
        return (f"{type(self).__name__}: {self.ID or ''}"
                f"{stream_info(self._source, self._sink)}\n")

    def _source_info(self):
        return f"{source}-{source.outs.index(self)}" if (source:=self.source) else self.ID


class AbstractMissingStream:
    """
    Create a missing stream that acts as a dummy in inlets and outlets
    until replaced by an actual stream.
    """
    __slots__ = ('_source', '_sink')
    line = 'Stream'
    ID = 'missing stream'
    disconnect_source = AbstractStream.disconnect_source
    disconnect_sink = AbstractStream.disconnect_sink
    disconnect = AbstractStream.disconnect
    
    def __init__(self, source=None, sink=None):
        self._source = source
        self._sink = sink
    
    @property
    def source(self):
        return self._source
    @property
    def sink(self):
        return self._sink
    
    def _get_tooltip_string(self, format, full):
        return ''
    
    def __bool__(self):
        return False

    def __repr__(self):
        return f'<{type(self).__name__}>'

    def __str__(self):
        return self.ID

    def show(self):
        print(self._basic_info())
        

# %% Utilities for docking

def n_missing(ub, N):
    if ub < N: raise RuntimeError(f"size exceeds {ub}")
    return ub - N

DOCKING_WARNINGS = True

def ignore_docking_warnings(f):
    def g(*args, **kwargs):
        global DOCKING_WARNINGS
        warn = DOCKING_WARNINGS
        DOCKING_WARNINGS = False
        try:
            return f(*args, **kwargs)
        finally:
            DOCKING_WARNINGS = warn
    g.__name__ = f.__name__
    return g

class IgnoreDockingWarnings:
    """
    Ignore docking warnings within a context by using an IgnoreDockingWarnings object.
    
    """
    __slots__ = ('original_value',)
    
    def __enter__(self): 
        global DOCKING_WARNINGS
        self.original_value = DOCKING_WARNINGS
        DOCKING_WARNINGS = False
        return 
    
    def __exit__(self, type, exception, traceback):
        global DOCKING_WARNINGS
        DOCKING_WARNINGS = self.original_value
        if exception: raise exception
        
# %% List objects for inlet and outlet streams

class StreamSequence:
    """
    Abstract class for a sequence of streams for a unit.
    
    Abstract methods:
        * _dock(self, stream) -> AbstractStream
        * _redock(self, stream) -> AbstractStream
        * _undock(self) -> None
        * _load_missing_stream(self)
    
    """
    __slots__ = ('_size', '_streams', '_fixed_size')
    MissingStream = AbstractMissingStream
    Stream = AbstractStream
    
    def __init_subclass__(cls):
        cls.stream_types = (cls.Stream, cls.MissingStream)
    
    def __init__(self, size, streams, thermo, fixed_size, stacklevel):
        self._size = size
        self._fixed_size = fixed_size
        Stream = self.Stream
        stream_types = self.stream_types
        dock = self._dock
        redock = self._redock
        if streams == ():
            self._streams = [dock(Stream(thermo=thermo)) for i in range(size)]
        else:
            isa = isinstance
            if fixed_size:
                self._initialize_missing_streams()
                if streams is not None:
                    if isa(streams, str):
                        self._streams[0] = dock(Stream(streams, thermo=thermo))
                    elif isa(streams, stream_types):
                        self._streams[0] = redock(streams, stacklevel)
                    else:
                        N = len(streams)
                        n_missing(size, N) # Make sure size is not too big
                        self._streams[:N] = [redock(i, stacklevel+1) if isa(i, Stream)
                                             else dock(Stream(i, thermo=thermo)) for i in streams]
            elif streams is not None:
                if isa(streams, str):
                    self._streams = [dock(Stream(streams, thermo=thermo))]
                elif isinstance(streams, stream_types):
                    self._streams = [redock(streams, stacklevel)]
                else:
                    self._streams = loaded_streams = []
                    for i in streams:
                        if isa(i, Stream):
                            s = redock(i, stacklevel)
                        elif i is None:
                            s = self._create_missing_stream()
                        else:
                            s = Stream(i, thermo=thermo)
                            dock(s)
                        loaded_streams.append(s)
            else:
                self._initialize_missing_streams()
        
    def _create_missing_stream(self):
        return self.MissingStream(None, None)
        
    def _create_N_missing_streams(self, N):
        return [self._create_missing_stream() for i in range(N)]
    
    def _initialize_missing_streams(self):
        #: All input streams
        self._streams = self._create_N_missing_streams(self._size)
        
    def __add__(self, other):
        return self._streams + other
    def __radd__(self, other):
        return other + self._streams
    
    # DO NOT DELETE: These should be implemented by child class
    # def _dock(self, stream): return stream
    # def _redock(self, stream, stacklevel): return stream
    # def _undock(self, stream): pass

    def _set_streams(self, slice, streams, stacklevel):
        streams = [self._as_stream(i) for i in streams]
        all_streams = self._streams
        for stream in all_streams[slice]: self._undock(stream)
        all_streams[slice] = streams
        stacklevel += 1
        for stream in all_streams: self._redock(stream, stacklevel)
        if self._fixed_size:
            size = self._size
            N_streams = len(all_streams)
            if N_streams < size:
                N_missing = n_missing(size, N_streams)
                if N_missing:
                    all_streams[N_streams: size] = self._create_N_missing_streams(N_missing)
       
    def _as_stream(self, stream):
        if stream is None:
            stream = self._create_missing_stream()
        elif not isinstance(stream, (AbstractStream, AbstractMissingStream)):
            raise TypeError(
                f"'{type(self).__name__}' object can only contain "
                f"streams; not '{type(stream).__name__}' objects"
            )
        return stream
       
    @property
    def size(self):
        return self._streams.__len__()
    
    def __len__(self):
        return self._streams.__len__()
    
    def __bool__(self):
        return bool(self._streams)
    
    def _set_stream(self, int, stream, stacklevel):
        stream = self._as_stream(stream)
        self._undock(self._streams[int])
        self._streams[int] = self._redock(stream, stacklevel+1)
    
    def empty(self):
        for i in self._streams: self._undock(i)
        self._initialize_missing_streams()
    
    def insert(self, index, stream):
        if self._fixed_size: 
            raise RuntimeError(f"size of '{type(self).__name__}' object is fixed")
        self._undock(stream)
        self._dock(stream)
        self._streams.insert(index, stream)
    
    def append(self, stream):
        if self._fixed_size: 
            raise RuntimeError(f"size of '{type(self).__name__}' object is fixed")
        self._undock(stream)
        self._dock(stream)
        self._streams.append(stream)
    
    def extend(self, streams):
        if self._fixed_size: 
            raise RuntimeError(f"size of '{type(self).__name__}' object is fixed")
        for i in streams:
            self._undock(i)
            self._dock(i)
            self._streams.append(i)
    
    def replace(self, stream, other_stream):
        index = self.index(stream)
        self[index] = other_stream

    def index(self, stream):
        return self._streams.index(stream)

    def pop(self, index):
        streams = self._streams
        if self._fixed_size:
            stream = streams[index]
            missing_stream = self._create_missing_stream()
            self.replace(stream, missing_stream)
        else:
            stream = streams.pop(index)
        return stream

    def remove(self, stream):
        if self._fixed_size:
            self.replace(stream, self._create_missing_stream())
        else:
            self._undock(stream)
            self._streams.remove(stream)
        
    def clear(self):
        if self._fixed_size:
            self._initialize_missing_streams()
        else:
            for i in self._streams: self._undock(i)
            self._streams.clear()
    
    def reverse(self):
        self.streams.reverse()
    
    def __iter__(self):
        return iter(self._streams)
    
    def __getitem__(self, index):
        return self._streams[index]
    
    def __setitem__(self, index, item):
        isa = isinstance
        if isa(index, int):
            self._set_stream(index, item, 2)
        elif isa(index, slice):
            self._set_streams(index, item, 2)
        else:
            raise IndexError("Only intergers and slices are valid "
                             f"indices for '{type(self).__name__}' objects")
    
    def __repr__(self):
        return repr(self._streams)


class AbstractInlets(StreamSequence):
    """Create an Inlets object which serves as input streams for a unit."""
    __slots__ = ('_sink',)
    
    def __init__(self, sink, size, streams, thermo, fixed_size, stacklevel):
        self._sink = sink
        super().__init__(size, streams, thermo, fixed_size, stacklevel)
    
    @property
    def sink(self):
        return self._sink
    
    def _create_missing_stream(self):
        return self.MissingStream(None, self._sink)
    
    def _dock(self, stream): 
        stream._sink = self._sink
        return stream

    def _redock(self, stream, stacklevel): 
        sink = stream._sink
        if sink:
            ins = sink._ins
            if ins is not self:
                if stream in ins:
                    ins.remove(stream)
                    stream._sink = new_sink = self._sink
                    if (DOCKING_WARNINGS 
                        and sink.ID and new_sink.ID
                        and sink.ID != new_sink.ID):
                        warn(f"undocked inlet {stream} from {sink}; "
                             f"{stream} is now docked at {new_sink}", 
                             RuntimeWarning, stacklevel + 1)
                else:
                    stream._sink = self._sink
        else:
            stream._sink = self._sink
        return stream
    
    def _undock(self, stream): 
        stream._sink = None
    
        
class AbstractOutlets(StreamSequence):
    """Create an Outlets object which serves as output streams for units."""
    __slots__ = ('_source',)
    
    def __init__(self, source, size, streams, thermo, fixed_size, stacklevel):
        self._source = source
        super().__init__(size, streams, thermo, fixed_size, stacklevel)
    
    @property
    def source(self):
        return self._source
    
    def _create_missing_stream(self):
        return self.MissingStream(self._source, None)
    
    def _dock(self, stream): 
        stream._source = self._source
        return stream

    def _redock(self, stream, stacklevel): 
        source = stream._source
        if source:
            outs = source._outs
            if outs is not self:
                if stream in outs:
                    outs.remove(stream)
                    stream._source = new_source = self._source
                    if (DOCKING_WARNINGS 
                        and source.ID and new_source.ID
                        and source.ID != new_source.ID):
                        warn(f"undocked outlet {stream} from {source}; "
                             f"{stream} is now docked at {new_source}", 
                             RuntimeWarning, stacklevel + 1)
                else:
                    stream._source = self._source
        else:
            stream._source = self._source
        return stream
    
    def _undock(self, stream): 
        stream._source = None



# %%  Configuration bookkeeping

class Connection(NamedTuple):
    source: object
    source_index: int
    stream: object
    sink_index: int
    sink: object
    
    def reconnect(self):
        # Does not attempt to connect auxiliaries with owners (which should not be possible)
        source = self.source
        sink = self.sink
        if source:
            if not (sink and getattr(sink, '_owner', None) is source):
                source.outs[self.source_index] = self.stream
        else:
            self.stream.disconnect_source()
        if sink:
            if not (source and getattr(source, '_owner', None) is sink):
                sink.ins[self.sink_index] = self.stream
        else:
            self.stream.disconnect_sink()


# %% Piping notation

class InletPipe:
    """
    Create an InletPipe object that connects an stream to a unit using -pipe-
    notation:
    
    Parameters
    ----------
    stream : AbstractStream
    index : int
        
    Examples
    --------
    First create a stream and a Mixer:
    
    >>> from biosteam import Stream, Mixer, settings
    >>> settings.set_thermo(['Water'])
    >>> stream = Stream('s1')
    >>> unit = Mixer('M1', outs=('out'))
    
    Inlet pipes are created using -pipe- notation:
        
    >>> stream-1
    <InletPipe: s1-1>
    
    Use pipe notation to create a sink and connect the stream:
    
    >>> stream-1-unit # The last unit is returned to continue piping; just ignore this
    <Mixer: M1>
    >>> unit.show()
    Mixer: M1
    ins...
    [0] missing stream
    [1] s1
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow: 0
    outs...
    [0] out
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow: 0
    
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


class OutletPipe:
    """
    Create an OutletPipe object that connects an stream to a unit using -pipe-
    notation:
    
    Parameters
    ----------
    stream : AbstractStream
    index : int
    
    Examples
    --------
    First create a stream and a Mixer:
    
    >>> from biosteam import Stream, Mixer, settings
    >>> settings.set_thermo(['Water'])
    >>> stream = Stream('s1')
    >>> unit = Mixer('M1')
    
    Outlet pipes are created using -pipe- notation:
        
    >>> 1**stream
    <OutletPipe: 1-s1>
    
    Use -pipe- notation to create a source and connect the stream:
    
    >>> unit**0**stream # First unit is returned to continue backwards piping; just ignore this
    <Mixer: M1>
    >>> unit.show()
    Mixer: M1
    ins...
    [0] missing stream
    [1] missing stream
    outs...
    [0] s1
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow: 0
    
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


# %% System inlets and outlets

class InletPort:
    __slots__ = ('sink', 'index')
    
    @classmethod
    def from_inlet(cls, inlet):
        sink = inlet.sink
        if not sink: raise ValueError(f'stream {inlet} is not an inlet to any unit')
        index = sink.ins.index(inlet)
        return cls(sink, index)
    
    def __init__(self, sink, index):
        self.sink = sink
        self.index = index
      
    def __eq__(self, other):
        return self.sink is other.sink and self.index == other.index  
      
    def _sorting_key(self):
        return (self.sink.ID[1:], self.sink.ID, self.index)
        
    def get_stream(self):
        return self.sink.ins[self.index]
    
    def set_stream(self, stream, stacklevel):
        self.sink.ins._set_stream(self.index, stream, stacklevel+1)
    
    def __str__(self):
        return f"{self.index}-{self.sink}"
    
    def __repr__(self):
        return f"{type(self).__name__}({self.sink}, {self.index})"


class OutletPort:
    __slots__ = ('source', 'index')
    
    @classmethod
    def from_outlet(cls, outlet):
        source = outlet.source
        if not source: raise ValueError(f'stream {outlet} is not an outlet to any unit')
        index = source.outs.index(outlet)
        return cls(source, index)
    
    def __init__(self, source, index):
        self.source = source
        self.index = index
    
    def __eq__(self, other):
        return self.source is other.source and self.index == other.index
    
    def _sorting_key(self):
        return (self.source.ID[1:], self.source.ID[0], self.index)
    
    def get_stream(self):
        return self.source.outs[self.index]
    
    def set_stream(self, stream, stacklevel):
        self.source.outs._set_stream(self.index, stream, stacklevel+1)
    
    def __str__(self):
        return f"{self.source}-{self.index}"
    
    def __repr__(self):
        return f"{type(self).__name__}({self.source}, {self.index})"


class StreamPorts:
    __slots__ = ('_ports',)
    
    @classmethod
    def from_inlets(cls, inlets, sort=None):
        return cls([InletPort.from_inlet(i) for i in inlets], sort)
    
    @classmethod
    def from_outlets(cls, outlets, sort=None):
        return cls([OutletPort.from_outlet(i) for i in outlets], sort)
    
    def __init__(self, ports, sort=None):
        if sort: ports = sorted(ports, key=lambda x: x._sorting_key())
        self._ports = tuple(ports)    
    
    def __bool__(self):
        return bool(self._ports)
        
    def __iter__(self):
        for i in self._ports: yield i.get_stream()
    
    def __len__(self):
        return len(self._ports)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self._ports[index])
        else:
            return self._ports[index].get_stream()
    
    def __setitem__(self, index, item):
        isa = isinstance
        if isa(index, int):
            self._set_stream(index, item, 2)
        elif isa(index, slice):
            self._set_streams(index, item, 2)
        else:
            raise IndexError("Only intergers and slices are valid "
                            f"indices for '{type(self).__name__}' objects")
          
    def _set_stream(self, int, stream, stacklevel):
        self._ports[int].set_stream(stream, stacklevel+1)
    
    def _set_streams(self, slice, streams, stacklevel):
        ports = self._ports[slice]
        stacklevel += 1
        if len(streams) == len(ports):
            for i, j in zip(ports, streams): i.set_stream(j, stacklevel)
        else:
            raise IndexError("number of inlets must match the size of slice")
    
    def __repr__ (self):
        ports = ', '.join([str(i) for i in self._ports])
        return f"[{ports}]"


# %% Inlet and outlet representation

def repr_ins_and_outs(layout, ins, outs, T, P, flow, composition, N, IDs, sort, data):
    info = ''
    if ins:
        info += 'ins...\n'
        i = 0
        for stream in ins:
            unit = stream._source
            source_info = f'from  {type(unit).__name__}-{unit}' if unit else ''
            name = str(stream)
            if name == '-' and source_info: 
                name = ''
            else:
                name += '  '
            if stream and data:
                stream_info = stream._info(layout, T, P, flow, composition, N, IDs, sort)
                index = stream_info.index('\n')
                number = f'[{i}] '
                spaces = len(number) * ' '
                info += number + name + source_info + stream_info[index:].replace('\n', '\n' + spaces) + '\n'
            else:
                info += f'[{i}] {name}' + source_info + '\n'
            i += 1
    if outs:
        info += 'outs...\n'
        i = 0
        for stream in outs:
            unit = stream._sink
            sink_info = f'to  {type(unit).__name__}-{unit}' if unit else ''
            name = str(stream)
            if name == '-' and sink_info: 
                name = ''
            else:
                name += '  '
            if stream and data:
                stream_info = stream._info(layout, T, P, flow, composition, N, IDs, sort)
                index = stream_info.index('\n')
                number = f'[{i}] '
                spaces = len(number) * ' '
                info += number + name + sink_info + stream_info[index:].replace('\n', '\n' + spaces) + '\n'
            else:
                info += f'[{i}] {name}' + sink_info + '\n'
            i += 1
    return info[:-1]

# %% Auxiliary piping

def superposition_property(name):
    @property
    def p(self):
        return getattr(self.port.get_stream(), name)
    @p.setter
    def p(self, value):
        setattr(self.port.get_stream(), name, value)
        
    return p

def _superposition(cls, parent, port):
    excluded = set([*cls.__dict__, port, '_' + port, 'port'])
    for name in (*parent.__dict__, *AbstractStream.__slots__):
        if name in excluded: continue
        setattr(cls, name, superposition_property(name))
    return cls

def superposition(parent, port):
    return lambda cls: _superposition(cls, parent, port)


# %% Nodes
streams = Optional[Sequence[AbstractStream]] 

@thermo_user
@registered(ticket_name='U')
class AbstractUnit:
    #: Whether unit leverages information from other units within the system 
    #: regardless of whether or not they are directly connected.
    _universal = False
    
    #: Whether streams do not mix, but do interact.
    _interaction = False
    
    #: Class for initiallizing inlets.
    Inlets = AbstractInlets
    
    #: Class for initiallizing outlets.
    Outlets = AbstractOutlets
    
    #: Class for initiallizing streams.
    Stream = AbstractStream
    
    #: Class for initiallizing missing streams.
    MissingStream = AbstractMissingStream
    
    #: Initialize unit operation with key-word arguments.
    _init = AbstractMethod
    
    #: Run mass and energy balances and update outlet streams (without user-defined specifications).
    _run = AbstractMethod
    
    #: **class-attribute** Expected number of inlet streams. Defaults to 1.
    _N_ins: int = 1  
    
    #: **class-attribute** Expected number of outlet streams. Defaults to 1
    _N_outs: int = 1
    
    #: **class-attribute** Whether the number of streams in :attr:`~Unit.ins` is fixed.
    _ins_size_is_fixed: bool = True
    
    #: **class-attribute** Whether the number of streams in :attr:`~Unit.outs` is fixed.
    _outs_size_is_fixed: bool = True
    
    #: **class-attribute** Used for piping warnings.
    _stacklevel: int = 5
    
    #: **class-attribute** Name denoting the type of Unit class. Defaults to the class
    #: name of the first child class
    line: str = 'Unit'
    
    #: **class-attribute** Settings for diagram representation. Defaults to a 
    #: box with the same number of inlet and outlet edges as :attr:`~Unit._N_ins` 
    #: and :attr:`~Unit._N_outs`.
    _graphics: UnitGraphics = box_graphics
    
    #: **class-attribute** Whether to skip detailed simulation when inlet 
    #: streams are empty. If inlets are empty and this flag is True,
    #: detailed mass and energy balance, design, and costing algorithms are skipped
    #: and all outlet streams are emptied.
    _skip_simulation_when_inlets_are_empty = False
    
    #: **class-attribute** Name of attributes that are auxiliary units. These units
    #: will be accounted for in the purchase and installed equipment costs
    #: without having to add these costs in the :attr:`~Unit.baseline_purchase_costs` dictionary.
    #: Heat and power utilities are also automatically accounted for.
    auxiliary_unit_names: tuple[str, ...] = ()

    #: **class-attribute** Index for auxiliary inlets to parent unit for graphviz diagram settings.
    _auxin_index = {}

    #: **class-attribute** Index for auxiliary outlets to parent unit for graphviz diagram settings.
    _auxout_index = {}

    def __init_subclass__(cls):
        dct = cls.__dict__
        if '__init__' in dct and '_init' not in dct :
            init = dct['__init__']
            if hasattr(init, 'extension'): cls._init = init.extension
        elif dct.get('_init'):
            _init = dct['_init']
            cls.__init__ = extended_signature(cls.__init__, _init)
            cls.__init__.extension = _init
        if '__init__' in dct or '_init' in dct:
            init = dct['__init__']
            annotations = init.__annotations__
            for i in ('ins', 'outs'):
                if i not in annotations: annotations[i] = streams
            if '_stacklevel' not in dct: cls._stacklevel += 1
        if 'Stream' not in dct: return
        Stream = cls.Stream
        @superposition(Stream, 'sink')
        class SuperpositionInlet(Stream): 
            __slots__ = ()
            def __init__(self, port, sink=None):
                self.port = port
                self._sink = sink

        @superposition(Stream, 'source')
        class SuperpositionOutlet(Stream):
            __slots__ = ()
            def __init__(self, port, source=None):
                self.port = port
                self._source = source
        
        cls.SuperpositionInlet = SuperpositionInlet
        cls.SuperpositionOutlet = SuperpositionOutlet

    def __init__(self,
            ID: Optional[str]='',
            ins: streams=None,
            outs: streams=(),
            thermo: Optional[tmo.Thermo]=None,
            **kwargs
        ):
        self._isdynamic = False
        self._system = None
        self._register(ID)
        self._load_thermo(thermo)
    
        ### Initialize streams
        self.auxins = {} #: dict[int, stream] Auxiliary inlets by index.
        self.auxouts = {} #:  dict[int, stream] Auxiliary outlets by index.
        self._init_inlets(ins)
        self._init_outlets(outs)
    
        ### Initialize specification    
    
        #: All specification functions
        self._specifications: list[Callable] = []
        
        #: Whether to run mass and energy balance after calling
        #: specification functions
        self.run_after_specifications: bool = False 
        
        #: Whether to prioritize unit operation specification within recycle loop (if any).
        self.prioritize: bool = False
        
        #: Safety toggle to prevent infinite recursion
        self._active_specifications: set[ProcessSpecification] = set()
        
        self._init(**kwargs)
        
    def _init_inlets(self, ins):
        self._ins = self.Inlets(
            self, self._N_ins, ins, self._thermo, self._ins_size_is_fixed, self._stacklevel,
        )
        
    def _init_outlets(self, outs, **kwargs):
        self._outs = self.Outlets(
            self, self._N_outs, outs, self._thermo, self._outs_size_is_fixed, self._stacklevel,
        )
        
    @property
    def ins(self) -> Sequence[AbstractStream]:
        """List of all inlet streams."""
        return self._ins    
    @property
    def outs(self) -> Sequence[AbstractStream]:
        """List of all outlet streams."""
        return self._outs
        
    @property
    def auxiliary_units(self) -> list[AbstractUnit]:
        """Return list of all auxiliary units."""
        getfield = getattr
        isa = isinstance
        auxiliary_units = []
        for name in self.auxiliary_unit_names:
            unit = getfield(self, name, None)
            if unit is None: continue 
            if isa(unit, Iterable):
                auxiliary_units.extend(unit)
            else:
                auxiliary_units.append(unit)
        return auxiliary_units

    @property
    def nested_auxiliary_units(self) -> list[AbstractUnit]:
        """Return list of all auxiliary units, including nested ones."""
        getfield = getattr
        isa = isinstance
        auxiliary_units = []
        for name in self.auxiliary_unit_names:
            unit = getfield(self, name, None)
            if unit is None: continue 
            if isa(unit, Iterable):
                auxiliary_units.extend(unit)
                for u in unit:
                    if not isinstance(u, AbstractUnit): continue
                    for auxunit in u.auxiliary_units:
                        auxiliary_units.append(auxunit)
                        auxiliary_units.extend(auxunit.nested_auxiliary_units)
            else:
                auxiliary_units.append(unit)
                if not isinstance(unit, AbstractUnit): continue
                for auxunit in unit.auxiliary_units:
                    auxiliary_units.append(auxunit)
                    auxiliary_units.extend(auxunit.nested_auxiliary_units)
        return auxiliary_units

    def _diagram_auxiliary_units_with_names(self) -> list[tuple[str, AbstractUnit]]:
        """Return list of name - auxiliary unit pairs."""
        getfield = getattr
        isa = isinstance
        auxiliary_units = []
        names = (
            self.diagram_auxiliary_unit_names 
            if hasattr(self, 'diagram_auxiliary_unit_names')
            else self.auxiliary_unit_names
        )
        for name in names:
            unit = getfield(self, name, None)
            if unit is None: continue 
            if isa(unit, Iterable):
                for i, u in enumerate(unit):
                    auxiliary_units.append(
                        (f"{name}[{i}]", u)
                    )
            else:
                auxiliary_units.append(
                    (name, unit)
                )
        return auxiliary_units

    def get_auxiliary_units_with_names(self) -> list[tuple[str, AbstractUnit]]:
        """Return list of name - auxiliary unit pairs."""
        getfield = getattr
        isa = isinstance
        auxiliary_units = []
        for name in self.auxiliary_unit_names:
            unit = getfield(self, name, None)
            if unit is None: continue 
            if isa(unit, Iterable):
                for i, u in enumerate(unit):
                    auxiliary_units.append(
                        (f"{name}[{i}]", u)
                    )
            else:
                auxiliary_units.append(
                    (name, unit)
                )
        return auxiliary_units

    def _diagram_nested_auxiliary_units_with_names(self, depth=-1) -> list[AbstractUnit]:
        """Return list of all diagram auxiliary units, including nested ones."""
        auxiliary_units = []
        if depth: 
            depth -= 1
        else:
            return auxiliary_units
        for name, auxunit in self._diagram_auxiliary_units_with_names():
            if auxunit is None: continue 
            auxiliary_units.append((name, auxunit))
            if not isinstance(auxunit, AbstractUnit): continue
            auxiliary_units.extend(
                [('.'.join([name, i]), j)
                 for i, j in auxunit._diagram_nested_auxiliary_units_with_names(depth)]
            )
        return auxiliary_units

    def get_nested_auxiliary_units_with_names(self, depth=-1) -> list[AbstractUnit]:
        """Return list of all auxiliary units, including nested ones."""
        auxiliary_units = []
        if depth: 
            depth -= 1
        else:
            return auxiliary_units
        for name, auxunit in self.get_auxiliary_units_with_names():
            if auxunit is None: continue 
            auxiliary_units.append((name, auxunit))
            if not isinstance(auxunit, AbstractUnit): continue
            auxiliary_units.extend(
                [('.'.join([name, i]), j)
                 for i, j in auxunit.get_nested_auxiliary_units_with_names(depth)]
            )
        return auxiliary_units

    def _unit_auxins(self, N_streams, streams, thermo):
        if streams is None or streams == ():
            Stream = self.Stream
            return [self.auxin(Stream(None, thermo=thermo)) for i in range(N_streams)]
        elif isinstance(streams, (tmo.AbstractStream, tmo.AbstractMissingStream)) or streams.__class__ is str:
            return self.auxin(streams, thermo=thermo)
        else:
            return [self.auxin(i, thermo=thermo) for i in streams]

    def _unit_auxouts(self, N_streams, streams, thermo):
        if streams is None or streams == ():
            Stream = self.Stream
            return [self.auxout(Stream(None, thermo=thermo)) for i in range(N_streams)]
        elif isinstance(streams, (tmo.AbstractStream, tmo.AbstractMissingStream)) or streams.__class__ is str:
            return self.auxout(streams, thermo=thermo)
        else:
            return [self.auxout(i, thermo=thermo) for i in streams]
    
    def auxiliary(
            self, name, cls, ins=None, outs=(), thermo=None,
            **kwargs
        ):
        """
        Create and register an auxiliary unit operation. Inlet and outlet
        streams automatically become auxlets so that parent unit streams will
        not disconnect.

        """
        if thermo is None: thermo = self.thermo
        auxunit = cls.__new__(cls)
        stack = getattr(self, name, None)
        if isinstance(stack, list): 
            name = f"{name}[{len(stack)}]"
            stack.append(auxunit)
        else:
            setattr(self, name, auxunit)
        auxunit.owner = self # Avoids property package checks
        auxunit.__init__(
            '.' + name, 
            self._unit_auxins(cls._N_ins, ins, thermo), 
            self._unit_auxouts(cls._N_outs, outs, thermo),
            thermo, 
            **kwargs
        )
        return auxunit

    def auxlet(self, stream: AbstractStream, thermo=None):
        """
        Define auxiliary unit inlet or outlet. This method has two
        behaviors:

        * If the stream is not connected to this unit, define the Stream 
          object's source or sink to be this unit without actually connecting 
          it to this unit.

        * If the stream is already connected to this unit, return a superposition
          stream which can be connected to auxiliary units without being disconnected
          from this unit. 

        """
        Stream = self.Stream
        if thermo is None: thermo = self.thermo
        if stream is None: 
            stream = Stream(None, thermo=thermo)
            stream._sink = stream._source = self
        elif isinstance(stream, str): 
            stream = Stream('.' + stream, thermo=thermo)
            stream._source = stream._sink = self
        elif self is stream._source and stream in self._outs:
            port = OutletPort.from_outlet(stream)
            stream = self.SuperpositionOutlet(port)
            self.auxouts[port.index] = stream
        elif self is stream._sink and stream in self._ins:
            port = InletPort.from_inlet(stream)
            stream = self.SuperpositionInlet(port)
            self.auxins[port.index] = stream
        else:
            if stream._source is None: stream._source = self
            if stream._sink is None: stream._sink = self
        return stream

    def auxin(self, stream: AbstractStream, thermo=None):
        """
        Define auxiliary unit inlet. This method has two
        behaviors:

        * If the stream is not connected to this unit, define the Stream 
          object's source to be this unit without actually connecting it to this unit.

        * If the stream is already connected to this unit, return a superposition
          stream which can be connected to auxiliary units without being disconnected
          from this unit.

        """
        Stream = self.Stream
        if thermo is None: thermo = self.thermo
        if stream is None: 
            stream = Stream(None, thermo=thermo)
            stream._sink = stream._source = self
        elif isinstance(stream, str): 
            stream = Stream('.' + stream, thermo=thermo)
            stream._source = stream._sink = self
        elif self is stream._sink and stream in self._ins:
            port = InletPort.from_inlet(stream)
            stream = self.SuperpositionInlet(port)
            self.auxins[port.index] = stream
        else:
            if stream._source is None: stream._source = self
            if stream._sink is None: stream._sink = self
        return stream

    def auxout(self, stream: AbstractStream, thermo=None):
        """
        Define auxiliary unit outlet. This method has two
        behaviors:

        * If the stream is not connected to this unit, define the Stream 
          object's source or sink to be this unit without actually connecting 
          it to this unit.

        * If the stream is already connected to this unit, return a superposition
          stream which can be connected to auxiliary units without being disconnected
          from this unit. 

        """
        Stream = self.Stream
        if thermo is None: thermo = self.thermo
        if stream is None: 
            stream = Stream(None, thermo=thermo)
            stream._sink = stream._source = self
        elif isinstance(stream, str): 
            stream = Stream('.' + stream, thermo=thermo)
            stream._source = stream._sink = self
        elif self is stream._source and stream in self._outs:
            port = OutletPort.from_outlet(stream)
            stream = self.SuperpositionOutlet(port)
            self.auxouts[port.index] = stream
        else:
            if stream._source is None: stream._source = self
            if stream._sink is None: stream._sink = self
        return stream

    def _assembled_from_auxiliary_units(self):
        #: Serves for checking whether to include this unit in auxiliary diagrams.
        #: If all streams are in common, it must be assembled by auxiliary units.
        return not set([i.ID for i in self.ins + self.outs]).difference(
            sum([[i.ID for i in (i.ins + i.outs)] for i in self.auxiliary_units], [])
        )
    
    
    def get_node(self):
        """Return unit node attributes for graphviz."""
        if tmo.preferences.minimal_nodes:
            return self._graphics.get_minimal_node(self)
        else:
            node = self._graphics.get_node_tailored_to_unit(self)
            node['tooltip'] = self._get_tooltip_string()
            return node
    
    def add_specification(self, 
            f: Optional[Callable]=None, 
            run: Optional[bool]=None, 
            args: Optional[tuple]=(),
            impacted_units: Optional[tuple[AbstractUnit, ...]]=None,
            prioritize: Optional[bool]=None,
        ):
        """
        Add a specification.

        Parameters
        ----------
        f : 
            Specification function runned for mass and energy balance.
        run : 
            Whether to run the built-in mass and energy balance after 
            specifications. Defaults to False.
        args : 
            Arguments to pass to the specification function.
        impacted_units :
            Other units impacted by specification. The system will make sure to 
            run itermediate upstream units when simulating.
        prioritize :
            Whether to prioritize the unit operation within a recycle loop (if any).
            
        Examples
        --------
        :doc:`../tutorial/Process_specifications`

        See Also
        --------
        add_bounded_numerical_specification
        specifications
        run

        Notes
        -----
        This method also works as a decorator.

        """
        if not f: return lambda f: self.add_specification(f, run, args, impacted_units, prioritize)
        if not callable(f): raise ValueError('specification must be callable')
        self._specifications.append(ProcessSpecification(f, args, impacted_units))
        if run is not None: self.run_after_specifications = run
        if prioritize is not None: self.prioritize = prioritize
        return f
    
    def add_bounded_numerical_specification(self, f=None, *args, **kwargs):
        """
        Add a bounded numerical specification that solves x where f(x) = 0 using an 
        inverse quadratic interpolation solver.
        
        Parameters
        ----------
        f : Callable
            Objective function in the form of f(x, *args).
        x : float, optional
            Root guess.
        x0, x1 : float
            Root bracket. Solution must lie within x0 and x1.
        xtol : float, optional
            Solver stops when the root lies within xtol. Defaults to 0.
        ytol : float, optional 
            Solver stops when the f(x) lies within ytol of the root. Defaults to 5e-8.
        args : tuple, optional
            Arguments to pass to f.
        maxiter : 
            Maximum number of iterations. Defaults to 50.
        checkiter : bool, optional
            Whether to raise a Runtime error when tolerance could not be 
            satisfied before the maximum number of iterations. Defaults to True.
        checkroot : bool, optional
            Whether satisfying both tolerances, xtol and ytol, are required 
            for termination. Defaults to False.
        checkbounds : bool, optional
            Whether to raise a ValueError when in a bounded solver when the 
            root is not certain to lie within bounds (i.e. f(x0) * f(x1) > 0.).
            Defaults to True.
            
        Examples
        --------
        :doc:`../tutorial/Process_specifications`

        See Also
        --------
        add_specification
        specifications
        
        Notes
        -----
        This method also works as a decorator.

        """
        if not f: return lambda f: self.add_bounded_numerical_specification(f, *args, **kwargs)
        if not callable(f): raise ValueError('f must be callable')
        self._specifications.append(BoundedNumericalSpecification(f, *args, **kwargs))
        return f
    
    def run_phenomena(self):
        """
        Run mass and energy balance without converging phenomena. This method also runs specifications
        user defined specifications unless it is being run within a 
        specification (to avoid infinite loops). 
        
        See Also
        --------
        _run
        specifications
        add_specification
        add_bounded_numerical_specification
        
        """
        if self._skip_simulation_when_inlets_are_empty and all([i.isempty() for i in self._ins]):
            for i in self._outs: i.empty()
            return
        if hasattr(self, '_run_phenomena'): 
            self._run = self._run_phenomena
            try: self._run_with_specifications()
            finally: del self._run
        else:
            self._run_with_specifications()
    
    def run(self):
        """
        Run mass and energy balance. This method also runs specifications
        user defined specifications unless it is being run within a 
        specification (to avoid infinite loops). 
        
        See Also
        --------
        _run
        specifications
        add_specification
        add_bounded_numerical_specification
        
        """
        if self._skip_simulation_when_inlets_are_empty and all([i.isempty() for i in self._ins]):
            for i in self._outs: i.empty()
            return
        self._run_with_specifications()
        
    def _run_with_specifications(self):
        specifications = self._specifications
        if specifications:
            active_specifications = self._active_specifications
            if len(active_specifications) == len(specifications):
                self._run()
            else:
                for ps in specifications: 
                    if ps in active_specifications: continue
                    active_specifications.add(ps)
                    try: ps()
                    finally: active_specifications.remove(ps)
                if self.run_after_specifications: self._run()
        else:
            self._run()
    
    def path_from(self, units, inclusive=False, system=None):
        """
        Return a tuple of units and systems starting from `units` until this one 
        (not inclusive by default).
        
        """
        units = (units,) if isinstance(units, AbstractUnit) else tuple(units)
        if system: # Includes recycle loops
            path = system.path_section(units, (self,))
        else: # Path outside system, so recycle loops may not converge (and don't have to)
            path = []
            added_units = set()
            upstream_units = self.get_upstream_units()
            for unit in units:
                if unit in upstream_units: add_path_segment(unit, self, path, added_units)
            if inclusive and unit not in added_units: path.append(unit)
        return path        
    
    def path_until(self, units, inclusive=False, system=None):
        """
        Return a tuple of units and systems starting from this one until the end
        units (not inclusive by default).
        
        """
        units = (units,) if isinstance(units, AbstractUnit) else tuple(units)
        if system: # Includes recycle loops
            path = system.path_section((self,), units, inclusive)
        else: # Path outside system, so recycle loops may not converge (and don't have to)
            path = []
            added_units = set()
            downstream_units = self.get_downstream_units()
            for unit in units:
                if unit in downstream_units: add_path_segment(self, unit, path, added_units)
            if inclusive and unit not in added_units: path.append(unit)
        return path
    
    def run_until(self, units, inclusive=False, system=None):
        """
        Run all units and converge all systems starting from this one until the end units
        (not inclusive by default).
        
        See Also
        --------
        path_until
        
        """
        isa = isinstance
        for i in self.path_until(units, inclusive, system): 
            if isa(i, AbstractUnit): i.run()
            else: i.converge() # Must be a system
        
        
    @property
    def system(self):
        return self._system
    
    @ignore_docking_warnings
    def disconnect(self, discard=False, inlets=None, outlets=None, join_ends=False):
        ins = self._ins
        outs = self._outs
        if inlets is None: 
            inlets = [i for i in ins if i.source]
            ins[:] = ()
        else:
            for i in inlets: ins[ins.index(i) if isinstance(i, AbstractStream) else i] = None
        if outlets is None: 
            outlets = [i for i in outs if i.sink]
            outs[:] = ()
        else:
           for o in outlets: outs[ins.index(o) if isinstance(o, AbstractStream) else o] = None
        if join_ends:
            if len(inlets) != len(outlets):
                raise ValueError("number of inlets must match number of outlets to join ends")
            for inlet, outlet in zip(inlets, outlets):
                outlet.sink.ins.replace(outlet, inlet)
        if discard: self.registry.discard(self)

    @ignore_docking_warnings
    def insert(self, stream: AbstractStream, inlet: int|AbstractStream=None, outlet: int|AbstractStream=None):
        """
        Insert unit between two units at a given stream connection.
        
        Examples
        --------
        >>> from biosteam import *
        >>> settings.set_thermo(['Water'], cache=True)
        >>> feed = Stream('feed')
        >>> other_feed = Stream('other_feed')
        >>> P1 = Pump('P1', feed, 'pump_outlet')
        >>> H1 = HXutility('H1', P1-0, T=310)
        >>> M1 = Mixer('M1', other_feed, 'mixer_outlet')
        >>> M1.insert(P1-0)
        >>> M1.show()
        Mixer: M1
        ins...
        [0] other_feed
            phase: 'l', T: 298.15 K, P: 101325 Pa
            flow: 0
        [1] pump_outlet  from  Pump-P1
            phase: 'l', T: 298.15 K, P: 101325 Pa
            flow: 0
        outs...
        [0] mixer_outlet  to  HXutility-H1
            phase: 'l', T: 298.15 K, P: 101325 Pa
            flow: 0
        
        """
        source = stream.source
        sink = stream.sink
        added_unit = False
        if outlet is None:
            if self._outs_size_is_fixed:
                if self._N_outs == 1:
                    sink.ins.replace(stream, self.outs[0])
                else:
                    raise ValueError("undefined outlet; must pass outlet when outlets are fixed and multiple are available")
            else:
                self.outs.append(stream)
                added_unit = True
        else:
            if isinstance(outlet, AbstractStream):
                if outlet.source is not self:
                    raise ValueError("source of given outlet must be this object")
            else:
                outlet = self.outs[outlet]
            sink.ins.replace(stream, outlet)
        if inlet is None:
            if self._ins_size_is_fixed or added_unit:
                if self._N_ins == 1:
                    source.outs.replace(stream, self.ins[0])
                else:
                    raise ValueError("undefined inlet; must pass inlet when inlets are fixed and multiple are available")
            else:
                self.ins.append(stream)
        else:
            if isinstance(inlet, AbstractStream):
                if inlet.sink is not self:
                    raise ValueError("sink of given inlet must be this object")
            else:
                inlet = self.outs[inlet]
            source.outs.replace(stream, inlet)
    
    @ignore_docking_warnings
    def take_place_of(self, other, discard=False):
        """Replace inlets and outlets from this unit with that of 
        another unit."""
        self._ins[:] = other.ins
        self._outs[:] = other.outs
        if discard: self.registry.discard(other)
    
    @ignore_docking_warnings
    def replace_with(self, other=None, discard=False):
        """Replace inlets and outlets from another unit with this unit."""
        if other is None:
            ins = self._ins
            outs = self._outs
            for inlet, outlet in zip(tuple(ins), tuple(outs)):
                source = inlet.source
                if source:
                    source.outs.replace(inlet, outlet)
                else:
                    sink = outlet.sink
                    if sink: sink.ins.replace(outlet, inlet)
            ins.empty()
            outs.empty()
        else:
            other.ins[:] = self._ins
            other.outs[:] = self._outs
        if discard: self.registry.discard(self)
                    
    # Forward pipping
    def __sub__(self, other):
        """Source streams."""
        isa = isinstance
        if hasattr(other, '_ins'):
            other._ins[:] = self._outs
            return other
        elif isa(other, int_types):
            return self._outs[other]
        elif isa(other, AbstractStream):
            self._outs[:] = (other,)
            return self
        elif isa(other, (tuple, list, np.ndarray)):
            if all([isa(i, int_types) for i in other]):
                outs = self._outs
                return [outs[i] for i in other]
            else:
                self._outs[:] = other
                return self
        else:
            return other.__rsub__(self)

    def __rsub__(self, other):
        """Sink streams."""
        isa = isinstance
        if isa(other, int_types):
            return self._ins[other]
        elif isa(other, AbstractStream):
            self._ins[:] = (other,)
            return self
        elif isa(other, (tuple, list, np.ndarray)):
            if all([isa(i, int_types) for i in other]):
                ins = self._ins
                return [ins[i] for i in other]
            else:
                self._ins[:] = other
                return self
        else:
            raise ValueError(f"cannot pipe '{type(other).__name__}' object")

    
    def _add_upstream_neighbors_to_set(self, set, ends, universal):
        """Add upsteam neighboring units to set."""
        for s in self._ins:
            u = s._source
            if u and (universal or not u._universal) and not (ends and s in ends):
                set.add(u)

    def _add_downstream_neighbors_to_set(self, set, ends, universal):
        """Add downstream neighboring units to set."""
        for s in self._outs:
            u = s._sink
            if u and (universal or not u._universal) and not (ends and s in ends):
                set.add(u)

    def get_downstream_units(self, ends=None, universal=True):
        """Return a set of all units downstream."""
        downstream_units = set()
        outer_periphery = set()
        self._add_downstream_neighbors_to_set(outer_periphery, ends, universal)
        inner_periphery = None
        old_length = -1
        new_length = 0
        while new_length != old_length:
            old_length = new_length
            inner_periphery = outer_periphery
            downstream_units.update(inner_periphery)
            outer_periphery = set()
            for unit in inner_periphery:
                unit._add_downstream_neighbors_to_set(outer_periphery, ends, universal)
            new_length = len(downstream_units)
        return downstream_units
    
    def get_upstream_units(self, ends=None, universal=True):
        """Return a set of all units upstream."""
        upstream_units = set()
        outer_periphery = set()
        self._add_upstream_neighbors_to_set(outer_periphery, ends, universal)
        inner_periphery = None
        old_length = -1
        new_length = 0
        while new_length != old_length:
            old_length = new_length
            inner_periphery = outer_periphery
            upstream_units.update(inner_periphery)
            outer_periphery = set()
            for unit in inner_periphery:
                unit._add_upstream_neighbors_to_set(outer_periphery, ends, universal)
            new_length = len(upstream_units)
        return upstream_units
    
    def neighborhood(self, 
            radius: Optional[int]=1, 
            upstream: Optional[bool]=True,
            downstream: Optional[bool]=True, 
            ends: Optional[AbstractStream]=None, 
            universal: Optional[bool]=None
        ):
        """
        Return a set of all neighboring units within given radius.
        
        Parameters
        ----------
        radius : 
            Maximum number streams between neighbors.
        downstream : 
            Whether to include downstream units.
        upstream : 
            Whether to include upstream units.
        ends :
            Streams that mark the end of the neighborhood.
        universal :
            Whether to include universal units.
        
        """
        radius -= 1
        neighborhood = set()
        if radius < 0: return neighborhood
        if upstream:self._add_upstream_neighbors_to_set(neighborhood, ends, universal)
        if downstream: self._add_downstream_neighbors_to_set(neighborhood, ends, universal)
        direct_neighborhood = neighborhood
        for i in range(radius):
            neighbors = set()
            for unit in direct_neighborhood:
                if upstream: unit._add_upstream_neighbors_to_set(neighbors, ends, universal)
                if downstream: unit._add_downstream_neighbors_to_set(neighbors, ends, universal)
            if neighbors == direct_neighborhood: break
            direct_neighborhood = neighbors
            neighborhood.update(direct_neighborhood)
        return neighborhood

    def diagram(self, radius: Optional[int]=0, upstream: Optional[bool]=True,
                downstream: Optional[bool]=True, file: Optional[str]=None, 
                format: Optional[str]=None, display: Optional[bool]=True,
                auxiliaries: Optional[bool]=-1,
                **graph_attrs):
        """
        Display a `Graphviz <https://pypi.org/project/graphviz/>`__ diagram
        of the unit and all neighboring units within given radius.
        
        Parameters
        ----------
        radius : 
            Maximum number streams between neighbors.
        downstream : 
            Whether to show downstream operations.
        upstream : 
            Whether to show upstream operations.
        file : 
            Must be one of the following:
            
            * [str] File name to save diagram.
            * [None] Display diagram in console.
            
        format : 
            Format of file.
        display : 
            Whether to display diagram in console or to return the graphviz 
            object.
        auxiliaries:
            Depth of auxiliary units to display.
        
        """
        import biosteam as bst
        if radius > 0:
            units = self.neighborhood(radius, upstream, downstream)
            units.add(self)
        else:
            units = [self]
        return bst.System(None, units).diagram(
            format=format, auxiliaries=auxiliaries, display=display, 
            file=file, title='', **graph_attrs
        )
    
    # Backwards pipping
    __pow__ = __sub__
    __rpow__ = __rsub__
    
    # Representation
    def _info(self, layout, T, P, flow, composition, N, IDs, sort, data):
        """Information on unit."""
        if self.ID:
            info = f'{type(self).__name__}: {self.ID}\n'
        else:
            info = f'{type(self).__name__}\n'
        ins = [(i.materialize() if hasattr(i, 'materialize') else i) for i in self.ins]
        outs = [(i.materialize() if hasattr(i, 'materialize') else i) for i in self.outs]
        return info + repr_ins_and_outs(layout, ins, outs, T, P, flow, composition, N, IDs, sort, data)

    def show(self, layout=None, T=None, P=None, flow=None, composition=None, N=None, IDs=None, sort=None, data=True):
        """Prints information on unit."""
        print(self._info(layout, T, P, flow, composition, N, IDs, sort, data))

    def _ipython_display_(self):
        import biosteam as bst
        if bst.preferences.autodisplay: self.diagram()
        self.show()


# %% Process specification

temporary_units_dump = []

def temporary_connection(source, sink):
    upstream = source.outs[0]
    downstream = sink.ins[0]
    temporary_stream = AbstractStream(None)
    if isinstance(upstream.sink, TemporarySource):
        upstream.sink.outs.append(temporary_stream)
    else:
        stream = AbstractStream(None)
        old_connection = upstream.get_connection()
        sink = upstream.sink
        if sink: upstream.sink.ins.replace(upstream, stream)
        TemporarySource(upstream, [stream, temporary_stream], old_connection)
    if isinstance(downstream.source, TemporarySink):
        downstream.source.ins.append(temporary_stream)
    else:
        stream = AbstractStream(None)
        old_connection = downstream.get_connection()
        downstream.sink.ins.replace(downstream, stream)
        TemporarySink([downstream, temporary_stream], stream, old_connection)


class TemporaryUnit:
    __slots__ = ('ID', '_ID', 'ins', 'outs', '_ins', '_outs', 'old_connection')
    _universal = False
    _interaction = False
    auxiliary_units = ()
    def __init__(self, ins, outs, old_connection):
        temporary_units_dump.append(self)
        self.ID = self._ID = 'TU' + str(len(temporary_units_dump))
        self.old_connection = old_connection
        self.ins = self._ins = AbstractInlets(
            self, self._N_ins, ins, None, self._ins_size_is_fixed, 5
        )
        self.outs = self._outs = AbstractOutlets(
            self, self._N_outs, outs, None, self._outs_size_is_fixed, 5
        )
        
    neighborhood = AbstractUnit.neighborhood
    get_downstream_units = AbstractUnit.get_downstream_units
    get_upstream_units = AbstractUnit.get_upstream_units
    _add_upstream_neighbors_to_set = AbstractUnit._add_upstream_neighbors_to_set
    _add_downstream_neighbors_to_set = AbstractUnit._add_downstream_neighbors_to_set
    __repr__ = AbstractUnit.__repr__
    __str__ = AbstractUnit.__str__
        

class TemporarySource(TemporaryUnit):
    __slots__ = ()
    _N_ins = 1
    _N_outs = 2
    _ins_size_is_fixed = True
    _outs_size_is_fixed = False
    

class TemporarySink(TemporaryUnit):
    __slots__ = ()
    _N_ins = 2
    _N_outs = 1
    _ins_size_is_fixed = False
    _outs_size_is_fixed = True


class ProcessSpecification:
    __slots__ = ('f', 'args', 'impacted_units', 'path')
    
    def __init__(self, f, args, impacted_units):
        self.f = f
        self.args = args
        if impacted_units: 
            self.impacted_units = tuple(impacted_units)
        else:
            self.impacted_units = self.path = ()
        
    def __call__(self):
        self.f(*self.args)
        isa = isinstance
        for i in self.path: 
            if isa(i, AbstractUnit): i.run()
            else: i.converge() # Must be a system
            
    def compile_path(self, unit):
        self.path = unit.path_from(self.impacted_units, system=unit._system) if self.impacted_units else ()
        
    def create_temporary_connections(self, unit):
        # Temporary connections are created first than the path because 
        # temporary connections may change the system configuration 
        # such that the path is incorrect.
        impacted_units = self.impacted_units
        if impacted_units:
            upstream_units = unit.get_upstream_units()
            for other in impacted_units:
                if other not in upstream_units: temporary_connection(unit, other)
            
    def __repr__(self):
        return f"{type(self).__name__}(f={display_asfunctor(self.f)}, args={self.args}, impacted_units={self.impacted_units})"


class BoundedNumericalSpecification:
    __slots__ = (
        'f', 'x0', 'x1', 'y0', 'y1', 'x', 'xtol', 'ytol', 'args', 
        'maxiter', 'checkroot', 'checkiter', 'checkbounds', 'x_last',
    )
    
    def __init__(self, 
            f: Callable,
            x0: float, 
            x1: float, 
            y0: Optional[float]=None, 
            y1: Optional[float]=None, 
            x: Optional[float]=None,
            xtol: float=0.,
            ytol: float=5e-8,
            args: Tuple[Any, ...]=(), 
            maxiter: int=50,
            checkroot: bool=False, 
            checkiter: bool=True, 
            checkbounds: bool=True
        ):
        self.f = f
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.x = x
        self.xtol = xtol
        self.ytol = ytol
        self.args = args
        self.maxiter = maxiter
        self.checkroot = checkroot
        self.checkiter = checkiter
        self.checkbounds = checkbounds
        self.x_last = None
        
    def __call__(self):
        self.x = IQ_interpolation(
            self.f, self.x0, self.x1, self.y0, self.y1, self.x, self.xtol, self.ytol, 
            self.args, self.maxiter, self.checkroot, self.checkiter, self.checkbounds,
        )
        return self.x
    
    def compile_path(self, unit): pass
    
    def create_temporary_connections(self, unit): pass


# %% Special network configuration

disjunctions = []

def mark_disjunction(stream):
    """Mark stream as deterministic after a linear flowsheet simulation.
    In other words, the stream flow rates and thermal condition will not change
    when solving recycle loops. This will prevent the system from
    forming recycle loops about this stream."""
    port = OutletPort.from_outlet(stream)
    if port not in disjunctions:
        disjunctions.append(port)

def unmark_disjunction(stream):
    port = OutletPort.from_outlet(stream)
    if port in disjunctions:
        disjunctions.remove(port)


def get_recycle_sink(recycle):
    if hasattr(recycle, 'sink'):
        return recycle.sink
    else:
        for i in recycle: return i.sink

def sort_feeds_big_to_small(feeds):
    if feeds:
        feed_priorities = tmo.AbstractStream.feed_priorities
        def feed_priority(feed):
            if feed in feed_priorities:
                return feed_priorities[feed]
            elif feed:
                try:
                    return 1. - feed.F_mass / F_mass_max if F_mass_max else 1.
                except:
                    return 2
            else:
                return 2.
        F_mass_max = max([i.F_mass for i in feeds])
        feeds.sort(key=feed_priority)

# %% Path tools

class PathSource:
    
    __slots__ = ('source', 'units')
    
    def __init__(self, source, ends=None):
        self.source = source
        if isinstance(source, Network):
            self.units = units = set()
            for i in source.units: units.update(i.get_downstream_units(ends=ends, universal=False))
        else:
            self.units = units = source.get_downstream_units(ends=ends, universal=False)
            
    def downstream_from(self, other):
        source = self.source
        if isinstance(source, Network):
            return self is not other and any([i in other.units for i in source.units])
        else:
            return source in other.units
        
    def __repr__(self):
        return f"{type(self).__name__}({str(self.source)})"
        

def find_linear_and_cyclic_paths_with_recycle(feed, ends, units):
    paths_with_recycle, linear_paths = find_paths_with_and_without_recycle(
        feed, ends, units
    )
    cyclic_paths_with_recycle = []
    for path_with_recycle in paths_with_recycle:
        cyclic_path_with_recycle = path_with_recycle_to_cyclic_path_with_recycle(path_with_recycle)
        cyclic_paths_with_recycle.append(cyclic_path_with_recycle)
    cyclic_paths_with_recycle.sort(key=lambda x: -len(x[0]))
    return simplified_linear_paths(linear_paths), cyclic_paths_with_recycle

def find_paths_with_and_without_recycle(feed, ends, units):
    paths_without_recycle  = []
    paths_with_recycle = []
    fill_path(feed, [], paths_with_recycle, paths_without_recycle, ends, units)
    return paths_with_recycle, paths_without_recycle

def fill_path(feed, path, paths_with_recycle,
              paths_without_recycle,
              ends, units):
    unit = feed.sink
    if not unit or unit._universal or unit not in units:
        paths_without_recycle.append(path)
    elif unit in path: 
        if len(unit.outs) == 1 and unit.outs[0] in ends: 
            paths_without_recycle.append(path)
        else:
            ends.add(feed)
            path_with_recycle = path, feed
            paths_with_recycle.append(path_with_recycle)
    elif feed in ends:
        paths_without_recycle.append(path)
    else:
        path.append(unit)
        outlets = unit._outs
        if outlets:
            first_outlet, *other_outlets = outlets
            for outlet in other_outlets:
                new_path = path.copy()
                fill_path(outlet, new_path,
                          paths_with_recycle,
                          paths_without_recycle,
                          ends, units)
            fill_path(first_outlet, path,
                      paths_with_recycle,
                      paths_without_recycle,
                      ends, units)

def path_with_recycle_to_cyclic_path_with_recycle(path_with_recycle):
    path, recycle = path_with_recycle
    unit = recycle.sink
    recycle_index = path.index(unit)
    return (path[recycle_index:], recycle)

def simplified_linear_paths(linear_paths):
    if not linear_paths: return linear_paths
    linear_paths.sort(key=len)
    units, *unit_sets = [set(i) for i in linear_paths]
    for i in unit_sets: units.update(i)
    simplified_paths = []
    for i, path in enumerate(linear_paths):
        simplify_linear_path(path, unit_sets[i:])
        if path:
            add_back_ends(path, units)
            simplified_paths.append(path)
    simplified_paths.reverse()
    return simplified_paths
    
def simplify_linear_path(path, unit_sets):
    if path and unit_sets:
        for unit in path.copy():
            for unit_set in unit_sets:
                if unit in unit_set:
                    path.remove(unit)
                    break

def add_back_ends(path, units):
    for outlet in path[-1]._outs:
        sink = outlet._sink 
        if sink in units: 
            path.append(sink)

def nested_network_units(path):
    units = set()
    isa = isinstance
    for i in path:
        if isa(i, Network):
            units.update(i.units)
        else:
            units.add(i)
    return units

# %% Network

class Network:
    """
    Create a Network object that defines a network of unit operations.
    
    Parameters
    ----------
    path : Iterable[:class:`~biosteam.Unit` or :class:`~biosteam.Network`]
        A path of unit operations and subnetworks.
    recycle : :class:`~thermosteam.Stream` or set[:class:`~thermosteam.Stream`]
        A recycle stream(s), if any.
    
    Examples
    --------
    Create a network representing two nested recycle loops:
        
    >>> from biosteam import (
    ...     main_flowsheet as f,
    ...     Pump, Mixer, Splitter,
    ...     Stream, settings
    ... )
    >>> f.set_flowsheet('two_nested_recycle_loops')
    >>> settings.set_thermo(['Water'], cache=True)
    >>> feedstock = Stream('feedstock', Water=1000)
    >>> water = Stream('water', Water=10)
    >>> recycle = Stream('recycle')
    >>> inner_recycle = Stream('inner_recycle')
    >>> product = Stream('product')
    >>> inner_water = Stream('inner_water', Water=10)
    >>> P1 = Pump('P1', feedstock)
    >>> P2 = Pump('P2', water)
    >>> P3 = Pump('P3', inner_water)
    >>> M1 = Mixer('M1', [P1-0, P2-0, recycle])
    >>> M2 = Mixer('M2', [M1-0, P3-0, inner_recycle])
    >>> S2 = Splitter('S2', M2-0, ['', inner_recycle], split=0.5)
    >>> S1 = Splitter('S1', S2-0, [product, recycle], split=0.5)
    >>> network = Network(
    ... [P1,
    ...  P2,
    ...  P3,
    ...  Network(
    ...     [M1,
    ...      Network(
    ...          [M2,
    ...           S2],
    ...          recycle=inner_recycle),
    ...      S1],
    ...     recycle=recycle)])
    >>> network.show()
    Network(
        [P1,
         P2,
         P3,
         Network(
            [M1,
             Network(
                [M2,
                 S2],
                recycle=S2-1),
             S1],
            recycle=S1-1)])
    
    """
    
    __slots__ = ('path', 'units', 'recycle', 'recycle_sink')
    
    def __init__(self, path, recycle=None):
        self.path = path
        self.recycle = recycle
        self.recycle_sink = get_recycle_sink(recycle) if recycle else None
        try: self.units = set(path)
        except: self.units = nested_network_units(path)
    
    def __eq__(self, other):
        return isinstance(other, Network) and self.path == other.path and self.recycle == other.recycle
    
    def get_all_recycles(self, all_recycles=None):
        if all_recycles is None:
            all_recycles = set()
        recycle = self.recycle
        if recycle:
            if hasattr(recycle, 'sink'):
                all_recycles.add(recycle)
            else:
                all_recycles.update(recycle)
        for i in self.path:
            if isinstance(i, Network): i.get_all_recycles(all_recycles)
        return all_recycles
    
    def sort_without_recycle(self, ends):
        if self.recycle: return
        path = tuple(self.path)
        
        def sort(network):
            subpath = tuple(network.path)
            for o, j in enumerate(subpath):
                if isinstance(j, Network):
                    added = sort(j)
                    if added: return True
                elif j in sources: 
                    self.path.remove(u)
                    network.path.insert(o + 1, u)
                    return True
                    
        for n, u in enumerate(path):
            if isinstance(u, Network): continue
            sources = set([i.source for i in u.ins if i.source and i not in ends])
            subpath = path[n+1:]
            for m, i in enumerate(subpath):
                if isinstance(i, Network):
                    added = sort(i)
                    if added: break
                elif i in sources: 
                    self.path.remove(u)
                    self.path.insert(n + m, u)
                    break
    
    def sort(self, ends):
        isa = isinstance
        for i in self.path: 
            if isa(i, Network): i.sort(ends)
        path_sources = [PathSource(i, ends) for i in self.path]
        N = len(path_sources)
        if not N: return
        for _ in range(N * N):
            stop = True
            for i in range(N - 1):
                upstream = path_sources[i]
                for j in range(i + 1, N):
                    downstream = path_sources[j]
                    if upstream.downstream_from(downstream):
                        if downstream.downstream_from(upstream):
                            if isinstance(downstream.source, Network):
                                if isinstance(upstream.source, Network):
                                    recycles = downstream.source.streams.intersection(upstream.source.streams) 
                                else:
                                    recycles = downstream.source.streams.intersection(upstream.source.outs)
                            elif isinstance(upstream.source, Network):
                                recycles = upstream.source.streams.intersection(downstream.source.outs)
                            else:
                                recycles = set([i for i in upstream.source.outs if i in downstream.source.ins])
                            recycles = [i for i in recycles if i not in ends]
                            if recycles:
                                self.add_recycle(set(recycles))
                                stop = False
                        else:
                            path_sources.remove(downstream)
                            path_sources.insert(i, downstream)
                            upstream = downstream
                            stop = False
                        break
            if stop: break
        self.path = [i.source for i in path_sources]
        if not stop: warn(RuntimeWarning('network path could not be determined'))
    
    @classmethod
    def from_feedstock(cls, feedstock, feeds=(), ends=None, units=None, final=True, recycles=True):
        """
        Create a Network object from a feedstock.
        
        Parameters
        ----------
        feedstock : :class:`~thermosteam.Stream`
            Main feedstock of the process.
        feeds : Iterable[:class:`~thermosteam.Stream`]
            Additional feeds to the process.
        ends : Iterable[:class:`~thermosteam.Stream`], optional
            Streams that not products, but are ultimately specified through
            process requirements and not by its unit source.
        units : Iterable[:class:`~biosteam.Unit`], optional
            All unit operations within the network.
            
        """
        ends = set(ends) if ends else set()
        units = frozenset(units) if units else frozenset()
        recycle_ends = ends.copy()
        linear_paths, cyclic_paths_with_recycle = find_linear_and_cyclic_paths_with_recycle(
            feedstock, ends, units
        )
        linear_networks = [Network(i) for i in linear_paths]
        if linear_networks:
            network, *linear_networks = [Network(i) for i in linear_paths]
            for linear_network in linear_networks:
                network.join_linear_network(linear_network) 
        else:
            network = Network([])
        if recycles: 
            recycle_networks = [Network(*i) for i in cyclic_paths_with_recycle]
            for recycle_network in recycle_networks:
                network.join_recycle_network(recycle_network)
        ends.update(network.streams)
        disjunction_streams = set([i.get_stream() for i in disjunctions])
        for feed in feeds:
            if feed in ends or feed.sink._universal: continue
            downstream_network = cls.from_feedstock(feed, (), ends, units, final=False)
            new_streams = downstream_network.streams
            connections = ends.intersection(new_streams)
            connecting_units = {stream._sink for stream in connections
                                if stream._source and stream._sink
                                and stream not in disjunction_streams
                                and stream._sink in units}
            ends.update(new_streams)
            N_connections = len(connecting_units)
            if N_connections == 0:
                network._append_network(downstream_network)
            elif N_connections == 1:
                connecting_unit, = connecting_units
                network.join_network_at_unit(downstream_network,
                                             connecting_unit)
            else:
                connecting_unit = network.first_unit(connecting_units)
                network.join_network_at_unit(downstream_network,
                                             connecting_unit)
        if final:
            recycle_ends.update(disjunction_streams)
            recycle_ends.update(network.get_all_recycles())
            recycle_ends.update(tmo.utils.products_from_units(network.units))
            network.sort_without_recycle(recycle_ends)
            if recycles: network.reduce_recycles()
            network.sort(recycle_ends)
            network.add_interaction_units()
        return network
    
    @classmethod
    def from_units(cls, units, ends=None, recycles=True):
        """
        Create a System object from all units given.

        Parameters
        ----------
        units : Iterable[:class:`biosteam.Unit`]
            Unit operations to be included.
        ends : Iterable[:class:`~thermosteam.Stream`], optional
            End streams of the system which are not products. Specify this
            argument if only a section of the complete system is wanted, or if
            recycle streams should be ignored.

        """
        unit_set = set(units)
        for u in tuple(unit_set):
            # Do not include auxiliary units
            for au in u.auxiliary_units: 
                au.owner = u
                unit_set.discard(au)
        units = [u for u in units if u in unit_set] 
        feeds = tmo.utils.feeds_from_units(units) + [AbstractStream(source=None, sink=i) for i in units if not i._ins]
        sort_feeds_big_to_small(feeds)
        if feeds:
            feedstock, *feeds = feeds
            if not ends:
                ends = tmo.utils.products_from_units(units) + [i.get_stream() for i in disjunctions]
            system = cls.from_feedstock(
                feedstock, feeds, ends, units, final=True, recycles=recycles,
            )
        else:
            system = cls(())
        return system
    
    def reduce_recycles(self):
        for i in self.path:
            if isinstance(i, Network): i.reduce_recycles()
        if len(self.path) == 1:
            network = self.path[0]
            if isinstance(network, Network):
                self.path = network.path
                self.add_recycle(network.recycle)
        recycle = self.recycle
        if recycle and isinstance(recycle, set):
            sinks = set([i.sink for i in recycle])
            if len(sinks) == 1:
                sink = sinks.pop()
                if len(sink.outs) == 1:
                    self.recycle = sink.outs[0]
            else:
                sources = set([i.source for i in recycle])
                if len(sources) == 1:
                    source = sources.pop()
                    if len(source.ins) == 1:
                        self.recycle = source.ins[0]
    
    def add_interaction_units(self, excluded=None):
        isa = isinstance
        path = self.path
        if excluded is None: excluded = set()
        for i, u in enumerate(path):
            if isa(u, Network):
                u.add_interaction_units(excluded)
            else:
                if u._interaction: excluded.add(u)
                for s in u.outs:
                    sink = s.sink
                    if u in excluded: continue
                    if u._interaction and sink in path[:i] and not any([(sink in i.units if isa(i, Network) else sink is i) for i in path[i+1:]]):
                        excluded.add(sink)
                        path.insert(i+1, sink)
        if len(path) > 1 and path[-1] is path[0]: path.pop()
    
    @property
    def streams(self):
        return tmo.utils.streams_from_units(self.units)
    
    def first_unit(self, units):
        isa = isinstance
        for i in self.path:
            if isa(i, Network):
                if not i.units.isdisjoint(units):
                    return i.first_unit(units)
            elif i in units:
                return i
        raise ValueError('network does not contain any of the given units') # pragma: no cover
    
    def isdisjoint(self, network):
        return self.units.isdisjoint(network.units)
        
    def join_network_at_unit(self, network, unit):
        isa = isinstance
        path_tuple = tuple(self.path)
        self._remove_overlap(network, path_tuple)
        for index, item in enumerate(self.path):
            if isa(item, Network) and unit in item.units:
                if network.recycle:
                    item.join_network_at_unit(network, unit)
                    self.units.update(network.units)
                else:
                    self._insert_linear_network(index, network)
                return
            elif unit is item:
                if network.recycle:
                    self._insert_recycle_network(index, network)
                else:
                    self._insert_linear_network(index, network)
                return
        self.join_linear_network(network)
    
    def join_linear_network(self, linear_network):
        path = self.path
        path_tuple = tuple(path)
        units = linear_network.units
        self._remove_overlap(linear_network, path_tuple)
        for index, item in enumerate(path_tuple):
            if isinstance(item, Network):
                if item.units.intersection(units): self.join_linear_network(item)
            elif item in units:
                self._insert_linear_network(index, linear_network)
                return
        self._append_linear_network(linear_network)
    
    def join_recycle_network(self, network):
        if self.recycle_sink is network.recycle_sink:
            # Feed forward scenario
            self.add_recycle(network.recycle)
            network.recycle_sink = network.recycle = None 
            self._add_linear_network(network)
            return
        path = self.path
        isa = isinstance
        path_tuple = tuple(path)
        self._remove_overlap(network, path_tuple)
        subunits = network.units
        for index, i in enumerate(path_tuple):
            if isa(i, Network) and not network.isdisjoint(i):
                i.join_recycle_network(network)
                self.units.update(subunits)
                return
        for index, item in enumerate(path_tuple):
            if not isa(item, Network) and item in subunits:
                self._insert_recycle_network(index, network)
                return
        raise ValueError('networks must have units in common to join') # pragma: no cover
    
    def add_recycle(self, stream):
        if stream is None: return 
        recycle = self.recycle
        if recycle is None: 
            self.recycle = stream
            return
        if recycle is stream:
            return 
        isa = isinstance
        if isa(recycle, set):
            if isa(stream, set):
                recycle.update(stream)
            else:
                recycle.add(stream)
        else:
            if isa(stream, set):
                self.recycle = {self.recycle, *stream}
            else: 
                self.recycle = {self.recycle, stream}
     
    def _remove_overlap(self, network, path_tuple):
        path = self.path
        units = network.units
        isa = isinstance
        for i in path_tuple:
            if (not isa(i, Network) and i in units): path.remove(i)
    
    def _append_linear_network(self, network):
        self.path.extend(network.path)
        self.units.update(network.units)
    
    def _append_recycle_network(self, network):
        self.path.append(network)
        self.units.update(network.units)
    
    def _append_network(self, network):
        if self.recycle:
            cls = type(self)
            new = cls.__new__(cls)
            new.path = self.path; new.units = self.units
            new.recycle = self.recycle; new.recycle_sink = self.recycle_sink
            self.recycle = self.recycle_sink = None
            self.path = [new, network] if network.recycle else [new, *network.path]
            self.units = self.units.union(network.units)
        elif network.recycle:
            self._append_recycle_network(network)
        else:
            self._append_linear_network(network)
    
    def _insert_linear_network(self, index, network):
        path = self.path
        self.path = [*path[:index], *network.path, *path[index:]]
        self.units.update(network.units)
    
    def _insert_recycle_network(self, index, network):
        path = self.path
        path.insert(index, network)
        self.units.update(network.units)
        if len(path) == 1:
            network = path[0]
            if isinstance(network, Network):
                self.path = network.path
                self.recycle = network.recycle
                self.recycle_sink = network.recycle_sink

    def _add_linear_network(self, network):
        path = self.path
        isa = isinstance
        path_tuple = tuple(path)
        self._remove_overlap(network, path_tuple)
        subunits = network.units
        for index, i in enumerate(path_tuple):
            if isa(i, Network) and not network.isdisjoint(i):
                i._add_linear_network(network)
                self.units.update(subunits)
                return
        for index, item in enumerate(path_tuple):
            if not isa(item, Network) and item in subunits:
                self._insert_linear_network(index, network)
                return
        self._append_linear_network(network)
    
    def __repr__(self): # pragma: no cover
        recycle = self.recycle
        if recycle:
            return f"{type(self).__name__}(path={self.path}, recycle={self.recycle})"
        else:
            return f"{type(self).__name__}(path={self.path})"
    
    def _info(self, spaces):
        info = f"{type(self).__name__}("
        spaces += 4 * " "
        end = ',\n' + spaces
        path_info = []
        path = self.path
        isa = isinstance
        info += '\n' + spaces
        for i in path:
            path_info.append(i._info(spaces) if isa(i, Network) else str(i))
        info += '[' + (end + " ").join(path_info) + ']'
        recycle = self.recycle
        if recycle:
            if isinstance(recycle, set):
                recycle = ", ".join([i._source_info() for i in recycle])
                recycle = '[' + recycle + ']'
            else:
                recycle = recycle._source_info()
            info += end + f"recycle={recycle})"
        else:
            info += ')'
        return info
    
    def _ipython_display_(self):
        self.show()
    
    def show(self):
        print(self._info(spaces=""))
