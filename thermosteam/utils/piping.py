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
from typing import NamedTuple, Optional
from .decorators import registered, thermo_user
import numpy as np

int_types = (int, np.int32)

__all__ = ('stream_info', 'AbstractStream', 'AbstractUnit',
           'Connection', 'Inlet', 'Outlet')

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
    __slots__ = ('_ID', '_source', '_sink')
    
    def __init__(self, ID=None, source=None, sink=None):
        self._ID = ID
        self._source = source
        self._sink = sink

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
        if source:
            outs = source.outs
            index = outs.index(self)
            outs[index] = None

    def disconnect_sink(self):
        """Disconnect stream from sink."""
        sink = self._sink
        if sink:
            ins = sink.ins
            index = ins.index(self)
            ins[index] = None

    def disconnect(self):
        """Disconnect stream from unit."""
        self.disconnect_source()
        self.disconnect_sink()

    def __sub__(self, index):
        if isinstance(index, int):
            return Inlet(self, index)
        elif hasattr(index, 'separate_out'):
            new = self.copy()
            new.separate_out(index)   
            return new
        else:
            return index.__rsub__(self)
    
    def __rsub__(self, index):
        if isinstance(index, int):
            return Outlet(self, index)
        else:
            return index.__sub__(self)
    
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

    def show(self):
        return self.materialize().show()
    
    def print(self):
        return self.materialize().print()
    
    def _ipython_display_(self):
        return self.materialize()._ipython_display_()
    
    def _get_tooltip_string(self):
        return self.materialize()._get_tooltip_string()


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
    MissingStream = Stream = AbstractStream
    def __init__(self, size, streams, thermo, fixed_size, stacklevel):
        self._size = size
        self._fixed_size = fixed_size
        Stream = self.Stream
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
                    elif isa(streams, Stream):
                        self._streams[0] = redock(streams, stacklevel)
                    else:
                        N = len(streams)
                        n_missing(size, N) # Make sure size is not too big
                        self._streams[:N] = [redock(i, stacklevel+1) if isa(i, Stream)
                                             else dock(Stream(i, thermo=thermo)) for i in streams]
            elif streams is not None:
                if isa(streams, str):
                    self._streams = [dock(Stream(streams, thermo=thermo))]
                elif isinstance(streams, Stream):
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
        return self.MissingStream(source=None, sink=None)
        
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
        elif not isinstance(stream, (self.Stream, self.MissingStream)):
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
        self._undock(stream)
        missing_stream = self._create_missing_stream()
        self.replace(stream, missing_stream)
        
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
    __slots__ = ('_sink', '_fixed_size')
    
    def __init__(self, sink, size, streams, thermo, fixed_size, stacklevel):
        self._sink = sink
        super().__init__(size, streams, thermo, fixed_size, stacklevel)
    
    @property
    def sink(self):
        return self._sink
    
    def _create_missing_stream(self):
        return self.MissingStream(source=None, sink=self._sink)
    
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
    """Create an Outlets object which serves as output streams for nodes."""
    __slots__ = ('_source',)
    
    def __init__(self, source, size, streams, thermo, fixed_size, stacklevel):
        self._source = source
        super().__init__(size, streams, thermo, fixed_size, stacklevel)
    
    @property
    def source(self):
        return self._source
    
    def _create_missing_stream(self):
        return self.MissingStream(source=self._source, sink=None)
    
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

class Inlet:
    """
    Create an Inlet object that connects an stream to a unit using piping
    notation:
    
    Parameters
    ----------
    stream : AbstractStream
    index : int
        
    Examples
    --------
    First create an stream and a unit:
    
    >>> from thermosteam import AbstractStream, AbstractUnit
    >>> stream = AbstractStream('s1')
    >>> unit = AbstractUnit('M1', outs=('out'))
    
    Inlet objects are created using -pipe- notation:
        
    >>> stream-0
    <Inlet: s1-0>
    
    Use pipe notation to create an inlet and connect the stream:
    
    >>> stream-0-unit # The last unit is returned to continue piping; just ignore this
    <Unit: M1>
    >>> unit.show()
    Unit: M1
    ins...
    [1] s1
    outs...
    [0] out
    
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


class Outlet:
    """
    Create an Outlet object that connects an stream to a unit using piping
    notation:
    
    Parameters
    ----------
    stream : AbstractStream
    index : int
    
    Examples
    --------
    First create an stream and a unit:
    
    >>> from thermosteam import AbstractStream, AbstractUnit
    >>> stream = AbstractStream('s1')
    >>> unit = AbstractUnit('M1')
    
    Outlet objects are created using -pipe- notation:
        
    >>> 0**stream
    <Outlet: 0-s1>
    
    Use -pipe- notation to create an outlet and connect the stream:
    
    >>> unit**0**stream # First unit is returned to continue backwards piping; just ignore this
    <Unit: M1>
    >>> unit.show()
    Unit: M1
    ins...
    [0] 
    outs...
    [0] s1
    
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


# %% Nodes

@thermo_user
@registered(ticket_name='U')
class AbstractUnit:
    _universal = False
    _interaction = False
    Inlets = AbstractInlets
    Outlets = AbstractOutlets
    def _init_inlets(self, ins):
        self._ins = self.Inlets(
            self, self._N_ins, ins, self._thermo, self._ins_size_is_fixed, self._stacklevel,
        )
        
    def _init_outlets(self, outs, **kwargs):
        self._outs = self.Outlets(
            self, self._N_outs, outs, self._thermo, self._outs_size_is_fixed, self._stacklevel,
        )
    
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
        Insert unit between two nodes at a given stream connection.
        
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
        added_node = False
        if outlet is None:
            if self._outs_size_is_fixed:
                if self._N_outs == 1:
                    sink.ins.replace(stream, self.outs[0])
                else:
                    raise ValueError("undefined outlet; must pass outlet when outlets are fixed and multiple are available")
            else:
                self.outs.append(stream)
                added_node = True
        else:
            if isinstance(outlet, AbstractStream):
                if outlet.source is not self:
                    raise ValueError("source of given outlet must be this object")
            else:
                outlet = self.outs[outlet]
            sink.ins.replace(stream, outlet)
        if inlet is None:
            if self._ins_size_is_fixed or added_node:
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
        """Add upsteam neighboring nodes to set."""
        for s in self._ins:
            u = s._source
            if u and (universal or not u._universal) and not (ends and s in ends):
                set.add(u)

    def _add_downstream_neighbors_to_set(self, set, ends, universal):
        """Add downstream neighboring nodes to set."""
        for s in self._outs:
            u = s._sink
            if u and (universal or not u._universal) and not (ends and s in ends):
                set.add(u)

    def get_downstream_nodes(self, ends=None, universal=True):
        """Return a set of all nodes downstream."""
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
    
    def get_upstream_nodes(self, ends=None, universal=True):
        """Return a set of all nodes upstream."""
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
        Return a set of all neighboring nodes within given radius.
        
        Parameters
        ----------
        radius : 
            Maximum number streams between neighbors.
        downstream : 
            Whether to include downstream nodes.
        upstream : 
            Whether to include upstream nodes.
        ends :
            Streams that mark the end of the neighborhood.
        universal :
            Whether to include universal nodes.
        
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
            nodes = self.neighborhood(radius, upstream, downstream)
            nodes.add(self)
        else:
            nodes = [self]
        return bst.System(None, nodes).diagram(
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
