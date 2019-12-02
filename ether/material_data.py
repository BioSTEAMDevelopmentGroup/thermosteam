# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:41:50 2019

@author: yoelr
"""
from .settings import settings
from .exceptions import UndefinedPhase
import numpy as np

__all__ = ('MaterialData', 'MultiPhaseMaterialData')

phase_names = {'s': 'solid',
               'l': 'liquid',
               'L': 'LIQUID',
               'g': 'vapor'}

all_index = slice(None)

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]

class MaterialData:
    __slots__ = ('phase', 'T', 'P', '_data', '_units', '_chemicals')
    
    def __init__(self, phase, T, P, data=None, units=None, chemicals=None, **ID_data):
        self._chemicals = chemicals = settings.get_default_chemicals(chemicals)
        self.phase = phase
        self._units = units
        self.T = T
        self.P = P
        if data:
            if not isinstance(data, np.ndarray):
                data = np.array(data, float)
            if ID_data:
                raise ValueError("may specify either 'data' or "
                                 "'ID_data', but not both")
            elif data.size == chemicals.size:
                self.data = data
            else:
                raise ValueError('size of data must be equal to '
                                 'size of ID_data')
        else:
            self._data = np.zeros(chemicals.size, float)
            if ID_data:
                IDs, data = zip(*ID_data.items())
                self._data[chemicals.indices(IDs)] = data
    
    @property
    def data(self):
        return self._data
    @property
    def units(self):
        return self._units
    @property
    def chemicals(self):
        return self._chemicals
        
    def sum(self):
        return self._data.sum()
    
    def __iter__(self):
        return self._data.__iter__()
        
    def __getitem__(self, IDs):
        if isinstance(IDs, str):
            return self._data[self._chemicals.index(IDs)]
        elif IDs == all_index:
            return self._data
        else:
            return self._data[self._chemicals.indices(IDs)]
    
    def __setitem__(self, IDs, data):
        if isinstance(IDs, str):
            index = self._chemicals.index(IDs)
        elif IDs == all_index:
            self._data[:] = data
        else:
            index = self._chemicals.indices(IDs)
            self._data[index] = data
    
    def __repr__(self):
        data = [f"{ID}={i}" for ID, i in zip(self._chemicals.IDs, self._data)]
        if data:
            kwdata = ", " + ", ".join(data)
        else:
            kwdata = ""
        return f"{type(self).__name__}('{self.phase}', {self.T}, {self.P}{kwdata})"
    
    def _info(self, N):
        """Return string with all specifications."""
        basic_info = (f"{type(self).__name__}:\n"
                     +f" phase: '{self.phase}', T: {self.T:.2f} K, P: {self.P:.6g} Pa\n")
        IDs = self.chemicals.IDs
        data = self.data
        IDs, data = nonzeros(IDs, data)
        len_ = len(data)
        if len_ == 0:
            return basic_info + ' data: (empty)' 
        else:
            units = self.units
            if units:
                start_data = f" data ({units}): "
            else:
                start_data = " data: "
        new_line_spaces = len(start_data) * ' '        
        data_info = ''
        lengths = [len(i) for i in IDs]
        maxlen = max(lengths) + 1
        _N = N - 1
        for i in range(len_-1):
            spaces = ' ' * (maxlen - lengths[i])
            if i == _N:
                data_info += '...\n' + new_line_spaces
                break
            data_info += IDs[i] + spaces + f' {data[i]:.3g}\n' + new_line_spaces
        spaces = ' ' * (maxlen - lengths[len_-1])
        data_info += IDs[len_-1] + spaces + f' {data[len_-1]:.3g}'
        return (basic_info 
              + start_data
              + data_info)

    def show(self, N=5):
        """Print all specifications.
        
        Parameters
        ----------
        N: int, optional
            Number of compounds to display.
        
        """
        print(self._info(N))
    _ipython_display_ = show
      
        
class MultiPhaseMaterialData:
    __slots__ = ('T', 'P', '_phases', '_phase_index', '_data', '_units', '_chemicals')
    
    def __init__(self, phases, T, P, data=None, units=None, chemicals=None):
        self._chemicals = chemicals = settings.get_default_chemicals(chemicals)
        self._phases = phases
        self._units = units
        self._phase_index  = {j:i for i,j in enumerate(phases)}
        self.T = T
        self.P = P
        if data:
            if not isinstance(data, np.ndarray):
                data = np.array(data, float)
            M, N = data.shape
            assert M == len(phases), ('number of phases must be equal to '
                                      'the number of data rows')
            assert N == chemicals.size, ('size of chemicals '
                                        'must be equal to '
                                        'number of data columns')
            self._data = data
        else:
            shape = (len(phases), chemicals.size)
            self._data = np.zeros(shape, float)
    
    def phase_index(self, phase):
        try:
            return self._phase_index[phase]
        except KeyError:
            raise UndefinedPhase(phase)
    
    def phases_index(self, phases):
        try:
            index = self._phase_index
            return [index[i] for i in phases]
        except KeyError:
            for i in phases:
                if i not in index:
                    raise UndefinedPhase(i)
    
    @property
    def data(self):
        return self._data
    @property
    def phases(self):
        return self._phases
    @property
    def units(self):
        return self._units
    @property
    def chemicals(self):
        return self._chemicals
    
    def sum_phases(self):
        return self._data.sum(0)
    
    def sum_chemicals(self):
        return self._data.sum(1)
        
    def sum(self):
        return self._data.sum()
    
    def __iter__(self):
        return zip(self._phases, self._data)
    
    def __getitem__(self, phases_IDs):
        isa = isinstance
        if isa(phases_IDs, tuple):
            phases, IDs = phases_IDs
            if isa(IDs, str):
                IDs_index = self._chemicals.index(IDs)
            elif IDs == all_index:
                IDs_index = ...
            else:
                IDs_index = self._chemicals.indices(IDs)
        else:
            phases = phases_IDs
            IDs_index = ...
        if phases == all_index:
            phases_index = phases
        elif len(phases) == 1:
            phases_index = self.phase_index(phases)
        else:
            phases_index = self.phases_index(phases)
        return self._data[phases_index, IDs_index]
    
    def __setitem__(self, phases_IDs, data):
        isa = isinstance
        all_index = slice(None)
        if isa(phases_IDs, tuple):
            phases, IDs = phases_IDs
            if isa(IDs, str):
                IDs_index = self._chemicals.index(IDs)
            elif IDs == all_index:
                IDs_index = ...
            else:
                IDs_index = self._chemicals.indices(IDs)
        else:
            phases = phases_IDs
            IDs_index = ...
        if phases == all_index:
            phases_index = phases
        elif len(phases) == 1:
            phases_index = self.phase_index(phases)
        else:
            phases_index = self.phases_index(phases)
        self._data[phases_index, IDs_index] = data
    
    def __repr__(self):
        return f"{type(self).__name__}(phases='{self.phases}', T={self.T}, P={self.P}, data=...)"
    
    def _info(self, N):
        """Return string with all specifications."""
        basic_info = (f"{type(self).__name__}:\n"
                     +f" phases: '{self.phases}', T: {self.T:.2f} K, P: {self.P:.6g} Pa\n")
        IDs = self.chemicals.IDs
        index, = np.where(self.sum_phases() != 0)
        len_ = len(index)
        if len_ == 0:
            return basic_info + ' data: (empty)' 
        all_IDs = [IDs[i] for i in index]

        # Length of species column
        all_lengths = [len(i) for i in IDs]
        maxlen = max(all_lengths + [9])  # include length of the word 'species'

        # Set up chemical data for all phases
        phases_flowrates_info = ''
        add_header = bool(self.units)
        for phase in self.phases:
            phase_data = self[phase, all_IDs]
            IDs, data = nonzeros(IDs, phase_data)
            
            # Get basic structure for phase data
            phase_full_name = phase_names[phase]
            beginning = f' {phase_full_name}: '
            new_line_spaces = len(beginning) * ' '

            # Make up for soild and vapor phase having 5 letters
            # (while liquid has 6 letters)
            if phase in 'sg':
                beginning += ' '
                new_line_spaces += ' '

            # Set chemical data
            flowrates = ''
            l = len(data)
            lengths = [len(i) for i in IDs]
            _N = N - 1
            for i in range(l-1):
                spaces = ' ' * (maxlen - lengths[i])
                if i == _N:
                    flowrates += '...\n' + new_line_spaces
                    break
                flowrates += f'{IDs[i]} ' + spaces + \
                    f' {data[i]:.3g}\n' + new_line_spaces
            spaces = ' ' * (maxlen - lengths[l-1])
            flowrates += (f'{IDs[l-1]} ' + spaces
                          + f' {data[l-1]:.3g}')

            # Add header to flow rates
            if add_header:
                spaces = ' ' * (maxlen - 9)
                beginning = (f'{new_line_spaces}chemicals{spaces}  {self.units}\n'
                             + beginning)
            else:
                add_header = False

            # Put it together
            phases_flowrates_info += beginning + flowrates + '\n\n'
            
        return basic_info + phases_flowrates_info[:-2]
    show = MaterialData.show
    _ipython_display_ = show