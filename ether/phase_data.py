# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:41:50 2019

@author: yoelr
"""
from .settings import settings
from .exceptions import UndefinedPhase
import numpy as np

__all__ = ('PhaseData',
           'MultiPhaseData',
           'PhaseMolarFlow', 
           'MultiPhaseMolarFlow',
           'PhaseMassFlow', 
           'MultiPhaseMassFlow',
           'PhaseVolumetricFlow',
           'MultiPhaseVolumetricFlow')

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]
        
class PhaseData:
    __slots__ = ('_data', '_chemicals')
    units = None
    
    def __new__(cls, chemicals=None, **IDdata):
        self = cls.blank()
        if IDdata:
            IDs, data = zip(*IDdata.items())
            self._data[self._chemicals.indices(IDs)] = data
        return self
    
    def _set_chemicals(self, chemicals):
        self._chemicals = chemicals = settings.get_default_chemicals(chemicals)
    
    @classmethod
    def blank(cls, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        self._data = np.zeros(self._chemicals.size, float)
        return self
    
    @classmethod
    def from_data(cls, data=None, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        if data:
            if not isinstance(data, np.ndarray):
                data = np.array(data, float)
            elif data.size == chemicals.size:
                self.data = data
            else:
                raise ValueError('size of data must be equal to '
                                 'size of chemicals')
        else:
            self._data = np.zeros(chemicals.size, float)
    
    @property
    def data(self):
        return self._data
    @property
    def chemicals(self):
        return self._chemicals
        
    def sum(self):
        return self._data.sum()
        
    def _get_index(self, IDs):
        isa = isinstance
        if isa(IDs, str):
            return self._chemicals.index(IDs)
        elif isa(IDs, slice):
            return IDs
        else:
            return self._chemicals.indices(IDs)
        
    def __getitem__(self, IDs):
        return self._data[self._get_index(IDs)]
    
    def __setitem__(self, IDs, data):
        self._data[self._get_index(IDs)] = data
    
    def __iter__(self):
        return self._data.__iter__()
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        IDdata = [f"{ID}={i}" for ID, i in zip(self._chemicals.IDs, self._data) if i]
        if len(IDdata) > 1 and tab:
            dlim = ",\n" + tab
            IDdata = "\n" + tab + dlim.join(IDdata)
        else:
            IDdata = ', '.join(IDdata)
        return f"{type(self).__name__}({IDdata})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
        IDs = self.chemicals.IDs
        data = self.data
        IDs, data = nonzeros(IDs, data)
        len_ = len(data)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n "
        else:
            basic_info = f"{type(self).__name__}:\n "
        new_line_spaces = ' '        
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
      

class MultiPhaseData:
    __slots__ = ('_phases', '_phase_index', '_phase_data', '_data', '_chemicals')
    _cached_phase_index = {}
    _PhaseData = PhaseData
    units = PhaseData.units
    
    def __new__(cls, phases=None, chemicals=None, **phase_data):
        self = cls.blank(phases or "".join(phase_data), chemicals)
        if phase_data:
            data = self._data
            chemical_indices = self._chemicals.indices
            phase_index = self._get_phase_index
            for phase, IDdata in phase_data.items():
                IDs, row = zip(*IDdata)
                data[phase_index(phase), chemical_indices(IDs)] = row
        return self
    
    _set_chemicals = PhaseData._set_chemicals
    
    def _set_phases(self, phases):
        self._phases = phases
        cached = self._cached_phase_index
        if phases in cached:
            self._phase_index = cached[phases]
        else:
            self._phase_index = cached[phases] = {j:i for i,j in enumerate(phases)}
    
    @classmethod
    def blank(cls, phases, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        shape = (len(phases), self._chemicals.size)
        self._data = data = np.zeros(shape, float)
        self._phase_data = tuple(zip(phases, data))
        return self
    
    @classmethod
    def from_data(cls, phases, data=None, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        M_phases = len(phases)
        N_chemicals = self._chemicals.size
        if data:
            if not isinstance(data, np.ndarray):
                data = np.array(data, float)
            M, N = data.shape
            assert M == M_phases, ('number of phases must be equal to '
                                   'the number of data rows')
            assert N == N_chemicals, ('size of chemicals '
                                      'must be equal to '
                                      'number of data columns')
            self._data = data
        else:
            self._data = data = np.zeros((M_phases, N_chemicals), float)
        self._phase_data = tuple(zip(phases, data))
    
    @property
    def data(self):
        return self._data
    @property
    def phases(self):
        return self._phases
    @property
    def chemicals(self):
        return self._chemicals
    
    def sum(self):
        return self._data.sum()
    
    def sum_phases(self):
        return self._data.sum(0)
    
    def sum_chemicals(self):
        return self._data.sum(1)
    
    def get_phase(self, phase):
        phase_data = object.__new__(self._PhaseData)
        phase_data._data = self._data[self._phase_index[phase]]
        phase_data._chemicals = self._chemicals
        return phase_data
    
    def _get_index(self, phases_IDs):
        isa = isinstance
        if isa(phases_IDs, tuple):
            phases, IDs = phases_IDs
            if isa(IDs, str):
                IDs_index = self._chemicals.index(IDs)
            elif isa(IDs, slice):
                IDs_index = IDs
            else:
                IDs_index = self._chemicals.indices(IDs)
        else:
            phases = phases_IDs
            IDs_index = ...
        if isa(phases, slice):
            phases_index = phases
        elif len(phases) == 1:
            phases_index = self._get_phase_index(phases)
        else:
            phases_index = self._get_phase_indices(phases)
        return phases_index, IDs_index
    
    def _get_phase_index(self, phase):
        try:
            return self._phase_index[phase]
        except KeyError:
            raise UndefinedPhase(phase)
    
    def _get_phase_indices(self, phases):
        try:
            index = self._phase_index
            return [index[i] for i in phases]
        except KeyError:
            for i in phases:
                if i not in index:
                    raise UndefinedPhase(i)
    
    def __getitem__(self, phases_IDs):
        phases, IDs = index = self._get_index(phases_IDs)
        data = self._data
        if isinstance(phases, list):
            return [data[i, IDs] for i in phases]
        else:                
            return data[index]
    
    def __setitem__(self, phases_IDs, data):
        phases, IDs = index = self._get_index(phases_IDs)
        if isinstance(phases, list):
            data = np.asarray(data)
            if data.ndim <= 1:
                for i in phases: self._data[i, IDs] = data
            elif len(phases) == len(data):
                for i, row in zip(phases, data): self._data[i, IDs] = row
            else:
                raise ValueError(f"shape mismatch: value array of shape {data.shape} "
                                  "could not be broadcast to indexing result of shape "
                                 f"{(len(phases), self._data[:, IDs].shape[1])}")
        else:
            self._data[index] = data
    
    def __iter__(self):
        return self._phase_data.__iter__()
    
    def __format__(self, tabs="1"):
        IDs = self._chemicals.IDs
        phase_data = []
        for phase, data in self._phase_data:
            IDdata = ", ".join([f"('{ID}', {i:.3g})" for ID, i in zip(IDs, data) if i])
            phase_data.append(f"{phase}=[{IDdata}]")
        tabs = int(tabs)
        if tabs:
            tab = tabs*4*" "
            dlim = ",\n" + tab 
        else:
            dlim = ", "
        phase_data = dlim.join(phase_data)
        if self.sum_chemicals().all():
            phases = ""
            if phase_data:
                phase_data = "\n" + tab + phase_data
        else:
            phases = f'phases={self.phases}'
            if phase_data:
                phase_data = dlim + phase_data
        return f"{type(self).__name__}({phases}{phase_data})"
    
    def __repr__(self):
        return self.__format__("1")
    
    def _info(self, N):
        """Return string with all specifications."""
        IDs = self.chemicals.IDs
        index, = np.where(self.sum_phases() != 0)
        len_ = len(index)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n"
        else:
            basic_info = f"{type(self).__name__}:\n"
        all_IDs = [IDs[i] for i in index]

        # Length of species column
        all_lengths = [len(i) for i in IDs]
        maxlen = max(all_lengths + [8])  # include length of the word 'species'

        # Set up chemical data for all phases
        phases_flowrates_info = ''
        for phase in self.phases:
            phase_data = self[phase, all_IDs]
            IDs, data = nonzeros(all_IDs, phase_data)
            if not IDs: continue
        
            # Get basic structure for phase data
            beginning = f' ({phase}) '
            new_line_spaces = len(beginning) * ' '

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

            # Put it together
            phases_flowrates_info += beginning + flowrates + '\n'
            
        return basic_info + phases_flowrates_info[:-1]
    show = PhaseData.show
    _ipython_display_ = show
    
def new_PhaseData(name, units):
    PhaseDataSubclass = type('Phase' + name, (PhaseData,), {})
    MultiPhaseDataSubclass = type('MultiPhase' + name, (MultiPhaseData,), {})
    PhaseDataSubclass.__slots__ = MultiPhaseDataSubclass.__slots__ = ()
    PhaseDataSubclass.units = MultiPhaseDataSubclass.units = units    
    MultiPhaseDataSubclass._PhaseData = PhaseDataSubclass
    return PhaseDataSubclass, MultiPhaseDataSubclass
    
PhaseMolarFlow, MultiPhaseMolarFlow = new_PhaseData('MolarFlow', 'kmol/hr')
PhaseMassFlow, MultiPhaseMassFlow = new_PhaseData('MassFlow', 'kg/hr')
PhaseVolumetricFlow, MultiPhaseVolumetricFlow = new_PhaseData('VolumetricFlow', 'm^3/hr')