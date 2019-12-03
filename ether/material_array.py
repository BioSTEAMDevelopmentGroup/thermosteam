# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:41:50 2019

@author: yoelr
"""
from .settings import settings
from .exceptions import UndefinedPhase
import numpy as np

__all__ = ('ChemicalArray',
           'PhaseArray',
           'MaterialArray',
           'ChemicalMolarFlow', 
           'PhaseMolarFlow',
           'MolarFlow',
           'ChemicalMassFlow', 
           'PhaseMassFlow',
           'MassFlow',
           'ChemicalVolumetricFlow',
           'PhaseVolumetricFlow',
           'VolumetricFlow')

# %% Utilities

def nonzeros(IDs, data):
    index, = np.where(data != 0)
    return [IDs[i] for i in index], data[index]

# %% Abstract data emulator
    
class ArrayEmulator:
    __slots__ = ()
    _cached_phase_index = {}
    chemicals = units = phases = None
    
    def _assert_safety(self, other):
        if isinstance(other, ArrayEmulator):
            if settings._debug:
                assert other.chemicals is self.chemicals, "chemicals do not match"
                assert other.units == self.units, "units do not match"
                assert other.phases == self.phases, "phases do not match"
            return True
        return False
    
    def copy(self):
        new = self._copy_without_data()
        new._data = self._data.copy()
        return new
    
    @property
    def data(self):
        return self._data
    
    def __getattr__(self, attr):
        return getattr(self._data, attr)
    
    def __getitem__(self, key):
        return self._data[self._get_index(key)]
    
    def __setitem__(self, key, data):
        self._data[self._get_index(key)] = data
    
    def __lt__(self, other):
        return self._data.__lt__(other._data
                                 if self._assert_safety(other)
                                 else other)
    
    def __le__(self, other):
        return self._data.__le__(other._data
                                 if self._assert_safety(other)
                                 else other)
    
    def __eq__(self, other):
        return self._data.__eq__(other._data
                                 if self._assert_safety(other)
                                 else other)
    
    def __ne__(self, other):
        return self._data.__ne__(other._data
                                 if self._assert_safety(other)
                                 else other)
    
    def __gt__(self, other):
        return self._data.__gt__(other._data
                                 if self._assert_safety(other)
                                 else other)
    
    def __ge__(self, other):
        return self._data.__ge__(other._data
                                 if self._assert_safety(other)
                                 else other)
    
    def __add__(self, other):
        new = self.copy()
        new._data.__iadd__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
    
    def __sub__(self, other):
        new = self.copy()
        new._data.__isub__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
    
    def __mul__(self, other):
        new = self.copy()
        new._data.__imul__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
        
    def __floordiv__(self, other):
        new = self.copy()
        new._data.__ifloordiv__(other._data
                                if self._assert_safety(other)
                                else other)
        return new
    
    def __mod__(self, other):
        new = self.copy()
        new._data.__imod__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
    
    def __divmod__(self, other):
        new = self.copy()
        new._data.__idivmod__(other._data
                              if self._assert_safety(other)
                              else other)
        return new
    
    def __pow__(self, other):
        new = self.copy()
        new._data.__idivmod__(other._data
                              if self._assert_safety(other)
                              else other)
        return new
    
    def __lshift__(self, other):
        new = self.copy()
        new._data.__ilshift__(other._data
                              if self._assert_safety(other)
                              else other)
        return new
        
    def __rshift__(self, other):
        new = self.copy()
        new._data.__irshift__(other._data
                              if self._assert_safety(other)
                              else other)
        return new
    
    def __and__(self, other):
        new = self.copy()
        new._data.__iand__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
    
    def __xor__(self, other):
        new = self.copy()
        new._data.__ixor__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
    
    def __or__(self, other):
        new = self.copy()
        new._data.__ior__(other._data
                          if self._assert_safety(other)
                          else other)
        return new
    
    def __radd__(self, other):
        new = self.copy()
        new._data.__iadd__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
        
    def __rsub__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rsub__(other._data
                                        if self._assert_safety(other)
                                        else other)
        return new
    
    def __rmul__(self, other):
        new = self.copy()
        new._data.__imul__(other._data
                           if self._assert_safety(other)
                           else other)
        return new
        
    def __rtruediv__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rtruediv__(other._data
                                            if self._assert_safety(other)
                                            else other)
        return new
        
    def __rfloordiv__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rtruediv__(other._data
                                            if self._assert_safety(other)
                                            else other)
        return new
        
    def __rmod__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rmod__(other._data
                                        if self._assert_safety(other)
                                        else other)
        return new
        
    def __rdivmod__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rdivmod__(other._data
                                           if self._assert_safety(other)
                                           else other)
        return new
    
    def __rpow__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rpow__(other._data
                                        if self._assert_safety(other)
                                        else other)
        return new
    
    def __rlshift__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rlshift__(other._data
                                           if self._assert_safety(other)
                                           else other)
        return new
    
    def __rrshift__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rrshift__(other._data
                                           if self._assert_safety(other)
                                           else other)
        return new
    
    def __rand__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rand__(other._data
                                        if self._assert_safety(other)
                                        else other)
        return new
    
    def __rxor__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__rxor__(other._data
                                        if self._assert_safety(other)
                                        else other)
        return new
    
    def __ror__(self, other):
        new = self._copy_without_data()
        new._data = self._data.__ror__(other._data
                                       if self._assert_safety(other)
                                       else other)
        return new

    def __iadd__(self, other):
        self._data.__iadd__(other._data
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __isub__(self, other):
        self._data.__isub__(other._data 
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __imul__(self, other):
        self._data.__imul__(other._data
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __itruediv__(self, other):
        self._data.__itruediv__(other._data
                                if self._assert_safety(other)
                                else other)
        return self
        
    def __ifloordiv__(self, other):
        self._data.__ifloordiv__(other._data
                                 if self._assert_safety(other)
                                 else other)
        return self
        
    def __imod__(self, other):
        self._data.__imod__(other._data
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __ipow__(self, other):
        self._data.__ipow__(other._data
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __ilshift__(self, other):
        self._data.__ilshift__(other._data
                               if self._assert_safety(other)
                               else other)
        return self
        
    def __irshift__(self, other):
        self._data.__irshift__(other._data 
                               if self._assert_safety(other)
                               else other)
        return self
        
    def __iand__(self, other):
        self._data.__iand__(other._data
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __ixor__(self, other):
        self._data.__ixor__(other._data
                            if self._assert_safety(other)
                            else other)
        return self
        
    def __ior__(self, other):
        self._data.__ior__(other._data
                           if self._assert_safety(other)
                           else other)
        return self
        
    def __neg__(self):
        new = self._copy_without_data()
        new._data = -self._data
        return new
        
    def __pos__(self):
        new = self._copy_without_data()
        new._data = +self._data
        return new
        
    def __abs__(self):
        new = self._copy_without_data()
        new._data = self._data.abs()
        return new


# %% Phase data

class ChemicalArray(ArrayEmulator):
    __slots__ = ('_data', '_chemicals')
    
    def __new__(cls, chemicals=None, **IDdata):
        self = cls.blank()
        if IDdata:
            IDs, data = zip(*IDdata.items())
            self._data[self._chemicals.indices(IDs)] = data
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._chemicals)
    
    def _set_chemicals(self, chemicals):
        self._chemicals = chemicals = settings.get_chemicals(chemicals)
    
    def fraction(self):
        data = self._data
        sum = data.sum()
        return ChemicalArray.from_data(data/sum if sum else data.copy(), self._chemicals)
    
    def copy(self):
        new = self._copy_without_data()
        new._data = self._data.copy()
        return new
    
    def _copy_without_data(self):
        new = super().__new__(self.__class__)
        new._chemicals = self._chemicals
        return new
    
    @classmethod
    def blank(cls, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        self._data = np.zeros(self._chemicals.size, float)
        return self
    
    @classmethod
    def from_data(cls, data, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        if not isinstance(data, np.ndarray):
            data = np.array(data, float)
        if settings.debug:
            assert data.size == self._chemicals.size, ('size of data must be equal to '
                                                       'size of chemicals')
        self._data = data
        return self
    
    @property
    def chemicals(self):
        return self._chemicals
        
    def _get_index(self, IDs):
        isa = isinstance
        if isa(IDs, str):
            return self._chemicals.index(IDs)
        elif isa(IDs, slice):
            return IDs
        else:
            return self._chemicals.indices(IDs)
    
    def __iter__(self):
        return self._data.__iter__()
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        IDdata = [f"{ID}={i:.4g}" for ID, i in zip(self._chemicals.IDs, self._data) if i]
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
            data_info += IDs[i] + spaces + f' {data[i]:.4g}\n' + new_line_spaces
        spaces = ' ' * (maxlen - lengths[len_-1])
        data_info += IDs[len_-1] + spaces + f' {data[len_-1]:.4g}'
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
      
    
class PhaseArray(ArrayEmulator):
    
    def __new__(cls, phases, **phase_data):
        self = cls.blank(phases or phase_data)
        if phase_data:
            phases, data = zip(*phase_data)
            self._data[self._get_phase_indices(phases)] = data
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases)
    
    @property
    def phases(self):
        return self._phases
    
    def fraction(self):
        data = self._data
        sum = data.sum()
        return PhaseArray.from_data(data/sum if sum else data.copy(), self._phases)
    
    def _set_phases(self, phases):
        self._phases = phases = tuple(phases)
        cached = self._cached_phase_index
        if phases in cached:
            self._phase_index = cached[phases]
        else:
            self._phase_index = cached[phases] = {j:i for i,j in enumerate(phases)}
    
    def _get_phase_index(self, phase):
        try:
            return self._phase_index[phase]
        except KeyError:
            raise UndefinedPhase(phase)
    
    def _get_phase_indices(self, phases):
        index = self._phase_index
        try:
            return [index[i] for i in phases]
        except KeyError:
            for i in phases:
                if i not in index:
                    raise UndefinedPhase(i)
    
    def _copy_without_data(self):
        new = super().__new__(self.__class__)
        new._phases = self._phases
        return new
    
    @classmethod
    def blank(cls, phases):
        self = super().__new__(cls)
        self._set_phases(phases)
        self._data = np.zeros(len(self._phases), float)
        return self
    
    @classmethod
    def from_data(cls, data, phases):
        self = super().__new__(cls)
        self._set_phases(phases)
        if not isinstance(data, np.ndarray):
            data = np.array(data, float)
        if settings.debug:
            assert data.size == len(self._phases), ('size of data must be equal to '
                                                    'size of chemicals')
        self._data = data
        return self
    
    def _get_index(self, phases):
        if len(phases) == 1:
            return self._get_phase_index(phases)
        elif isinstance(phases, slice):
            return phases
        else:
            return self._get_phase_indices(phases)
    
    def __iter__(self):
        return self._data.__iter__()
    
    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs) 
        tab = tabs*4*" "
        phase_data = [f"{phase}={i:.4g}" for phase, i in zip(self.phases, self.data) if i]
        if len(phase_data) > 1 and tab:
            dlim = ",\n" + tab
            phase_data = "\n" + tab + dlim.join(phase_data)
        else:
            phase_data = ', '.join(phase_data)
        return f"{type(self).__name__}({phase_data})"
    
    def __repr__(self):
        return self.__format__()
    
    def _info(self, N):
        """Return string with all specifications."""
        phases = self.phases
        data = self.data
        phases, data = nonzeros(phases, data)
        phases = [f'({i})' for i in phases]
        len_ = len(data)
        if len_ == 0:
            return f"{type(self).__name__}: (empty)"
        elif self.units:
            basic_info = f"{type(self).__name__} ({self.units}):\n "
        else:
            basic_info = f"{type(self).__name__}:\n "
        new_line_spaces = ' '        
        data_info = ''
        lengths = [len(i) for i in phases]
        maxlen = max(lengths) + 1
        _N = N - 1
        for i in range(len_-1):
            spaces = ' ' * (maxlen - lengths[i])
            if i == _N:
                data_info += '...\n' + new_line_spaces
                break
            data_info += phases[i] + spaces + f' {data[i]:.4g}\n' + new_line_spaces
        spaces = ' ' * (maxlen - lengths[len_-1])
        data_info += phases[len_-1] + spaces + f' {data[len_-1]:.4g}'
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
            

class MaterialArray(ArrayEmulator):
    __slots__ = ('_phases', '_phase_index', '_phase_data', '_data', '_chemicals')
    _ChemicalArray = ChemicalArray
    _PhaseArray = PhaseArray
    
    def __new__(cls, phases=None, chemicals=None, **phase_data):
        self = cls.blank(phases or phase_data, chemicals)
        if phase_data:
            data = self._data
            chemical_indices = self._chemicals.indices
            phase_index = self._get_phase_index
            for phase, IDdata in phase_data.items():
                IDs, row = zip(*IDdata)
                data[phase_index(phase), chemical_indices(IDs)] = row
        return self
    
    def __reduce__(self):
        return self.from_data, (self._data, self._phases, self._chemicals)
    
    def phase_fraction(self):
        data = self._data
        phase_data = data.sum(0, keepdims=True)
        phase_data[phase_data == 0] = 1.
        return MaterialArray.from_data(data/phase_data, self._phases, self._chemicals)
    
    def chemical_fraction(self):
        data = self._data
        chemical_data = data.sum(1, keepdims=True)
        chemical_data[chemical_data == 0] = 1.
        return MaterialArray.from_data(data/chemical_data, self._phases, self._chemicals)
    
    _set_chemicals = ChemicalArray._set_chemicals
    _set_phases = PhaseArray._set_phases
    
    def __matmul__(self, other):
        return self._PhaseArray.from_data(self._data @ other, self._phases)
    
    def _copy_without_data(self):
        new = super().__new__(self.__class__)
        new._chemicals = self._chemicals
        new._phases = self._phases
        new._phase_index = self._phase_index
        return new
    
    @classmethod
    def blank(cls, phases, chemicals=None):
        self = super().__new__(cls)
        self._phase_data = None
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        shape = (len(self._phases), self._chemicals.size)
        self._data = np.zeros(shape, float)
        return self
    
    @classmethod
    def from_data(cls, data, phases, chemicals=None):
        self = super().__new__(cls)
        self._set_chemicals(chemicals)
        self._set_phases(phases)
        M_phases = len(self._phases)
        N_chemicals = self._chemicals.size
        if not isinstance(data, np.ndarray):
            data = np.array(data, float)
        if settings.debug:
            M, N = data.shape
            assert M == M_phases, ('number of phases must be equal to '
                                   'the number of data rows')
            assert N == N_chemicals, ('size of chemicals '
                                      'must be equal to '
                                      'number of data columns')
        self._data = data
        return self
    
    @property
    def phases(self):
        return self._phases
    @property
    def chemicals(self):
        return self._chemicals
    
    def phase_array(self):
        return self._PhaseArray.from_data(self._data.sum(1), self._phases)
    
    def chemical_array(self):
        return self._ChemicalArray.from_data(self._data.sum(0), self._chemicals)
    
    def get_phase(self, phase):
        phase_data = object.__new__(self._ChemicalArray)
        phase_data._data = self._data[self._phase_index[phase]]
        phase_data._chemicals = self._chemicals
        return phase_data
    
    def _get_index(self, phase_IDs):
        isa = isinstance
        if isa(phase_IDs, tuple):
            phases, IDs = phase_IDs
            if isa(IDs, str):
                IDs_index = self._chemicals.index(IDs)
            elif isa(IDs, slice) or IDs == ...:
                IDs_index = IDs
            else:
                IDs_index = self._chemicals.indices(IDs)
        else:
            phases = phase_IDs
            IDs_index = ...
        if isa(phases, slice) or phases == ...:
            phase_index = phases
        else:
            phase_index = self._get_phase_index(phases)
        return phase_index, IDs_index
    
    def _get_phase_index(self, phase):
        try:
            return self._phase_index[phase]
        except KeyError:
            raise UndefinedPhase(phase)
    
    def __iter__(self):
        if self._phase_data:
            return self._phase_data.__iter__()
        else:
            self._phase_data = iter = tuple(zip(self._phases, self._data))
            return iter.__iter__()
    
    def __format__(self, tabs="1"):
        IDs = self._chemicals.IDs
        phase_data = []
        for phase, data in self._phase_data:
            IDdata = ", ".join([f"('{ID}', {i:.4g})" for ID, i in zip(IDs, data) if i])
            if IDdata:
                phase_data.append(f"{phase}=[{IDdata}]")
        tabs = int(tabs)
        if tabs:
            tab = tabs*4*" "
            dlim = ",\n" + tab 
        else:
            dlim = ", "
        phase_data = dlim.join(phase_data)
        if self.phase_array().all():
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
        index, = np.where(self.chemical_array() != 0)
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
                    f' {data[i]:.4g}\n' + new_line_spaces
            spaces = ' ' * (maxlen - lengths[l-1])
            flowrates += (f'{IDs[l-1]} ' + spaces
                          + f' {data[l-1]:.4g}')

            # Put it together
            phases_flowrates_info += beginning + flowrates + '\n'
            
        return basic_info + phases_flowrates_info[:-1]
    show = ChemicalArray.show
    _ipython_display_ = show
    
def new_Array(name, units):
    ChemicalArraySubclass = type('Chemical' + name, (ChemicalArray,), {})
    PhaseArraySubclass = type('Phase' + name, (PhaseArray,), {})
    MaterialArraySubclass = type(name, (MaterialArray,), {})
    ChemicalArraySubclass.__slots__ = \
    PhaseArraySubclass.__slots__ = \
    MaterialArraySubclass.__slots__ = ()
    ChemicalArraySubclass.units = \
    PhaseArraySubclass.units = \
    MaterialArraySubclass.units = units
    MaterialArraySubclass._ChemicalArray = ChemicalArraySubclass
    MaterialArraySubclass._PhaseArray = PhaseArraySubclass
    return ChemicalArraySubclass, PhaseArraySubclass, MaterialArraySubclass
    
ChemicalMolarFlow, PhaseMolarFlow, MolarFlow = new_Array('MolarFlow', 'kmol/hr')
ChemicalMassFlow, PhaseMassFlow, MassFlow = new_Array('MassFlow', 'kg/hr')
ChemicalVolumetricFlow, PhaseVolumetricFlow, VolumetricFlow = new_Array('VolumetricFlow', 'm^3/hr')