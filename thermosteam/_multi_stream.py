# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from ._stream import Stream
from ._thermal_condition import ThermalCondition
from .indexer import MolarFlowIndexer
from ._settings import settings
from .equilibrium import VLE
from .utils import Cache, assert_same_chemicals
import numpy as np

__all__ = ('MultiStream', )

class MultiStream(Stream):
    """Create a Stream object that defines material flow rates along with its thermodynamic state. Thermodynamic and transport properties of a stream are available as properties, while thermodynamic equilbrium (e.g. VLE, and bubble and dew points) are available as methods. 

    Parameters
    ----------
    ID='' : str, defaults to a unique ID
        A unique identification. If ID is None, stream will not be
        registered.

    flow=() : 2d array, optional
        All flow rates corresponding to `phases` by row and chemical IDs by column.

    thermo=() : Thermo, defaults to settings.Thermo
        Thermodynamic equilibrium package.

    units='kmol/hr' : str, optional
        Flow rate units of measure (only mass, molar, and
        volumetric flow rates are valid)

    phases='gl' : tuple['g', 'l', 's', 'G', 'L', 'S'], optional
        Tuple denoting the phases present.

    T=298.15 : float, optional
        Temperature (K).

    P=101325 : float, optional
        Pressure (Pa).

    price=0 : float, optional
        Price in USD/kg.
    
    **phase_flow : tuple[str, float]
        phase-(ID, flow) pairs
    
    """
    __slots__ = ()
    def __init__(self, ID="", flow=(), T=298.15, P=101325., phases='gl', units=None,
                 thermo=None, price=None, **phase_flows):
        self._TP = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._init_indexer(flow, phases, thermo.chemicals, phase_flows)
        self.price = price
        if units:
            name, factor = self._get_flow_name_and_factor(units)
            flow = getattr(self, 'i' + name)
            flow.data[:] = self._imol._data * factor
        self._sink = self._source = None
        self._init_cache()
        self._register(ID)
        
    def _init_indexer(self, flow, phases, chemicals, phase_flows):
        if flow is ():
            if phase_flows:
                imol = MolarFlowIndexer(phases, chemicals=chemicals, **phase_flows)
            else:
                imol = MolarFlowIndexer.blank(phases, chemicals)
        else:
            assert not phase_flows, ("may specify either 'flow' or "
                                    "'phase_flows', but not both")
            if isinstance(flow, MolarFlowIndexer):
                imol = flow
            else:
                imol = MolarFlowIndexer.from_data(flow, phases, chemicals)
        self._imol = imol
        
    def _init_cache(self):
        super()._init_cache()
        self._streams = {}
        self._vle_cache = Cache(VLE, self._imol, self._TP, thermo=self._thermo,
                                bubble_point_cache=self._bubble_point_cache,
                                dew_point_cache=self._dew_point_cache)
        
    def __getitem__(self, phase):
        streams = self._streams
        if phase in streams:
            stream = streams[phase]
        else:
            stream = Stream.__new__(Stream)
            stream._imol = self._imol.get_phase(phase)
            stream._ID = None
            stream._TP = self._TP
            stream._thermo = self._thermo
            streams[phase] = stream
        return stream
    
    def __iter__(self):
        for i in self._imol._phases: yield self[i]
    
    ### Property getters ###
    
    def get_flow(self, units, phase, IDs=...):
        name, factor = self._get_flow_name_and_factor(units)
        indexer = getattr(self, 'i' + name)
        return factor * indexer[phase, IDs]
    
    def set_flow(self, data, units, phase, IDs=...):
        name, factor = self._get_flow_name_and_factor(units)
        indexer = getattr(self, 'i' + name)
        indexer[phase, IDs] = np.asarray(data, dtype=float) / factor
    
    ### Stream data ###
    
    @property
    def phases(self):
        return self._imol._phases
    @phases.setter
    def phases(self, phases):
        phases = sorted(phases)
        if phases != self.phases:
            self._imol = self._imol.to_material_array(phases)
            self._vle_cache = Cache(VLE, self._imol, self._TP, thermo=self._thermo)
    
    ### Flow properties ###
            
    @property
    def mol(self):
        mol = self._imol._data.sum(0)
        mol.setflags(0)
        return mol
    @property
    def mass(self):
        mass = self.mol * self.chemicals.MW
        mass.setflags(0)
        return mass
    @property
    def vol(self):
        vol = self.ivol._data.sum(0)
        vol.setflags(0)
        return vol
    
    ### Net flow properties ###
    
    @property
    def H(self):
        return self.mixture.xH_at_TP(self._imol.iter_data(), self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self._imol.iter_data(), H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.xS_at_TP(self._imol.iter_data(), self._TP)
    @property
    def C(self):
        return self.mixture.xCn_at_TP(self._imol.iter_data(), self._TP)
    @property
    def F_vol(self):
        return 1000. * self.mixture.xV_at_TP(self._imol.iter_data(), self._TP)
    @F_vol.setter
    def F_vol(self, value):
        F_vol = self.F_vol
        if not F_vol: raise AttributeError("undefined composition; cannot set flow rate")
        self.ivol._data[:] *= value / F_vol / 1000.
    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self._imol['l'], self._TP)
    
    ### Composition properties ###
    
    @property
    def V(self):
        return self.mixture.xV_at_TP(self._imol.iter_composition(), self._TP)
    @property
    def kappa(self):
        return self.mixture.xkappa_at_TP(self._imol.iter_composition(), self._TP)        
    @property
    def Cn(self):
        return self.mixture.xCn_at_TP(self._imol.iter_composition(), self._TP)
    @property
    def mu(self):
        return self.mixture.xmu_at_TP(self._imol.iter_composition(), self._TP)

    @property
    def sigma(self):
        mol = self._imol['l']
        return self.mixture.sigma_at_TP(mol / mol.sum(), self._TP)
    @property
    def epsilon(self):
        mol = self._imol['l']
        return self.mixture.epsilon_at_TP(mol / mol.sum(), self._TP)
        
    ### Methods ###
        
    def copy_flow(self, stream, phase=..., IDs=..., *, remove=False, exclude=False):
        """Copy flow rates of stream to self.
        
        Parameters
        ----------
        stream : Stream
            Flow rates will be copied from here.
        IDs=None : iterable[str], defaults to all chemicals.
            Chemical IDs. 
        remove=False: bool, optional
            If True, copied chemicals will be removed from `stream`.
        exclude=False: bool, optional
            If True, exclude designated chemicals when copying.
        
        """
        chemicals = self.chemicals
        if exclude:
            IDs = chemicals.get_index(IDs)
            index = np.ones(chemicals.size, dtype=bool)
            index[IDs] = False
        
        if isinstance(stream, MultiStream):
            stream_phase_imol = stream.imol[phase]
            self_phase_imol = self.imol[phase]
            self_phase_imol[index] = stream_phase_imol[index]
            if remove: 
                stream_phase_imol[index] = 0
        elif stream.phase == phase:
            stream_imol = stream.imol
            self_phase_imol = self.imol[phase]
            self_phase_imol[index] = stream_imol[index]
            if remove: 
                stream_imol[index] = 0
        else:
            self_phase_imol = self.imol[phase]
            self_phase_imol[index] = 0
    
    def get_normalized_mol(self, IDs):
        z = self.imol[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_normalized_vol(self, IDs):
        z = self.ivol[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_normalized_mass(self, IDs):
        z = self.imass[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        multi = []; single = []; isa = isinstance
        for i in others:
            (multi if isa(i, MultiStream) else single).append(i)
        self.empty()
        for i in single:
            self._imol[i.phase] += i.mol    
        self._imol._data[:] += sum([i._imol._data for i in multi])
        self.H = sum([i.H for i in others])
        
    def link_with(self, other):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        self._TP = other._TP
        self._imol._data = other._imol._data
        self._streams = other._streams
        self._vle_cache = other._vle_cache
        self._dew_point_cache = other._dew_point_cache
        self._bubble_point_cache = other._bubble_point_cache
        self._imol._data_cache = other._imol._data_cache
    
    ### Equilibrium ###
    
    @property
    def vle(self):
        return self._vle_cache.retrieve()
    
    ### Casting ###
    
    @property
    def phase(self):
        return "".join(self._imol._phases)
    @phase.setter
    def phase(self, phase):
        assert len(phase) == 1, f'invalid phase {repr(phase)}'
        self.__class__ = Stream
        self._imol = self._imol.to_chemical_indexer(phase)
        
    ### Representation ###
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        from .indexer import nonzeros
        IDs = self.chemicals.IDs
        basic_info = self._basic_info()
        all_IDs, _ = nonzeros(self.chemicals.IDs, self.mol)
        all_IDs = tuple(all_IDs)
        display_units = self.display_units
        T_units = T or display_units.T
        P_units = P or display_units.P
        flow_units = flow or display_units.flow
        N = N or display_units.N
        basic_info += Stream._info_phaseTP(self, self.phases, T_units, P_units)
        len_ = len(all_IDs)
        if len_ == 0:
            return basic_info + ' flow: 0' 

        # Length of chemical column
        all_lengths = [len(i) for i in all_IDs]
        maxlen = max(all_lengths + [8]) 

        name, factor = self._get_flow_name_and_factor(flow_units)
        indexer = getattr(self, 'i' + name)
        first_line = f' flow ({flow_units}):'
        first_line_spaces = len(first_line)*" "

        # Set up chemical data for all phases
        phases_flowrates_info = ''
        for phase in self.phases:
            phase_data = factor * indexer[phase, all_IDs] 
            IDs, data = nonzeros(all_IDs, phase_data)
            if not IDs: continue
        
            # Get basic structure for phase data
            
            beginning = (first_line or first_line_spaces) + f' ({phase}) '
            first_line = False
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
    
    def print(self):
        from .utils import repr_kwarg, repr_couples
        IDs = self.chemicals.IDs
        phase_data = []
        for phase, data in self.imol.iter_data():
            IDdata = repr_couples(", ", IDs, data)
            if IDdata:
                phase_data.append(f"{phase}=[{IDdata}]")
        dlim = ", "
        phase_data = dlim.join(phase_data)
        phases = f'phases={self.phases}'
        if phase_data:
            phase_data = dlim + phase_data
        price = repr_kwarg('price', self.price)
        print(f"{type(self).__name__}(ID={repr(self.ID)}, {phases}, T={self.T:.2f}, "
              f"P={self.P:.6g}{price}{phase_data})")
    