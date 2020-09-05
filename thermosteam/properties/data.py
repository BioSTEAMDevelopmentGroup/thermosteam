# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:03:14 2020

@author: yoelr
"""
import os
import json

# %% Readers

class RowData:
    """
    Create a RowData object for fast data frame lookups by row.
    
    """
    __slots__ = ('index', 'values', 'columns')
    
    def __init__(self, index, values=None, columns=None):
        self.index = {key: i for i, key in enumerate(index)}
        self.values = values
        self.columns = columns if columns is None else {key: i for i, key in enumerate(columns)} 
    
    def get(self, row, col):
        columns = self.columns
        if not columns: raise RuntimeError('no columns implemented')
        return self.values[self.index[row], columns[col]]
    
    def __getitem__(self, row):
        return self.values[self.index[row]]
    
    def __contains__(self, row):
        return row in self.index   
    
def load_json(folder, json_file, hook=None):
    with open(os.path.join(folder, json_file)) as f:
        return json.loads(f.read(), object_hook=hook)
    
# %% Data

### Volume ###

from chemicals.volume import (
    rho_data_COSTALD, 
    rho_data_SNM0,
    rho_data_Perry_8E_105_l, rho_values_Perry_8E_105_l,
    rho_data_VDI_PPDS_2, rho_values_VDI_PPDS_2,
    rho_data_CRC_inorg_l, rho_values_CRC_inorg_l,
    rho_data_CRC_inorg_l_const,
    rho_data_CRC_inorg_s_const, 
    rho_data_CRC_virial, rho_values_CRC_virial,
)
from chemicals.miscdata import (
    VDI_saturation_dict,
    lookup_VDI_tabular_data,    
)

rho_data_COSTALD = RowData(rho_data_COSTALD.index, 
                           rho_data_COSTALD.values,
                           rho_data_COSTALD.columns)
rho_data_SNM0 = RowData(rho_data_SNM0.index, 
                        rho_data_SNM0.values,
                        rho_data_SNM0.columns)
rho_data_Perry_8E_105_l = RowData(rho_data_Perry_8E_105_l.index,
                                  rho_values_Perry_8E_105_l)
rho_data_VDI_PPDS_2 = RowData(rho_data_VDI_PPDS_2.index, 
                              rho_values_VDI_PPDS_2)
rho_data_CRC_inorg_l = RowData(rho_data_CRC_inorg_l.index, 
                               rho_values_CRC_inorg_l)
rho_data_CRC_inorg_l_const = RowData(rho_data_CRC_inorg_l_const.index, 
                                     rho_data_CRC_inorg_l_const.values,
                                     rho_data_CRC_inorg_l_const.columns)
rho_data_CRC_inorg_s_const = RowData(rho_data_CRC_inorg_s_const, 
                                     rho_data_CRC_inorg_s_const.values,
                                     rho_data_CRC_inorg_s_const.columns)
rho_data_CRC_virial = RowData(rho_data_CRC_virial.index, 
                              rho_values_CRC_virial)

### Phase change ###

from chemicals.phase_change import (
    phase_change_data_Perrys2_150, phase_change_values_Perrys2_150,
    phase_change_data_VDI_PPDS_4, phase_change_values_VDI_PPDS_4,
    phase_change_data_Alibakhshi_Cs, 
    Hvap_data_CRC,
    Hvap_data_Gharagheizi,
)

phase_change_data_Perrys2_150 = RowData(phase_change_data_Perrys2_150.index,
                                        phase_change_values_Perrys2_150)
phase_change_data_VDI_PPDS_4 = RowData(phase_change_data_VDI_PPDS_4.index, 
                                       phase_change_values_VDI_PPDS_4)
phase_change_data_Alibakhshi_Cs = RowData(phase_change_data_Alibakhshi_Cs.index,
                                          phase_change_data_Alibakhshi_Cs.values)
Hvap_data_CRC = RowData(Hvap_data_CRC.index, 
                        Hvap_data_CRC)
Hvap_data_Gharagheizi = RowData(Hvap_data_Gharagheizi.index, 
                                Hvap_data_Gharagheizi.values)

### Heat capacity ###

from chemicals.heat_capacity import (
    Cp_data_Poling,
    Cp_values_Poling,
    TRC_gas_data,
    TRC_gas_values,
    CRC_standard_data,
    Cp_dict_PerryI,
    zabransky_dict_sat_s,
    zabransky_dict_sat_p,
    zabransky_dict_const_s,
    zabransky_dict_const_p,
    zabransky_dict_iso_s,
    zabransky_dict_iso_p,
)

Cp_data_Poling = RowData(Cp_data_Poling.index, Cp_values_Poling)
TRC_gas_data = RowData(TRC_gas_data.index, TRC_gas_values)
CRC_standard_data = RowData(CRC_standard_data)

### Permitivity ###

from chemicals.permittivity import (
    permittivity_data_CRC, permittivity_values_CRC
)

permittivity_data_CRC = RowData(permittivity_data_CRC.index,
                                permittivity_values_CRC)

### Surface tension ###

from chemicals.interface import (
    sigma_data_Mulero_Cachadina, sigma_values_Mulero_Cachadina,
    sigma_data_Jasper_Lange, sigma_values_Jasper_Lange,
    sigma_data_Somayajulu, sigma_values_Somayajulu, sigma_data_Somayajulu2,
    sigma_values_Somayajulu2, sigma_data_VDI_PPDS_11, sigma_values_VDI_PPDS_11
)

sigma_data_Mulero_Cachadina = RowData(sigma_data_Mulero_Cachadina.index, sigma_values_Mulero_Cachadina)
sigma_data_Jasper_Lange = RowData(sigma_data_Jasper_Lange.index, sigma_values_Jasper_Lange,)
sigma_data_Somayajulu = RowData(sigma_data_Somayajulu.index, sigma_values_Somayajulu)
sigma_data_Somayajulu2 = RowData(sigma_data_Somayajulu2.index, sigma_values_Somayajulu2)
sigma_data_VDI_PPDS_11 = RowData(sigma_data_VDI_PPDS_11.index, sigma_values_VDI_PPDS_11)