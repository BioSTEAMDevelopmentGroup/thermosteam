# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:29:58 2019

@author: yoelr
"""
from . import (base,
               utils,
               properties,
               exceptions,
               functional,
               units_of_measure,
)
from ._chemical import Chemical
from ._chemicals import Chemicals, CompiledChemicals
from ._thermal_condition import ThermalCondition
from . import mixture
from ._thermo import Thermo
from ._settings import settings
from ._thermo_data import ThermoData
from . import (indexer,
               reaction,
               equilibrium
)
from ._stream import Stream
from ._multi_stream import MultiStream
from .base import functor
from flexsolve import speed_up

__all__ = ('Chemical', 'Chemicals', 'CompiledChemicals', 'Thermo', 'indexer',
           'Stream', 'MultiStream', 'ThermalCondition', 'mixture', 'ThermoData',
           'settings', 'functor', 'properties', 'base', 'equilibrium',
           'units_of_measure', 'exceptions', 'functional', 'reaction',
           'utils', 'speed_up')

# Set number of digits displayed
import numpy as np
import pandas as pd
np.set_printoptions(suppress=False)
np.set_printoptions(precision=3) 
pd.options.display.float_format = '{:.3g}'.format
pd.set_option('display.max_rows', 35)
pd.set_option('display.max_columns', 10)
pd.set_option('max_colwidth', 35)
del np, pd