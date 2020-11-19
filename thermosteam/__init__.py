# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from . import (constants,
               base,
               utils,
               chemicals,
               exceptions,
               functional,
               units_of_measure,
               separations,
               functors,
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
from .reaction import *
from flexsolve import speed_up

__all__ = ('Chemical', 'Chemicals', 'CompiledChemicals', 'Thermo', 'Stream',
           'MultiStream', 'ThermalCondition', 'mixture', 'ThermoData',
           *reaction.__all__, 'indexer', 'settings', 'functor', 'functors', 
           'chemicals', 'base', 'equilibrium', 'units_of_measure', 'exceptions',
           'functional', 'reaction', 'constants', 'utils', 'separations', 'speed_up')

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

__version__ = "0.22.1"