# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
# DO NOT DELETE:
# chemicals' numba feature does not offer any benefit to speed
# but it may be used later
# def use_numba_chemicals():
#     import chemicals
#     import chemicals.numba as numba
#     Module = type(chemicals)
#     try: numba.Lastovka_Shaw # First autoload with __getattr__
#     except: pass
#     def update_module(module, other,
#             ignored=chemicals.utils.numba_cache_blacklisted,
#         ):
#         dct = module.__dict__
#         isa = isinstance
#         isfunc = callable
#         for i, j in other.__dict__.items():
#             if i not in dct or i in ignored: continue
#             if isa(j, Module) and 'chemicals.' in j.__name__:
#                 update_module(dct[i], j)
#             else:
#                 if isfunc(j) and 'numba' in repr(j.__class__): dct[i] = j
#     update_module(chemicals, numba)
# use_numba_chemicals()
# del use_numba_chemicals
__version__ = "0.46.0"

from . import thermo
del thermo
from . import (
    constants,
    base,
    utils,
    chemicals,
    exceptions,
    functional,
    units_of_measure,
    separations,
    functors,
)
from .utils import docround
from .network import *
from ._chemical_data import ChemicalData
from ._chemical import Chemical
from ._chemicals import Chemicals, CompiledChemicals
from ._thermal_condition import ThermalCondition
from ._thermo import Thermo, IdealThermo
from ._settings import settings, ProcessSettings
from ._thermo_data import ThermoData
from . import (
    indexer,
    reaction,
    equilibrium,
    mixture,
    network,
)
from ._stream import Stream
from ._heat_and_power import Heat, Power
from ._multi_stream import MultiStream
from .base import functor
from .reaction import *
from .equilibrium import * 
from .mixture import *
from ._preferences import preferences

__all__ = ('Chemical', 'ChemicalData', 'Chemicals', 'CompiledChemicals', 'Thermo', 
           'IdealThermo', 'Stream', 'MultiStream', 'Heat', 'Power', 'ThermalCondition', 'ProcessSettings',
           'mixture', 'ThermoData', *reaction.__all__, *equilibrium.__all__,  *mixture.__all__,
           *network.__all__, 'preferences',
           'indexer', 'settings', 'functor', 'functors', 'chemicals', 'base', 
           'equilibrium', 'units_of_measure', 'exceptions', 'functional', 
           'reaction', 'constants', 'utils', 'separations')

# Set number of digits displayed
import numpy as np
import pandas as pd
np.set_printoptions(suppress=False)
np.set_printoptions(precision=3) 
pd.options.display.float_format = '{:.3g}'.format
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('max_colwidth', 35)
pd.set_option('display.width', 160)
del np, pd

