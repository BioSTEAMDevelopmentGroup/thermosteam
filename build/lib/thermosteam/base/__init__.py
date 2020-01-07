# -*- coding: utf-8 -*-

from . import functor
from . import thermo_model
from . import thermo_model_handle
from . import handle_builder
from . import phase_property
from . import units_of_measure
from . import documenter

__all__ = (*functor.__all__,
           *thermo_model.__all__,
           *thermo_model_handle.__all__,
           *handle_builder.__all__,
           *phase_property.__all__,
           *units_of_measure.__all__,
           *documenter.__all__)

from .functor import *
from .thermo_model import *
from .thermo_model_handle import *
from .handle_builder import *
from .phase_property import *
from .units_of_measure import *
from .documenter import *

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