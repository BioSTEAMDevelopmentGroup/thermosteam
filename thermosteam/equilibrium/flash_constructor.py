# -*- coding: utf-8 -*-
"""
"""
import thermosteam as tmo
from thermosteam.utils.decorators import chemicals_user
from thermo.flash import FlashPureVLS, FlashVLN, FlashVL
from thermo.bulk import default_settings
from thermo.phases import GibbsExcessLiquid, CEOSGas, VirialGas, VirialCorrelationsPitzerCurl
from thermo.eos_mix import PRMIX
from thermo.unifac import UNIFAC

__all__ = (
    'FlashConstructor',
)

@chemicals_user
class FlashConstructor:
    """
    Create a FlashConstructor object that predefines flash algorithms
    for easier creation of thermo Flash and Phase objects.
    
    Parameters
    ----------
    G : Phase subclass
        Class create gas phase object.
    L : Phase subclass
        Class create liquid phase object.
    S : Phase subclass
        Class create solid phase object.
    GE : GibbsExcessModel subclass
        Class create GibbsExcessModel object.
    Gkw : dict
        Key word arguments to initialize `G`.
    Lkw : Phase subclass
        Key word arguments to initialize `L`.
    Skw : Phase subclass
        Key word arguments to initialize `S`.
    GEkw : GibbsExcessModel subclass
        Key word arguments to initialize `GE`.
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>`, optional
        Object containing settings for calculating bulk and transport
        properties, [-]
    chemicals : :obj:`Chemicals <thermosteam.chemicals.Chemicals>`, optional
        Chemicals available to flash. Defaults to thermosteam.settings.get_chemicals()
    
    Examples
    --------
    >>> import thermo as tm
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Hexanol'])
    >>> flashpkg = tmo.equilibrium.FlashConstructor(
    ...     G=tm.CEOSGas, L=tm.GibbsExcessLiquid, S=tm.GibbsExcessSolid,
    ...     GE=tm.UNIFAC, GEkw=dict(version=1), Gkw=dict(eos_class=tm.PRMIX),
    ... )
    >>> flasher = flashpkg.flasher(N_liquid=2)
    >>> type(flasher).__name__
    'FlashVLN'
    >>> PT = flasher.flash(zs=[0.3, 0.2, 0.5], T=330, P=101325)
    >>> (PT.VF, PT.betas, PT.liquid0.zs, PT.H())
    (0.0,
     [0.027302, 0.972697],
     [0.943926, 0.051944, 0.004129],
     -46646.90)
    >>> flasher = flashpkg.flasher(['Water', 'Ethanol'], N_liquid=1)
    >>> type(flasher).__name__
    'FlashVL'
    >>> PT = flasher.flash(zs=[0.5, 0.5], T=353, P=101325)
    >>> (PT.VF, PT.gas.zs, PT.H())
    (0.312, [0.363, 0.636], -25473.987)
    >>> flasher = flashpkg.flasher(['Ethanol'])
    >>> type(flasher).__name__
    'FlashPureVLS'
    >>> PT = flasher.flash(T=353, P=101325)
    >>> (PT.VF, PT.gas.zs, PT.H())
    (1.0, [1.0], 3619.78)
    
    """
    __slots__ = (
        '_chemicals', 'settings',
        'G', 'Gkw',
        'L', 'Lkw', 
        'S', 'Skw',
        'GE', 'GEkw',
    )
    
    def __init__(self, G, L, S=None, GE=None,
                 Gkw=None, Lkw=None, Skw=None, GEkw=None,
                 settings=None, chemicals=None):
        self._load_chemicals(chemicals)
        self.G = G
        self.L = L
        self.S = S
        self.GE = GE
        self.Gkw = Gkw or {}
        self.Lkw = Lkw or {}
        self.Skw = Skw or {}
        self.GEkw = GEkw or {}
        self.settings = settings or default_settings
        
    @property
    def data(self):
        return tmo.ChemicalData(self.chemicals)
    
    def flasher(self, IDs=None, N_liquid=None, N_solid=None):
        return self._flash_from_data(
            self.data[IDs] if IDs else self.data,
            N_liquid, N_solid
        )

    def _solid_from_data(self, data):
        raise NotImplementedError("this method is not implemented yet")
        
    def _liquid_from_data(self, data):
        if self.GE:
            GE = self.GE.from_data(data, **self.GEkw)
            return self.L.from_data(data, GibbsExcessModel=GE, **self.Lkw)
        else:
            return self.L.from_data(data, **self.Lkw)

    def _gas_from_data(self, data):
        return self.G.from_data(data, **self.Gkw)

    def _flash_from_data(self, data, N_liquid, N_solid):
        N = len(data.CASs)
        if N_solid is None: N_solid = 0
        if N_liquid is None: N_liquid = 1
        if N == 0:
            raise ValueError(
                "IDs cannot be empty; at least one component ID must be given"
            )
        elif N == 1: # Pure component
            return FlashPureVLS(
                data, 
                data,
                self._gas_from_data(data),
                [self._liquid_from_data(data)
                  for i in range(N_liquid)],
                [self._solid_from_data(data)
                  for i in range(N_solid)],
                self.settings,
            )
        elif N_liquid == 1:
            if N_solid:
                raise NotImplementedError(
                    'multi-component flasher with solid phases '
                    'not implemented (yet)'
                )
            return FlashVL(
                data, 
                data,
                self._gas_from_data(data),
                self._liquid_from_data(data),
                self.settings,
            )
        else:
            if N_solid:
                raise NotImplementedError(
                    'multi-component flasher with solid phases '
                    'not implemented (yet)'
                )
            return FlashVLN(
                data, 
                data,
                [self._liquid_from_data(data)
                  for i in range(N_liquid)],
                self._gas_from_data(data),
                [],
                self.settings,
            )
 
def constructor(cls):
    return lambda f: setattr(cls, f.__name__, classmethod(f))
        
@constructor(GibbsExcessLiquid)
def from_data(cls, data,
            GibbsExcessModel=None,
            use_Hvap_caloric=False,
            use_Poynting=False,
            use_phis_sat=False,
            use_Tait=False,
            use_eos_volume=False,
            Psat_extrapolation='AB',
            equilibrium_basis=None,
            caloric_basis=None):
    return cls(
        data.VaporPressures, 
        VolumeLiquids=data.VolumeLiquids,
        HeatCapacityGases=data.HeatCapacityGases,
        GibbsExcessModel=GibbsExcessModel,
        eos_pure_instances=[i.eos for i in data.VolumeGases],
        EnthalpyVaporizations=data.EnthalpyVaporizations,
        HeatCapacityLiquids=data.HeatCapacityLiquids,
        use_Hvap_caloric=use_Hvap_caloric,
        use_Poynting=use_Poynting,
        use_phis_sat=use_phis_sat,
        use_Tait=use_Tait,
        use_eos_volume=use_eos_volume,
        equilibrium_basis=equilibrium_basis,
        caloric_basis=caloric_basis
    )

@constructor(CEOSGas)
def from_data(cls, data, eos_class=None):
    from thermo.interaction_parameters import IPDB
    if eos_class is None: eos_class = PRMIX
    kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', data.CASs, 'kij')
    eos_kwargs = dict(Tcs=data.Tcs,
                      Pcs=data.Pcs,
                      omegas=data.omegas,
                      kijs=kijs)
    return cls(eos_class, eos_kwargs, data.HeatCapacityGases)

@constructor(VirialGas)
def from_data(cls, data, model=None):
    if model is None:
        model = VirialCorrelationsPitzerCurl(data.Tcs, data.Pcs, data.omegas)
    return cls(model, HeatCapacityGases=data.HeatCapacityGases)

@constructor(UNIFAC)
def from_data(cls, data, version=1):
    if version == 0:
        chemgroups = data.UNIFAC_groups
    elif version == 1:
        chemgroups = data.UNIFAC_Dortmund_groups
    elif version == 2:
        chemgroups = data.PSRK_groups
    else:
        raise RuntimeError('chemgroups for version %d not yet implemented' %version)
    N = len(chemgroups)
    return cls.from_subgroups(298.15, N*[1./N], chemgroups, version=version)