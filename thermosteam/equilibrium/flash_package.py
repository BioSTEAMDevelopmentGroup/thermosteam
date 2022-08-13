# -*- coding: utf-8 -*-
"""
"""
from thermo.interaction_parameters import IPDB
import thermosteam as tmo
from thermo.bulk import default_settings
import thermo as tm

__all__ = (
    'FlashPackage',
)

class FlashPackage:
    """
    Create a FlashPackage object that predefines flash algorithms
    for easier creation of thermo Flash and Phase objects.
    
    Parameters
    ----------
    G : :obj:`Phase subclass <thermo.Phase>`, optional
        Class to create gas phase object. Defaults to `CEOSGas <thermo.CEOSGas>`.
    L : :obj:`Phase subclass <thermo.Phase>`, optional
        Class to create liquid phase object. Defaults to :obj:`GibbsExcessLiquid <thermo.GibbsExcessLiquid>`.
    S : :obj:`Phase subclass <thermo.Phase>`, optional
        Class to create solid phase object. Defaults to :obj:`GibbsExcessSolid <thermo.GibbsExcessSolid>`.
    GE : :obj:`GibbsExcessModel subclass <thermo.Phase>`, optional
        Class to create GibbsExcessModel object. Defaults to :obj:`UNIFAC <thermo.UNIFAC>`.
    Gkw : dict, optional
        Key word arguments to initialize `G`.
    Lkw : dict, optional
        Key word arguments to initialize `L`.
    Skw : dict, optional
        Key word arguments to initialize `S`.
    GEkw : dict, optional
        Key word arguments to initialize `GE`.
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>`, optional
        Object containing settings for calculating bulk and transport
        properties, [-]
    chemicals : :obj:`Chemicals <thermosteam.chemicals.Chemicals>`, optional
        Chemicals available to flash. Defaults to thermosteam.settings.get_chemicals()
    N_liquid : int, optional
        Default number of liquid phases. Defaults to 1.
    N_solid : int, optional
        Default number of solid phases. Defaults to 0.
        
    Examples
    --------
    >>> import thermo as tm
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Hexane'])
    >>> # VLLE using activity coefficients for the liquid phase 
    >>> # and equations of state for the gas phase.
    >>> flashpkg = tmo.equilibrium.FlashPackage(
    ...     G=tm.CEOSGas, L=tm.GibbsExcessLiquid, S=tm.GibbsExcessSolid,
    ...     GE=tm.UNIFAC, GEkw=dict(version=1), Gkw=dict(eos_class=tm.PRMIX),
    ... )
    >>> flasher = flashpkg.flasher(N_liquid=2)
    >>> type(flasher).__name__
    'FlashVLN'
    >>> PT = flasher.flash(zs=[0.3, 0.2, 0.5], T=330, P=101325)
    >>> tmo.docround([PT.VF, PT.betas, PT.liquid0.zs, PT.H()]) # doctest: +SKIP
    [0.752, [0.752, 0.248], [0.7617, 0.2337, 0.0046], -7644.4997]
    >>> # VLE using activity coefficients for the liquid phase 
    >>> # and equations of state for the gas phase.
    >>> flasher = flashpkg.flasher(['Water', 'Ethanol'], N_liquid=1)
    >>> type(flasher).__name__
    'FlashVL'
    >>> PT = flasher.flash(zs=[0.5, 0.5], T=353, P=101325)
    >>> tmo.docround([PT.VF, PT.gas.zs, PT.H()]) # doctest: +SKIP
    [0.32, [0.3644, 0.6356], -25179.1688]
    >>> # Single component equilibrium.
    >>> flasher = flashpkg.flasher(['Ethanol']) 
    >>> type(flasher).__name__
    'FlashPureVLS'
    >>> PT = flasher.flash(T=353, P=101325)
    >>> tmo.docround([PT.VF, PT.gas.zs, PT.H()]) # doctest: +SKIP
    [1.0, [1.0], 3619.7887]
    >>> # VLLE using virial equation of state for the gas phase
    >>> flashpkg.G, flashpkg.Gkw = tm.VirialGas, {}
    >>> flasher = flashpkg.flasher(N_liquid=2)
    >>> PT = flasher.flash(zs=[0.45, 0.05, 0.5], T=330, P=101325)
    >>> tmo.docround([PT.VF, PT.betas, PT.liquid0.zs, PT.H(), PT.phase_count]) # doctest: +SKIP
    [0.0, [0.4754, 0.5246], [0.9183, 0.0808, 0.0008], -33426.8214, 2]
    >>> # VLLE using SRK equation of state for the gas phase
    >>> flashpkg.G, flashpkg.Gkw = tm.CEOSGas, {'eos_class': tm.SRKMIX}
    >>> flasher = flashpkg.flasher(N_liquid=2)
    >>> PT = flasher.flash(zs=[0.45, 0.05, 0.5], T=330, P=101325)
    >>> tmo.docround([PT.VF, PT.betas, PT.liquid0.zs, PT.H(), PT.phase_count]) # doctest: +SKIP
    [0.0, [0.4754, 0.5246], [0.9183, 0.0808, 0.0008], -33426.8213, 2]
    >>> # VLLE using ideal gas
    >>> flashpkg.G, flashpkg.Gkw = tm.IdealGas, {}
    >>> flasher = flashpkg.flasher(N_liquid=2)
    >>> PT = flasher.flash(zs=[0.45, 0.05, 0.5], T=330, P=101325)
    >>> tmo.docround([PT.VF, PT.betas, PT.liquid0.zs, PT.H()]) # doctest: +SKIP
    [0.0, [0.4754, 0.5246], [0.9183, 0.0808, 0.0008], -33426.8213]
    
    """
    __slots__ = (
        'chemicals', 'settings',
        'G', 'Gkw',
        'L', 'Lkw', 
        'S', 'Skw',
        'GE', 'GEkw',
        'N_liquid', 'N_solid',
    )
    
    def __init__(self, G=tm.CEOSGas, L=tm.GibbsExcessLiquid, S=tm.GibbsExcessSolid,
                 GE=tm.UNIFAC, Gkw=None, Lkw=None, Skw=None, GEkw=None, 
                 settings=None, chemicals=None, N_liquid=None, N_solid=None):
        self.chemicals = tmo.settings.get_default_chemicals(chemicals)
        self.G = G
        self.L = L
        self.S = S
        self.GE = GE
        self.Gkw = Gkw or {}
        self.Lkw = Lkw or {}
        self.Skw = Skw or {}
        self.GEkw = GEkw or {}
        self.settings = settings or default_settings
        self.N_liquid = N_liquid or 2
        self.N_solid = N_solid or 0
        
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
        if self.L is tm.GibbsExcessLiquid and self.GE:
            GE = self.GE.from_data(data, **self.GEkw)
            return self.L.from_data(data, GibbsExcessModel=GE, **self.Lkw)
        else:
            return self.L.from_data(data, **self.Lkw)

    def _gas_from_data(self, data):
        return self.G.from_data(data, **self.Gkw)

    def _flash_from_data(self, data, N_liquid, N_solid):
        N = len(data.CASs)
        if N_solid is None: N_solid = self.N_solid
        if N_liquid is None: N_liquid = self.N_liquid
        if N == 0:
            raise ValueError(
                "IDs cannot be empty; at least one component ID must be given"
            )
        elif N == 1: # Pure component
            return tm.FlashPureVLS(
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
            return tm.FlashVL(
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
            return tm.FlashVLN(
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
        
@constructor(tm.GibbsExcessLiquid)
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

@constructor(tm.CEOSGas)
def from_data(cls, data, eos_class=None):
    if eos_class is None: eos_class = tm.PRMIX
    eos_kwargs = dict(Tcs=data.Tcs,
                      Pcs=data.Pcs,
                      omegas=data.omegas)
    try:
        eos_kwargs['kijs'] = IPDB.get_ip_asymmetric_matrix('ChemSep PR', data.CASs, 'kij')
    except:
        pass        
    return cls(eos_class, eos_kwargs, data.HeatCapacityGases)

tm.VirialGas.model_attributes = ('HeatCapacityGases', 'model') # TODO: Add attribute in Caleb's thermo
@constructor(tm.VirialGas)
def from_data(cls, data, model=None):
    if model is None:
        model = tm.VirialCSP(Tcs=data.Tcs, Pcs=data.Pcs, Vcs=data.Vcs, omegas=data.omegas)
    return cls(model, HeatCapacityGases=data.HeatCapacityGases)

@constructor(tm.IdealGas)
def from_data(cls, data):
    return cls(HeatCapacityGases=data.HeatCapacityGases)

@constructor(tm.UNIFAC)
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
