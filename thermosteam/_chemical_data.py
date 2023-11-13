# -*- coding: utf-8 -*-
"""
"""
from chemicals.lennard_jones import Stockmayer, molecular_diameter
from thermo.unifac import UNIFAC_RQ, Van_der_Waals_volume, Van_der_Waals_area
from chemicals.utils import Parachor
from collections.abc import Iterable
import thermosteam as tmo

class CachedProperty:
    __slots__ = ('fget', 'key')
    
    def __new__(cls, fget):
        self = super().__new__(cls)
        self.fget = fget
        self.key = fget.__name__
        return self
    
    def __get__(self, data, _):
        cache = data.cache
        key = self.key
        if key in cache:
            value = cache[key]
        else:
            cache[key] = value = self.fget(data)
        return value


class ChemicalData:
    """
    Create a ChemicalData object that works as thermo ChemicalPropertiesPackage
    and ChemicalConstantsPackage objects.
    """
    __slots__ = ('N', 'chemicals', 'index', 'cmps', 'cache')
    
    def __init__(self, chemicals):
        if isinstance(chemicals, tmo.CompiledChemicals):
            self.chemicals = chemicals.tuple
            self.index = chemicals._index
            self.N = chemicals.size
        else:
            self.chemicals = tuple(chemicals)
            self.N = len(chemicals)
        self.cmps = range(self.N)
        self.cache = {}
    
    def __iter__(self):
        return self.chemicals.__iter__()
    
    def __getitem__(self, IDs):
        chemicals = self.chemicals
        try:
            index = self.index
        except:
            self.index = index = {}
            repeated_names = set()
            names = set()
            all_names_list = []
            for i in chemicals:
                if not i.iupac_name: i.iupac_name = ()
                all_names = set([*i.iupac_name, *i.aliases, i.common_name, i.formula])
                all_names_list.append(all_names)
                for name in all_names:
                    if not name: continue
                    if name in names:
                        repeated_names.add(name)
                    else:
                        names.add(name)
            for all_names, i in zip(all_names_list, chemicals):
                ID = i.ID
                for name in all_names:
                    if name and name not in repeated_names:
                        index[name] = index[ID]
        if isinstance(IDs, str):
            return chemicals[index[IDs]]
        elif isinstance(IDs, Iterable):
            return ChemicalData([chemicals[index[i]] for i in IDs])
        else:
            raise KeyError("key must be a string or a tuple, not a '{type(IDs).__name__}' object")
    
    @CachedProperty
    def EnthalpyVaporizations(self):
        return [i.Hvap for i in self.chemicals]
    @CachedProperty
    def VaporPressures(self):
        return [i.Psat for i in self.chemicals]
    @CachedProperty
    def SublimationPressures(self):
        return [i.Psub for i in self.chemicals]
    @CachedProperty
    def EnthalpySublimations(self):
        return [i.Hsub for i in self.chemicals]
    @CachedProperty
    def SurfaceTensions(self):
        return [i.sigma for i in self.chemicals]
    @CachedProperty
    def PermittivityLiquids(self):
        return [i.epsilon for i in self.chemicals]
    @CachedProperty
    def CASs(self):
        return [i.CAS for i in self.chemicals]
    @CachedProperty
    def MWs(self):
        return [i.MW for i in self.chemicals]
    @CachedProperty
    def Tms(self):
        return [i.Tm for i in self.chemicals]
    @CachedProperty
    def Tbs(self):
        return [i.Tb for i in self.chemicals]
    @CachedProperty
    def Tcs(self):
        return [i.Tc for i in self.chemicals]
    @CachedProperty
    def Pcs(self):
        return [i.Pc for i in self.chemicals]
    @CachedProperty
    def Vcs(self):
        return [i.Vc for i in self.chemicals]
    @CachedProperty
    def omegas(self):
        return [i.omega for i in self.chemicals]
    @CachedProperty
    def Zcs(self):
        return [i.Zc for i in self.chemicals]
    @CachedProperty
    def Hfus_Tms(self):
        return [i.Hfus for i in self.chemicals]   
    @CachedProperty
    def Tts(self):
        return [i.Tt for i in self.chemicals]
    @CachedProperty
    def Pts(self):
        return [i.Pt for i in self.chemicals]
    @CachedProperty
    def dipoles(self):
        return [i.dipole for i in self.chemicals]
    @CachedProperty
    def charges(self):
        return [i.charge for i in self.chemicals]
    @CachedProperty
    def UNIFAC_groups(self):
        return [i.UNIFAC for i in self.chemicals]
    @CachedProperty
    def UNIFAC_Dortmund_groups(self):
        return [i.Dortmund for i in self.chemicals]
    @CachedProperty
    def PSRK_groups(self):
        return [i.PSRK for i in self.chemicals]
    @CachedProperty
    def similarity_variables(self):
        return [i.similarity_variable for i in self.chemicals]
    @CachedProperty
    def StielPolars(self):
        return [i.Stiel_Polar for i in self.chemicals]
    @CachedProperty
    def VolumeGases(self):
        return [i.V if i._locked_state else i.V.g for i in self.chemicals]
    @CachedProperty
    def VolumeLiquids(self):
        return [i.V if i._locked_state else i.V.l for i in self.chemicals]
    @CachedProperty
    def VolumeSolids(self):
        return [i.V if i._locked_state else i.V.s for i in self.chemicals]
    @CachedProperty
    def HeatCapacityGases(self):
        return [i.Cn if i._locked_state else i.Cn.g for i in self.chemicals]
    @CachedProperty
    def HeatCapacityLiquids(self):
        return [i.Cn if i._locked_state else i.Cn.l for i in self.chemicals]
    @CachedProperty
    def HeatCapacitySolids(self):
        return [i.Cn if i._locked_state else i.Cn.s for i in self.chemicals]
    @CachedProperty
    def ViscosityGases(self):
        return [i.mu if i._locked_state else i.mu.g for i in self.chemicals]
    @CachedProperty
    def ViscosityLiquids(self):
        return [i.mu if i._locked_state else i.mu.l for i in self.chemicals]
    @CachedProperty
    def ThermalConductivityGases(self):
        return [i.k if i._locked_state else i.k.g for i in self.chemicals]
    @CachedProperty
    def ThermalConductivityLiquids(self):
        return [i.k if i._locked_state else i.k.l for i in self.chemicals]
    @CachedProperty
    def rhocs(self):
        return [None if Vc is None else 1.0/Vc for Vc in self.chemicals.Vcs]
    @CachedProperty
    def rhocs_mass(self):
        return [None if None in (rhoc, MW) else rhoc * 1e-3 * MW
                for rhoc, MW in zip(self.rhocs, self.MWs)]
    @CachedProperty
    def Hfus_Tms_mass(self):
        return [None if None in (Hfus, MW) else Hfus * 1000.0 / MW
                for Hfus, MW in zip(self.Hfus_Tms, self.MWs)]
    @CachedProperty
    def Hvap_Tbs(self):
        return [Hvap(Tb) for Hvap, Tb in zip(self.EnthalpyVaporizations, self.Tbs)]
    @CachedProperty
    def Hvap_Tbs_mass(self):
        return [None if None in (Hvap, MW) else Hvap*1000.0/MW
                for Hvap, MW in zip(self.Hvap_Tbs, self.Tbs)]
    @CachedProperty
    def Vmg_STPs(self):
        return [i.TP_dependent_property(298.15, 101325.0)
                for i in self.chemicals.VolumeGases]
    @CachedProperty
    def rhog_STPs(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vmg_STPs]
    @CachedProperty
    def rhog_STPs_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vmg_STPs, self.MWs)]
    @CachedProperty
    def Vml_Tbs(self):
        return [None if Tb is None else i.T_dependent_property(Tb) for i, Tb in zip(self.VolumeLiquids, self.Tbs)]
    @CachedProperty
    def Vml_Tms(self):
        return [None if Tm is None else i.T_dependent_property(Tm)
                for i, Tm in zip(self.VolumeLiquids, self.Tms)]
    @CachedProperty
    def Vml_STPs(self):
        return [i.T_dependent_property(298.15) for i in self.chemicals.VolumeLiquids]
    @CachedProperty
    def Vml_60Fs(self):
        return [i.T_dependent_property(288.7055555555555) for i in self.chemicals.VolumeLiquids]
    @CachedProperty
    def rhol_STPs(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vml_STPs]
    @CachedProperty
    def rhol_60Fs(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vml_60Fs]
    @CachedProperty
    def rhol_STPs_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vml_STPs, self.MWs)]
    @CachedProperty
    def rhol_60Fs_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vml_60Fs, self.MWs)]
    @CachedProperty
    def Hsub_Tts(self):
        return [None if Tt is None else i(Tt)
                for i, Tt in zip(self.EnthalpySublimations, self.Tts)]
    @CachedProperty
    def Hsub_Tts_mass(self):
        return [Hsub*1000.0/MW if Hsub is not None else None for Hsub, MW in zip(self.Hsub_Tts, self.MWs)]
    @CachedProperty
    def Stockmayers(self):
        Tms = self.Tms
        Tbs = self.Tbs
        Tcs = self.Tcs
        Zcs = self.Zcs
        omegas = self.omegas
        CASs = self.CASs
        return [Stockmayer(Tm=Tms[i], Tb=Tbs[i], Tc=Tcs[i], 
                           Zc=Zcs[i], omega=omegas[i], CASRN=CASs[i]) 
                for i in self.cmps]
    @CachedProperty
    def molecular_diameters(self):
        Tcs = self.Tcs
        Zcs = self.Zcs
        omegas = self.omegas
        CASs = self.CASs
        Vml_Tms = self.Vml_Tms
        Vml_Tb = self.Vml_Tb
        Pcs = self.Pcs
        Vcs = self.Vcs
        return [molecular_diameter(Tc=Tcs[i], Pc=Pcs[i], Vc=Vcs[i],
                                   Zc=Zcs[i], omega=omegas[i],
                                   Vm=Vml_Tms[i], Vb=Vml_Tb[i], 
                                   CASRN=CASs[i])
                for i in self.cmps]
    @CachedProperty
    def Psat_298s(self):
        return [i(298.15) for i in self.chemicals.VaporPressures]
    @CachedProperty
    def Hvap_298s(self):
        return [o.T_dependent_property(298.15) for o in self.chemicals.EnthalpyVaporizations]
    @CachedProperty
    def Hvap_298s_mass(self):
        return [Hvap*1000.0/MW if Hvap is not None else None for Hvap, MW in zip(self.Hvap_298s, self.MWs)]
    @CachedProperty
    def Vms_Tms(self):
        return [None if Tm is None else i.T_dependent_property(Tm) for i, Tm in zip(self.VolumeSolids, self.Tms)]
    @CachedProperty
    def rhos_Tms(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vms_Tms]
    @CachedProperty
    def rhos_Tms_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vms_Tms, self.MWs)]
    @CachedProperty
    def sigma_STPs(self):
        return [i.T_dependent_property(298.15) for i in self.SurfaceTensions]
    @CachedProperty
    def sigma_Tbs(self):
        return [None if Tb is None else i.T_dependent_property(Tb)
                for i, Tb in zip(self.SurfaceTensions, self.Tbs)]
    @CachedProperty
    def sigma_Tms(self):
        return [None if Tm is None else i.T_dependent_property(Tm)
               for i, Tm in zip(self.SurfaceTensions, self.Tms)]
    @CachedProperty
    def Van_der_Waals_volumes(self):
        N = self.N
        N_range = self.cmps
        UNIFAC_Rs = [None]*N
        UNIFAC_groups = self.UNIFAC_groups
        for i in N_range:
            groups = UNIFAC_groups[i]
            if groups is not None:
                UNIFAC_Rs[i] = UNIFAC_RQ(groups)[0]
        return [Van_der_Waals_volume(UNIFAC_Rs[i]) if UNIFAC_Rs[i] is not None else None for i in N_range]
    @CachedProperty
    def Van_der_Waals_areas(self):
        N = self.N
        N_range = self.cmps
        UNIFAC_Qs = [None]*N
        UNIFAC_groups = self.UNIFAC_groups
        for i in N_range:
            groups = UNIFAC_groups[i]
            if groups is not None:
                UNIFAC_Qs[i] = UNIFAC_RQ(groups)[1]
        return [Van_der_Waals_area(UNIFAC_Qs[i]) if UNIFAC_Qs[i] is not None else None for i in N_range]
    @CachedProperty
    def Parachors(self):
        N = self.N
        Parachors = [None]*N
        sigma_STPs = self.sigma_STPs
        MWs = self.MWs
        rhol_STPs_mass = self.rhol_STPs_mass
        rhog_STPs_mass = self.rhog_STPs_mass
        for i in self.cmps:
            try:
                Parachors[i] = Parachor(sigma=sigma_STPs[i], MW=MWs[i], rhol=rhol_STPs_mass[i], rhog=rhog_STPs_mass[i])
            except:
                pass
        return Parachors
    