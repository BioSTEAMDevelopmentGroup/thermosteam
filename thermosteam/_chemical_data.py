# -*- coding: utf-8 -*-
"""
"""
from chemicals.lennard_jones import Stockmayer, molecular_diameter
from thermo.unifac import UNIFAC_RQ, Van_der_Waals_volume, Van_der_Waals_area
from chemicals.utils import Parachor
from collections import Iterable
import thermosteam as tmo

class ChemicalData:
    """
    Create a ChemicalData object that works as thermo ChemicalPropertiesPackage
    and ChemicalConstantsPackage objects.
    """
    __slots__ = ('N', 'chemicals', 'index', 'cmps')
    
    def __init__(self, chemicals):
        if isinstance(chemicals, tmo.CompiledChemicals):
            self.chemicals = chemicals.tuple
            self.index = chemicals._index
            self.N = chemicals.size
        else:
            self.chemicals = tuple(chemicals)
            self.N = len(chemicals)
        self.cmps = range(self.N)
    
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
    
    @property
    def EnthalpyVaporizations(self):
        return [i.Hvap for i in self.chemicals]
    @property
    def VaporPressures(self):
        return [i.Psat for i in self.chemicals]
    @property
    def SublimationPressures(self):
        return [i.Psub for i in self.chemicals]
    @property
    def EnthalpySublimations(self):
        return [i.Hsub for i in self.chemicals]
    @property
    def SurfaceTensions(self):
        return [i.sigma for i in self.chemicals]
    @property
    def PermittivityLiquids(self):
        return [i.epsilon for i in self.chemicals]
    @property
    def CASs(self):
        return [i.CAS for i in self.chemicals]
    @property
    def MWs(self):
        return [i.MW for i in self.chemicals]
    @property
    def Tms(self):
        return [i.Tm for i in self.chemicals]
    @property
    def Tbs(self):
        return [i.Tb for i in self.chemicals]
    @property
    def Tcs(self):
        return [i.Tc for i in self.chemicals]
    @property
    def Pcs(self):
        return [i.Pc for i in self.chemicals]
    @property
    def Vcs(self):
        return [i.Vc for i in self.chemicals]
    @property
    def omegas(self):
        return [i.omega for i in self.chemicals]
    @property
    def Zcs(self):
        return [i.Zc for i in self.chemicals]
    @property
    def Hfus_Tms(self):
        return [i.Hfus for i in self.chemicals]   
    @property
    def Tts(self):
        return [i.Tt for i in self.chemicals]
    @property
    def Pts(self):
        return [i.Pt for i in self.chemicals]
    @property
    def dipoles(self):
        return [i.dipole for i in self.chemicals]
    @property
    def charges(self):
        return [i.charge for i in self.chemicals]
    @property
    def UNIFAC_groups(self):
        return [i.UNIFAC for i in self.chemicals]
    @property
    def UNIFAC_Dortmund_groups(self):
        return [i.Dortmund for i in self.chemicals]
    @property
    def PSRK_groups(self):
        return [i.PSRK for i in self.chemicals]
    @property
    def similarity_variables(self):
        return [i.similarity_variable for i in self.chemicals]
    @property
    def StielPolars(self):
        return [i.Stiel_Polar for i in self.chemicals]
    @property
    def VolumeGases(self):
        return [i.V if i._locked_state else i.V.g for i in self.chemicals]
    @property
    def VolumeLiquids(self):
        return [i.V if i._locked_state else i.V.l for i in self.chemicals]
    @property
    def VolumeSolids(self):
        return [i.V if i._locked_state else i.V.s for i in self.chemicals]
    @property
    def HeatCapacityGases(self):
        return [i.Cn if i._locked_state else i.Cn.g for i in self.chemicals]
    @property
    def HeatCapacityLiquids(self):
        return [i.Cn if i._locked_state else i.Cn.l for i in self.chemicals]
    @property
    def HeatCapacitySolids(self):
        return [i.Cn if i._locked_state else i.Cn.s for i in self.chemicals]
    @property
    def ViscosityGases(self):
        return [i.mu if i._locked_state else i.mu.g for i in self.chemicals]
    @property
    def ViscosityLiquids(self):
        return [i.mu if i._locked_state else i.mu.l for i in self.chemicals]
    @property
    def ThermalConductivityGases(self):
        return [i.k if i._locked_state else i.k.g for i in self.chemicals]
    @property
    def ThermalConductivityLiquids(self):
        return [i.k if i._locked_state else i.k.l for i in self.chemicals]
    @property
    def rhocs(self):
        return [None if Vc is None else 1.0/Vc for Vc in self.chemicals.Vcs]
    @property
    def rhocs_mass(self):
        return [None if None in (rhoc, MW) else rhoc * 1e-3 * MW
                for rhoc, MW in zip(self.rhocs, self.MWs)]
    @property
    def Hfus_Tms_mass(self):
        return [None if None in (Hfus, MW) else Hfus * 1000.0 / MW
                for Hfus, MW in zip(self.Hfus_Tms, self.MWs)]
    @property
    def Hvap_Tbs(self):
        return [Hvap(Tb) for Hvap, Tb in zip(self.EnthalpyVaporizations, self.Tbs)]
    @property
    def Hvap_Tbs_mass(self):
        return [None if None in (Hvap, MW) else Hvap*1000.0/MW
                for Hvap, MW in zip(self.Hvap_Tbs, self.Tbs)]
    @property
    def Vmg_STPs(self):
        return [i.TP_dependent_property(298.15, 101325.0)
                for i in self.chemicals.VolumeGases]
    @property
    def rhog_STPs(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vmg_STPs]
    @property
    def rhog_STPs_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vmg_STPs, self.MWs)]
    @property
    def Vml_Tbs(self):
        return [None if Tb is None else i.T_dependent_property(Tb) for i, Tb in zip(self.VolumeLiquids, self.Tbs)]
    @property
    def Vml_Tms(self):
        return [None if Tm is None else i.T_dependent_property(Tm)
                for i, Tm in zip(self.VolumeLiquids, self.Tms)]
    @property
    def Vml_STPs(self):
        return [i.T_dependent_property(298.15) for i in self.chemicals.VolumeLiquids]
    @property
    def Vml_60Fs(self):
        return [i.T_dependent_property(288.7055555555555) for i in self.chemicals.VolumeLiquids]
    @property
    def rhol_STPs(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vml_STPs]
    @property
    def rhol_60Fs(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vml_60Fs]
    @property
    def rhol_STPs_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vml_STPs, self.MWs)]
    @property
    def rhol_60Fs_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vml_60Fs, self.MWs)]
    @property
    def Hsub_Tts(self):
        return [None if Tt is None else i(Tt)
                for i, Tt in zip(self.EnthalpySublimations, self.Tts)]
    @property
    def Hsub_Tts_mass(self):
        return [Hsub*1000.0/MW if Hsub is not None else None for Hsub, MW in zip(self.Hsub_Tts, self.MWs)]
    @property
    def Stockmayers(self):
        return [Stockmayer(Tm=self.Tms[i], Tb=self.Tbs[i], Tc=self.Tcs[i], 
                           Zc=self.Zcs[i], omega=self.omegas[i], CASRN=self.CASs[i]) 
                for i in self.cmps]
    @property
    def molecular_diameters(self):
        return [molecular_diameter(Tc=self.Tcs[i], Pc=self.Pcs[i], Vc=self.Vcs[i],
                                   Zc=self.Zcs[i], omega=self.omegas[i],
                                   Vm=self.Vml_Tms[i], Vb=self.Vml_Tbs[i], 
                                   CASRN=self.CASs[i])
                for i in self.cmps]
    @property
    def Psat_298s(self):
        return [i(298.15) for i in self.chemicals.VaporPressures]
    @property
    def Hvap_298s(self):
        return [o.T_dependent_property(298.15) for o in self.chemicals.EnthalpyVaporizations]
    @property
    def Hvap_298s_mass(self):
        return [Hvap*1000.0/MW if Hvap is not None else None for Hvap, MW in zip(self.Hvap_298s, self.MWs)]
    @property
    def Vms_Tms(self):
        return [None if Tm is None else i.T_dependent_property(Tm) for i, Tm in zip(self.VolumeSolids, self.Tms)]
    @property
    def rhos_Tms(self):
        return [1.0/V if V is not None else None for V in self.chemicals.Vms_Tms]
    @property
    def rhos_Tms_mass(self):
        return [1e-3*MW/V if V is not None else None for V, MW in zip(self.Vms_Tms, self.MWs)]
    @property
    def sigma_STPs(self):
        return [i.T_dependent_property(298.15) for i in self.SurfaceTensions]
    @property
    def sigma_Tbs(self):
        return [None if Tb is None else i.T_dependent_property(Tb)
                for i, Tb in zip(self.SurfaceTensions, self.Tbs)]
    @property
    def sigma_Tms(self):
        return [None if Tm is None else i.T_dependent_property(Tm)
               for i, Tm in zip(self.SurfaceTensions, self.Tms)]
    @property
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
    @property
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
    @property
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
    