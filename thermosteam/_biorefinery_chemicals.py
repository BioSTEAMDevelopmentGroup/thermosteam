# -*- coding: utf-8 -*-
"""
"""
import thermosteam as tmo
from chemicals.identifiers import pubchem_db

biorefinery_chemicals = {}

def search_biorefinery_chemicals(ID, **kwargs):
    if ID in biorefinery_chemicals: 
        f = biorefinery_chemicals[ID]
    else:
        key = ID.replace(' ', '').replace('-', '').replace('_', '').lower()
        if key in biorefinery_chemicals: 
            f = biorefinery_chemicals[key]
        else:
            return tmo.Chemical(ID, db='ChEDL', **kwargs)
    return f(ID, **kwargs)

# Lignocellulosic heat capacities:
# Assume heat capacity of lignin, cellulose, and hemicellulose
# and all components at 350 K are about the same.
# https://link.springer.com/article/10.1007/s10853-013-7815-6
# https://www.sciencedirect.com/science/article/pii/0032386182901252
Cp_cellulosic = 1.364

# Assume density is similar for most solutes and solids, in agreement with Superpro models for sugarcane.
rho_solute = 1540 # kg/m3
rho_solid = 700 # kg/m3 - pelleted biomass

# Heats of formation for cellulosic components are from Humbird 2011 report: https://www.nrel.gov/docs/fy11osti/47764.pdf
# They are originally found in calories, so we need to convert them to joule.
cal2joule = 4.184

def _register(f, aliases):
    keys = [*aliases, f.__name__]
    keys.extend([i.lower() for i in keys])
    try:
        metadata = pubchem_db.search(f.__name__)
    except:
        pass    
    else:
        iupac_name = metadata.iupac_name, 
        if not iupac_name: iupac_name = ()
        elif isinstance(iupac_name, str): iupac_name = (iupac_name,)
        keys = set([*keys, metadata.common_name, metadata.formula])
    keys = set([i for i in keys if i not in biorefinery_chemicals])
    for i in keys: biorefinery_chemicals[i] = f
    return f

def register(*aliases):
    if len(aliases) == 1 and not isinstance(aliases[0], str):
        return _register(aliases[0], ())
    else:
        return lambda f: _register(f, aliases)

# %% Non-interacting solids

@register
def Ash(ID, **kwargs):
    return tmo.Chemical(ID, MW=1., default=True, phase='s', db=None,
                        Cp=0.37656, **kwargs)

@register
def Flocculant(ID, **kwargs):
    return tmo.Chemical(ID, MW=1., default=True, phase='s', db=None,
                        Cp=Cp_cellulosic, **kwargs)

@register
def Solids(ID, **kwargs):
    return tmo.Chemical(ID, MW=1., default=True, phase='s', db=None,
                        Cp=1.100, **kwargs)

# %% Microbes and cellular components

@register
def Yeast(ID, **kwargs):
    return tmo.Chemical(ID, formula='CH1.61O0.56', rho=1112.6,
                        default=True, phase='s', db=None, 
                        Hf=(-7055.556554690305, 'J/g'), **kwargs)

@register('NYeast')
def NitrogenYeast(ID, **kwargs):
    return tmo.Chemical(ID, formula='CH1.61O0.56N0.16', phase='s',
                        rho=1112.6, default=True, db=None, 
                        Hf=(-7055.556554690305, 'J/g'), **kwargs)

@register('WWTsludge')
def Sluge(ID, **kwargs):
    return tmo.Chemical(ID, formula='CH1.64O0.39N0.23S0.0035', 
                        rho=1112.6, default=True, phase='s', 
                        Hf=-23200.01*cal2joule, db=None, **kwargs)

@register('Enzyme', 'Cellulase', 'Amylase')
def Protein(ID, **kwargs):
    return tmo.Chemical(ID, formula='CH1.57O0.31N0.29S0.007', 
                        rho=1112.6, default=True, phase='s', 
                        Hf=-17618*cal2joule, db=None, **kwargs)

# %% Cellulosic biomass components

@register('Mannose', 'Galactose')
def Glucose(ID, **kwargs):
    return tmo.Chemical(ID, search_ID='Glucose', equilibrium_phases='ls', 
                        db='ChEDL', N_solutes=1, **kwargs)

@register
def Sucrose(ID, **kwargs):
    return tmo.Chemical(ID, search_ID='Sucrose', equilibrium_phases='ls', 
                        db='ChEDL', N_solutes=2, **kwargs)

@register('Arabinose')
def Xylose(ID, **kwargs):
    return tmo.Chemical(ID, search_ID='Xylose', equilibrium_phases='ls',
                        db='ChEDL', **kwargs)

@register('Cellulose')
def Glucan(ID, **kwargs):
    return tmo.Chemical(ID, formula='C6H10O5', phase='s', db=None,
                        Hf=-233200*cal2joule, rho=rho_solid, Cp=Cp_cellulosic, 
                        default=True, **kwargs)

@register
def Hemicellulose(ID, **kwargs):
    return tmo.Chemical(ID, formula="C5H8O4", # Model formula as xylose monomer minus water
                        Hf=-761906.4, default=True, phase='s', db=None, 
                        Cp=Cp_cellulosic, **kwargs)

@register('Arabinan')
def Xylan(ID, **kwargs):
    return tmo.Chemical(ID, formula='C5H8O4', phase='s', db=None,
                        Hf=-182100*cal2joule, rho=rho_solid, Cp=Cp_cellulosic, 
                        default=True, **kwargs)

@register
def Lignin(ID, **kwargs):
    return tmo.Chemical(ID, formula='C8H8O3', phase='s', db=None,
                        Hf=-108248*cal2joule, rho=rho_solid, Cp=Cp_cellulosic, 
                        default=True, **kwargs)

@register
def SolubleLignin(ID, **kwargs):
    return tmo.Chemical(ID, formula='C8H8O3', phase='l', db=None,
                        Hf=-108248*cal2joule, rho=rho_solute, Cp=Cp_cellulosic,
                        default=True, **kwargs)

@register('GalactoseOligomer', 'MannoseOligomer')
def GlucoseOligomer(ID, **kwargs):
    return tmo.Chemical(ID, formula='C6H10O5', phase='l', db=None, 
                        Hf=-233200*cal2joule, rho=rho_solute, Cp=Cp_cellulosic, 
                        default=True, **kwargs)

@register('ArabinoseOligomer')
def XyloseOligomer(ID, **kwargs):
    return tmo.Chemical(ID, formula='C5H8O4', phase='l', db=None, 
                        Hf=-182100*cal2joule, rho=rho_solute, Cp=Cp_cellulosic, 
                        default=True, **kwargs)

@register
def Xylitol(ID, **kwargs):
    return tmo.Chemical(ID, formula='C5H12O5', phase='l', db=None,
                        Hf=-243145*cal2joule, rho=rho_solid, Cp=Cp_cellulosic, 
                        default=True, **kwargs)

@register
def Cellobiose(ID, **kwargs):
    return tmo.Chemical(ID, formula='C12H22O11', phase='l', db=None,
                        Hf=-480900*cal2joule, rho=rho_solid, Cp=Cp_cellulosic, 
                        default=True, **kwargs)
