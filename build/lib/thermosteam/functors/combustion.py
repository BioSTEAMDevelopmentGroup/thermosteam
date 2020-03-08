# -*- coding: utf-8 -*-
from .elements import compute_mass_fractions, compute_molecular_weight

__all__ = ('CombustionData', 'estimate_HHV_modified_Dulong')

# %% Combustion functions

combustible_elements = ('C', 'H', 'N', 'O', 'S', 'Br', 'I', 'Cl', 'F', 'P')
combustion_products = {'7782-44-7', '124-38-9', '7726-95-6', '7553-56-2',
                       '7647-01-0', '7664-39-3', '7446-09-5', '7727-37-9',
                       '16752-60-6', '7732-18-5', '630-08-0',
                       'Ash', 'H2O', 'CO2', 'CO', 'SO2', 'Br2',
                       'I2', 'HCl', 'HF', 'P4O10', 'O2', 'N2'}

Hf_combustion_products = {
    'H2O': -285825,
    'CO2': -393474,
    'SO2': -296800,
    'Br2': 30880,
    'I2': 62417,
    'HCl': -92173,
    'HF': -272711,
    'P4O10': -3009940,
    'O2': 0,
    'N2': 0,
    "Ash": 0,
}

def get_combustion_stoichiometry(atoms, MW=None):
    """
    Return a dictionary of the combustion stoichiometry of a chemical.
    
    Parameters
    ----------
    atoms : dict
        Dictionary of atoms and their counts.
    MW : float, optional
        Molecular weight of chemical.

    Notes
    -----
    The stoichiometry is given by:
        
    .. math::
        C_c H_h O_o N_n S_s Br_b I_i Cl_x F_f P_p + (c + s + \frac{h}{4} + \frac{5P}{4} - \frac{x + f}{4} - \frac{o}{2}) -> cCO_2 + \frac{b}{2}Br_2 + \frac{i}{2}I + xHCl + fHF + sSO_2 + \frac{n}{2}N_2 + \frac{p}{4}P_4O_{10} +\frac{h + x + f}{2}H_2O

    """
    combustion_atoms = {i:atoms.get(i, 0) for i in combustible_elements}
    C, H, N, O, S, Br, I, Cl, F, P = combustion_atoms.values()
    MW = MW or compute_molecular_weight(atoms)
    Ash = MW - compute_molecular_weight(combustion_atoms)
    stoichiometry = {
        'O2': (Cl + F)/4. + O/2. - (C + S + H/4. + 5*P/4.),
        'CO2': C,
        'Br2': Br/2.,
        'I2': I/2.,
        'HCl': Cl,
        'HF': F,
        'SO2': S,
        'N2': N/2.,
        'P4O10': P/4.,
        'H2O': (H - Cl - F)/2.,
        'Ash': Ash if Ash > 0.01 else 0
    }
    return {i: j for i,j in stoichiometry.items() if j}
    

def estimate_HHV_from_stoichiometry(stoichiometry, Hf):
    """Estimate the higher heating value [HHV; in J/mol] given a dictionary
    of the combustion stoichiometry and the heat of formation of the chemical."""
    return sum([stoichiometry[chem] * Hf_combustion_products[chem] for chem in stoichiometry]) - Hf

# TODO: Continue adding more methods for estimating HHV
def estimate_HHV_modified_Dulong(atoms, MW=None, check_oxygen_content=False):
    r"""
    Return higher heating value [HHV; in J/mol] based on the modified 
    Dulong's equation.
    
    Parameters
    ----------
    
    Notes
    -----
    The heat of combustion in J/mol is given by Dulong's equation [1]_:
    
    .. math:: 
        Hc (J/mol) = MW \cdot (338C + 1428(H - O/8)+ 95S)
    
    This equation is only good for <10 wt. % Oxygen content. Variables C, H, O,
    and S are atom weight fractions.
    
    References
    ----------
    .. [1] Brown et al., Energy Fuels 2010, 24 (6), 3639–3646.
    
    """
    mass_fractions = compute_mass_fractions(atoms, MW)
    C = mass_fractions.get('C', 0)
    H = mass_fractions.get('H', 0)
    O = mass_fractions.get('O', 0)
    S = mass_fractions.get('S', 0)
    if check_oxygen_content:
        assert O <= 0.105, f"Dulong's formula is only valid at 10 wt. % Oxygen or less ({O:.0%} given)"
    return - MW * (338*C  + 1428*(H - O/8)+ 95*S)

def estimate_LHV(HHV, N_H2O):
    """Estimate the lower heating value [LHV; in J/mol] of a chemical given
    the higher heating value [HHV; in J/mol] and the number of water
    molecules formed per molecule burned."""
    return HHV + 44011.496 * N_H2O


# %% Combustion reaction

class CombustionData:
    """
    Create a CombustionData object that contains the stoichiometry 
    coefficients of the reactants and products and the lower and higher 
    heating values of a chemical [LHV, HHV; in J/mol].
    
    Parameters
    ----------
    stoichiometry : dict[str: float] 
        Stoichiometry coefficients of the reactants and products.
    LHV : float 
        Lower heating value [J/mol].
    HHV : float
        Higher heating value [J/mol].
    Hf : float
        Heat of formation [J/mol].
    
    """
    __slots__ = ('stoichiometry', 'HHV', 'LHV', 'Hf')
    
    def __init__(self, stoichiometry, LHV, HHV, Hf):
        #: dict[str: float] Stoichiometry coefficients of the reactants and products
        self.stoichiometry = stoichiometry
        #: [float] Lower heating value [J/mol]
        self.LHV = LHV
        #: [float] Higher heating value [J/mol]
        self.HHV = HHV
        #: [float] Heat of formation [J/mol]
        self.Hf = Hf
    
    @classmethod
    def from_chemical_data(cls, atoms, CAS=None, MW=None, Hf=None, method='Stoichiometry'):
        r'''
        Return a CombustionData object that contains the stoichiometry 
        coefficients of the reactants and products and the lower and higher 
        heating values [LHV, HHV; in J/mol].
    
        Parameters
        ----------
        atoms : dict
            Dictionary of atoms and their counts.
        CAS : str, optional
            CAS of chemical.
        MW : float, optional
            Molecular weight of chemical [g/mol].
        Hf : float, optional
            Heat of formation of given chemical [J/mol].
            Required if method is "Stoichiometry".
        method : "Stoichiometry" or "Dulong"
            Method to estimate LHV and HHV.    
        
        Notes
        -----
        Default heats of formation for chemicals are at 298 K, 1 atm. The
        combustion reaction is based on the following equation:
        
        .. math::
            C_c H_h O_o N_n S_s Br_b I_i Cl_x F_f P_p + (c + s + \frac{h}{4} + \frac{5P}{4} - \frac{x + f}{4} - \frac{o}{2}) -> cCO_2 + \frac{b}{2}Br_2 + \frac{i}{2}I + xHCl + fHF + sSO_2 + \frac{n}{2}N_2 + \frac{p}{4}P_4O_{10} +\frac{h + x + f}{2}H_2O
    
        If the method is "Stoichiometry", the HHV is found using 
        through an energy balance on the reaction (i.e. heat of reaction).
        If the method is "Dulong", Dulong's equation is used [1]_:
    
        .. math:: 
            Hc (J/mol) = MW \cdot (338C + 1428(H - O/8)+ 95S)
            
        The LHV is calculated as follows:
            
        .. math::
            LHV = HHV + H_{vap} \cdot H_2O
            H_{vap} = 44011.496 \frac{J}{mol H_2O}
            H_2O = \frac{mol H_2O}{mol}
            
            
        Examples
        --------
        Liquid methanol burning:
    
        >>> from thermosteam.functors.combustion import CombustionData
        >>> CombustionData.from_chemical_data({'H': 4, 'C': 1, 'O': 1}, Hf=-239100)
        CombustionData(stoichiometry={'O2': -1.5, 'CO2': 1, 'H2O': 2.0}, LHV=-6.38e+05, HHV=-7.26e+05, Hf=-2.39e+05)
        
        '''
        if CAS in combustion_products:
            return cls("", 0, 0, Hf)
        stoichiometry = get_combustion_stoichiometry(atoms, MW)
        if method == 'Stoichiometry':
            HHV = estimate_HHV_from_stoichiometry(stoichiometry, Hf)
            N_H2O = stoichiometry.get('H2O', 0)
        elif method == 'Dulong':
            HHV = estimate_HHV_modified_Dulong(atoms, MW)
            Hf = HHV - estimate_HHV_from_stoichiometry(stoichiometry, 0)
            N_H2O = atoms.get('H', 0) / 2
        else:
            raise ValueError(f"invalid method {repr(method)}; method must be either 'Stoichiometry' or 'Dulong'")
        LHV = estimate_LHV(HHV, N_H2O)
        return cls(stoichiometry, LHV, HHV, Hf)
    
    def __repr__(self):
        return f"{type(self).__name__}(stoichiometry={repr(self.stoichiometry)}, LHV={self.LHV:.3g}, HHV={self.HHV:.3g}, Hf={self.Hf:.3g})"






