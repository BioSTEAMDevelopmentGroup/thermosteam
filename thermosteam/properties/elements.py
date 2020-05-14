# -*- coding: utf-8 -*-
"""
This module includes elemental data taken from [1]_ and [2]_, and 
functions to calculate molecular properties from elemental data.

References
----------
.. [1] N M O'Boyle, M Banck, C A James, C Morley, T Vandermeersch, and
       G R Hutchison. "Open Babel: An open chemical toolbox." J. Cheminf.
       (2011), 3, 33. DOI:10.1186/1758-2946-3-33
.. [2] RDKit: Open-source cheminformatics; http://www.rdkit.org
.. [3] Laštovka, Václav, Nasser Sallamie, and John M. Shaw. "A Similarity
       Variable for Estimating the Heat Capacity of Solid Organic Compounds:
       Part I. Fundamentals." Fluid Phase Equilibria 268, no. 1-2
       (June 25, 2008): 51-60. doi:10.1016/j.fluid.2008.03.019.
.. [4] Hill, Edwin A."“ON A SYSTEM OF INDEXING CHEMICAL LITERATURE;
       ADOPTED BY THE CLASSIFICATION DIVISION OF THE U. S. PATENT OFFICE.1."
       Journal of the American Chemical Society 22, no. 8 (August 1, 1900):
       478-94. doi:10.1021/ja02046a005.

"""

__all__ = (
    'PeriodicTable', 'compute_molecular_weight', 'compute_mass_fractions', 
    'compute_atom_fractions', 'compute_similarity_variable', 'atoms_to_Hill', 
    'parse_simple_formula', 'parse_nested_formula', 'homonuclear_elemental_gases',
    'charge_from_formula', 'serialize_formula', 'atoms_to_array',
)
import numpy as np
import os
import re
import string
from .readers import load_json

fpath = os.path
join = fpath.join
parent_path = join(fpath.dirname(__file__), 'Data')
folder =  join(parent_path, 'Misc')

# %% Element data

# Big problem: Atoms like N2, O2 point to only the singlet
homonuclear_elemental_gases = (1, 7, 8, 9, 17) # 35, 53
homonuclear_elemental_singlets_CASs = ("12385-13-6", "17778-88-0",
                                       "17778-80-2", "14762-94-8",
                                       "22537-15-1")

homonuclear_elemental_gases = frozenset(homonuclear_elemental_gases)
homonuclear_elemental_singlets_CASs = frozenset(homonuclear_elemental_singlets_CASs)

# %% Core

class PeriodicTable:
    """
    Periodic Table object for use in dealing with elements.

    Parameters
    ----------
    elements : Iterable[Element]
        List of Element objects

    Notes
    -----
    Has a length of 118 elements.

    See Also
    --------
    periodic_table
    Element
    
    """
    __slots__ = ('elements', 'numerical_index', 'symbol_index',
                 'name_index', 'CAS_index', 'indexes')
    def __init__(self, elements):
        self.elements = elements = tuple(elements)
        self.numerical_index = numerical_index = {}
        self.symbol_index = symbol_index = {}
        self.name_index = name_index = {}
        self.CAS_index = CAS_index = {}
        self.indexes = (symbol_index, numerical_index,
                            name_index, CAS_index)
        for e in elements:
            numerical_index[str(e.number)] = e
            symbol_index[e.symbol] = e
            name_index[e.name] = e
            name_index[e.name.lower()] = e
            CAS_index[e.CAS] = e
         
    def __contains__(self, key):
        for i in self.indexes:
            if key in i: return True
        return False

    def __len__(self):
        return 118

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, key):
        for i in self.indexes:
            if key in i: return i[key]


class Element:
    """
    Create an Element object that stores data on chemical elements. Supports most common
    properties. If a property is not available, it is set to None.

    Attributes
    ----------
    number : int
        Atomic number
    name : str
        name
    symbol : str
        Elemental symbol
    MW : float
        Molecular weight
    CAS : str
        CAS number
    period : str
        Period in the periodic table
    group : str
        Group in the periodic table
    block : str
        Block in the periodic table
    AReneg : float
        Allred and Rochow electronegativity
    rcov : float
        Covalent radius, [Angstrom]
    rvdw : float
        Van der Waals radius, [Angstrom]
    maxbonds : float
        Maximum valence of a bond with this element
    elneg : float
        Pauling electronegativity
    ionization : float
        Ionization potential, [eV]
    ionization : float
        elaffinity affinity, [eV]
    protons : int
        Number of protons
    electrons : int
        Number of electrons of the element in the ground state
    InChI : str
        Standard InChI string of the element
    InChI_key : str
        25-character hash of the compound's InChI.
    smiles : str
        Standard smiles string of the element
    PubChem : int
        PubChem Compound identifier (CID) of the chemical
    """
    __slots__ = ('number', 'symbol', 'name', 'CAS', 'MW', 'AReneg', 'rcov',
                 'rvdw', 'maxbonds', 'elneg', 'ionization', 'elaffinity',
                 'period', 'group', 'block', 'InChI_key', 'PubChem')

    def __init__(self, name, number, symbol, MW, CAS, AReneg, rcov, rvdw,
                 maxbonds, elneg, ionization, elaffinity, period, group, block,
                 PubChem, InChI_key):
        self.name = name
        self.number = number
        self.symbol = symbol
        self.MW = MW
        self.CAS = CAS
        self.period = period
        self.group = group
        self.block = block
        self.AReneg = AReneg
        self.rcov = rcov
        self.rvdw = rvdw
        self.maxbonds = maxbonds
        self.elneg = elneg
        self.ionization = ionization
        self.elaffinity = elaffinity
        self.InChI_key = InChI_key
        self.PubChem = PubChem

    @property
    def protons(self): return self.number
    @property
    def electrons(self): return self.number
    @property
    def InChI(self): return self.symbol # 'InChI=1S/' +
    @property
    def smiles(self): return '[' + self.symbol + ']'

    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"

#: Single instance of the PeriodicTable class
periodic_table = PeriodicTable(
    [Element(name, **data) for name, data in load_json(folder, 'elements.json').items()])

def compute_molecular_weight(atoms):
    r"""
    Return molecular weight of a molecule given a dictionary of its
    atoms and their counts, in the format {symbol: count}.

    .. math::
        MW = \sum_i n_i MW_i

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Returns
    -------
    MW : float
        Calculated molecular weight [g/mol]

    Notes
    -----
    Elemental data is from rdkit, with CAS numbers added. An exception is
    raised if an incorrect element symbol is given. Elements up to 118 are
    supported, as are deutreium and tritium.

    Examples
    --------
    >>> compute_molecular_weight({'H': 12, 'C': 20, 'O': 5}) # DNA
    332.30628

    """
    MW = 0
    for i in atoms:
        if i in periodic_table:
            MW += periodic_table[i].MW*atoms[i]
        elif i == 'D':
            # Hardcoded MW until an actual isotope db is created
            MW += 2.014102*atoms[i]
        elif i == 'T':
            # Hardcoded MW until an actual isotope db is created
            MW += 3.0160492*atoms[i]
        else:
            raise ValueError(f'molecule includes unknown atom {repr(i)}')
    return MW

def compute_mass_fractions(atoms, MW=None):
    r"""
    Return the mass fractions of each element in a compound,
    given a dictionary of its atoms and their counts, in the format
    {symbol: count}.

    .. math::
        w_i =  \frac{n_i MW_i}{\sum_i n_i MW_i}

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]
    MW : float, optional
        Molecular weight, [g/mol]

    Returns
    -------
    mfracs : dict
        dictionary of mass fractions of individual atoms, indexed by symbol
        with proper capitalization, [-]

    Notes
    -----
    Molecular weight is optional, but speeds up the calculation slightly. It
    is calculated using the function `compute_molecular_weight` if not specified.

    Elemental data is from rdkit, with CAS numbers added. An exception is
    raised if an incorrect element symbol is given. Elements up to 118 are
    supported.

    Examples
    --------
    >>> compute_mass_fractions({'H': 12, 'C': 20, 'O': 5})
    {'H': 0.03639798802478244, 'C': 0.7228692758981262, 'O': 0.24073273607709128}

    """
    if not MW:
        MW = compute_molecular_weight(atoms)
    mfracs = {}
    for i in atoms:
        if i in periodic_table:
            mfracs[i] = periodic_table[i].MW*atoms[i]/MW
        else:
            raise Exception('Molecule includes unknown atoms')
    return mfracs


def compute_atom_fractions(atoms):
    r"""
    Return the atomic fractions of each element in a compound,
    given a dictionary of its atoms and their counts, in the format
    {symbol: count}.

    .. math::
        a_i =  \frac{n_i}{\sum_i n_i}

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Returns
    -------
    afracs : dict
        dictionary of atomic fractions of individual atoms, indexed by symbol
        with proper capitalization, [-]

    Notes
    -----
    No actual data on the elements is used, so incorrect or custom compounds
    would not raise an error.

    Examples
    --------
    >>> compute_atom_fractions({'H': 12, 'C': 20, 'O': 5})
    {'H': 0.32432432432432434, 'C': 0.5405405405405406, 'O': 0.13513513513513514}

    """
    count = sum(atoms.values())
    afracs = {}
    for i in atoms:
        afracs[i] = atoms[i]/count
    return afracs

def compute_similarity_variable(atoms, MW=None):
    r"""
    Return the similarity variable of an compound, as defined in [3]_.
    Currently only applied for certain heat capacity estimation routines.

    .. math::
        \alpha = \frac{N}{MW} = \frac{\sum_i n_i}{\sum_i n_i MW_i}

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]
    MW : float, optional
        Molecular weight, [g/mol]

    Returns
    -------
    compute_similarity_variable : float
        Similarity variable as defined in [1]_, [mol/g]

    Notes
    -----
    Molecular weight is optional, but speeds up the calculation slightly. It
    is calculated using the function `compute_molecular_weight` if not specified.

    Examples
    --------
    >>> compute_similarity_variable({'H': 32, 'C': 15})
    0.2212654140784498

    """
    if not MW:
        MW = compute_molecular_weight(atoms)
    return sum(atoms.values())/MW

def atoms_to_Hill(atoms):
    r"""
    Determine the Hill formula of a compound as in [4]_, given a dictionary of its
    atoms and their counts, in the format {symbol: count}.

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Returns
    -------
    Hill_formula : str
        Hill formula, [-]

    Notes
    -----
    The Hill system is as follows:

    If the chemical has 'C' in it, this is listed first, and then if it has
    'H' in it as well as 'C', then that goes next. All elements are sorted
    alphabetically afterwards, including 'H' if 'C' is not present.
    All elements are followed by their count, unless it is 1.

    Examples
    --------
    >>> atoms_to_Hill({'H': 5, 'C': 2, 'Br': 1})
    'C2H5Br'

    """
    def str_ele_count(ele):
        if atoms[ele] == 1:
            count = ''
        else:
            count = str(atoms[ele])
        return count
    atoms = atoms.copy()
    s = ''
    if 'C' in atoms.keys():
        s += 'C' + str_ele_count('C')
        del atoms['C']
        if 'H' in atoms.keys():
            s += 'H' + str_ele_count('H')
            del atoms['H']
        for ele in sorted(atoms.keys()):
            s += ele + str_ele_count(ele)
    else:
        for ele in sorted(atoms.keys()):
            s += ele + str_ele_count(ele)
    return s

_formula_parser = re.compile(r'([A-Z][a-z]{0,2})([\d\.\d]+)?')

def parse_simple_formula(formula):
    r"""
    Basic formula parser, primarily for obtaining element counts from 
    formulas as formated in PubChem. Handles formulas with integer counts, 
    but no brackets, no hydrates, no charges, no isotopes, and no group
    multipliers.
    
    Strips charges from the end of a formula first. Accepts repeated chemical
    units. Performs no sanity checking that elements are actually elements.
    As it uses regular expressions for matching, errors are mostly just ignored.
    
    Parameters
    ----------
    formula : str
        Formula string, very simply formats only.

    Returns
    -------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Notes
    -----
    Inspiration taken from the thermopyl project, at
    https://github.com/choderalab/thermopyl.

    Examples
    --------
    >>> parse_simple_formula('CO2')
    {'C': 1, 'O': 2}
    """
    formula = formula.split('+')[0].split('-')[0]
    counts = {}
    for element, count in _formula_parser.findall(formula):
        if count.isdigit():
            count = int(count)
        elif count:
            count = float(count)
        else:
            count = 1
        if element in counts:
            counts[element] += count
        else:
            counts[element] = count
    return counts


formula_token_matcher_rational = re.compile('[A-Z][a-z]?|(?:\d*[.])?\d+|\d+|[()]')
letter_set = set(string.ascii_letters)
bracketed_charge_re = re.compile('\([+-]?\d+\)$|\(\d+[+-]?\)$|\([+-]+\)$')

def parse_nested_formula(formula, check=True):
    r"""
    Improved formula parser which handles braces and their multipliers, 
    as well as rational element counts.

    Strips charges from the end of a formula first. Accepts repeated chemical
    units. Performs no sanity checking that elements are actually elements.
    As it uses regular expressions for matching, errors are mostly just ignored.
    
    Parameters
    ----------
    formula : str
        Formula string, very simply formats only.
    check : bool
        If `check` is True, a simple check will be performed to determine if
        a formula is not a formula and an exception will be raised if it is
        not, [-]

    Returns
    -------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Notes
    -----
    Inspired by the approach taken by CrazyMerlyn on a reddit DailyProgrammer
    challenge, at https://www.reddit.com/r/dailyprogrammer/comments/6eerfk/20170531_challenge_317_intermediate_counting/

    Examples
    --------
    >>> pprint(parse_nested_formula('Pd(NH3)4.0001+2'))
    {'H': 12.0003, 'N': 4.0001, 'Pd': 1}
    """
    formula = formula.replace('[', '').replace(']', '')
    charge_splits = bracketed_charge_re.split(formula)
    if len(charge_splits) > 1:
        formula = charge_splits[0]
    else:
        formula = formula.split('+')[0].split('-')[0]
    
    stack = [[]]
    last = stack[0]
    tokens = formula_token_matcher_rational.findall(formula)
    # The set of letters in the tokens should match the set of letters
    if check:
        token_letters = set([j for i in tokens for j in i if j in letter_set])
        formula_letters = set(i for i in formula if i in letter_set)
        if formula_letters != token_letters:
            raise Exception('Input may not be a formula; extra letters were detected')
    
    for token in tokens:
        if token == "(":
            stack.append([])
            last = stack[-1]
        elif token == ")":
            temp_dict = {}
            for d in last:
                for ele, count in d.items():
                    if ele in temp_dict:
                        temp_dict[ele] = temp_dict[ele] + count
                    else:
                        temp_dict[ele] = count
            stack.pop()
            last = stack[-1]
            last.append(temp_dict)
        elif token.isalpha():
            last.append({token: 1})
        else:
            v = float(token)
            v_int = int(v)
            if v_int == v:
                v = v_int
            last[-1] = {ele: count*v for ele, count in last[-1].items()}
    ans = {}
    for d in last:
        for ele, count in d.items():
            if ele in ans:
                ans[ele] = ans[ele] + count
            else:
                ans[ele] = count
    return ans

def charge_from_formula(formula):
    r"""
    Basic formula parser to determine the charge from a formula - given
    that the charge is already specified as one element of the formula.

    Performs no sanity checking that elements are actually elements.
    
    Parameters
    ----------
    formula : str
        Formula string, very simply formats only, ending in one of '+x',
        '-x', n*'+', or n*'-' or any of them surrounded by brackets but always
        at the end of a formula.

    Returns
    -------
    charge : int
        Charge of the molecule, [faraday]

    Notes
    -----

    Examples
    --------
    >>> charge_from_formula('Br3-')
    -1
    >>> charge_from_formula('Br3(-)')
    -1
    """
    negative = '-' in formula
    positive = '+' in formula
    if positive and negative:
        raise ValueError('Both negative and positive signs were found in the formula; only one sign is allowed')
    elif not (positive or negative):
        return 0
    multiplier, sign = (-1, '-') if negative else (1, '+')
    
    hit = False
    if '(' in formula:
        hit = bracketed_charge_re.findall(formula)
        if hit:
            formula = hit[-1].replace('(', '').replace(')', '')

    count = formula.count(sign)
    if count == 1:
        splits = formula.split(sign)
        if splits[1] == '' or splits[1] == ')':
            return multiplier
        else:
            return multiplier*int(splits[1])
    else:
        return multiplier*count

def serialize_formula(formula):
    r"""
    Basic formula serializer to construct a consistently-formatted formula.
    This is necessary for handling user-supplied formulas, which are not always
    well formatted.

    Performs no sanity checking that elements are actually elements.
    
    Parameters
    ----------
    formula : str
        Formula string as parseable by the method parse_nested_formula, [-]

    Returns
    -------
    formula : str
        A consistently formatted formula to describe a molecular formula, [-]

    Examples
    --------
    >>> serialize_formula('Pd(NH3)4+3')
    'H12N4Pd+3'
    """
    charge = charge_from_formula(formula)
    element_dict = parse_nested_formula(formula)
    base = atoms_to_Hill(element_dict)
    if charge  == 0:
        pass
    elif charge > 0:
        if charge == 1:
            base += '+'
        else:
            base += '+' + str(charge)
    elif charge < 0:
        if charge == -1:
            base += '-'
        else:
            base +=  str(charge)
    return base

def atoms_to_array(atoms: dict) -> np.ndarray:
    symbol_index = periodic_table.symbol_index
    array = np.zeros(118)
    for symbol, value in atoms.items():
        index = symbol_index[symbol].number - 1
        array[index] = value
    return array

def array_to_atoms(array: np.ndarray) -> dict:
    index, = np.where(array != 0.)
    values = array[index]
    elements = periodic_table.elements
    symbols = [elements[i].symbol for i in index]
    return dict(zip(symbols, values))
        