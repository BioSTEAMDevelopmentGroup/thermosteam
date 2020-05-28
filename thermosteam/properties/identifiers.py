# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# A significant portion of this module originates from:
# Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
# 
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/thermo/blob/master/LICENSE.txt for details.
"""
All data and methods related to chemical identifiers.
"""

__all__ = ('checkCAS', 'chemical_metadata_from_any', 'PubChem', 'MW',
           'formula', 'smiles', 'InChI', 'InChI_Key', 'IUPAC_name',
           'name', 'synonyms', 'pubchem_db')
import re
import os
from .elements import periodic_table, homonuclear_elemental_gases, charge_from_formula, serialize_formula

folder = os.path.join('Data', 'Identifiers')
folder = os.path.join(os.path.dirname(__file__), folder)
searchable_format = re.compile(r"\B([A-Z])")

def spaceout_words(ID):
    return searchable_format.sub(r" \1", ID)

def to_searchable_format(ID):    
    return spaceout_words(ID).replace('_', ' ')

def CAS2int(i):
    r"""Converts CAS number of a compounds from a string to an int. This is
    helpful when storing large amounts of CAS numbers, as their strings take up
    more memory than their numerical representational. All CAS numbers fit into
    64 bit ints.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    CASRN : int
        CASRN [-]

    Notes
    -----
    Accomplishes conversion by removing dashes only, and then converting to an
    int. An incorrect CAS number will change without exception.

    Examples
    --------
    >>> CAS2int('7704-34-9')
    7704349
    """
    return int(i.replace('-', ''))

def int2CAS(i):
    r"""Converts CAS number of a compounds from an int to an string. This is
    helpful when dealing with int CAS numbers.

    Parameters
    ----------
    CASRN : int
        CASRN [-]

    Returns
    -------
    CASRN : string
        CASRN [-]

    Notes
    -----
    Handles CAS numbers with an unspecified number of digits. Does not work on
    floats.

    Examples
    --------
    >>> int2CAS(7704349)
    '7704-34-9'
    """
    i = str(i)
    return i[:-3]+'-'+i[-3:-1]+'-'+i[-1]

def to_nums(values):
    r"""Legacy function to turn a list of strings into either floats
    (if numeric), stripped strings (if not) or None if the string is empty.
    Accepts any numeric formatting the float function does.

    Parameters
    ----------
    values : list
        list of strings

    Returns
    -------
    values : list
        list of floats, strings, and None values [-]

    Examples
    --------
    >>> to_num(['1', '1.1', '1E5', '0xB4', ''])
    [1.0, 1.1, 100000.0, '0xB4', None]
    """
    return [to_num(i) for i in values]

def to_num(value):
    try:
        return float(value)
    except:
        if value == '':
            return None
        else:
            return value.strip()

def checkCAS(CASRN):
    """Checks if a CAS number is valid. Returns False if the parser cannot 
    parse the given string..

    Parameters
    ----------
    CASRN : string
        A three-piece, dash-separated set of numbers

    Returns
    -------
    result : bool
        Boolean value if CASRN was valid. If parsing fails, return False also.

    Notes
    -----
    Check method is according to Chemical Abstract Society. However, no lookup
    to their service is performed; therefore, this function cannot detect
    false positives.

    Function also does not support additional separators, apart from '-'.
    
    CAS numbers up to the series 1 XXX XXX-XX-X are now being issued.
    
    A long can hold CAS numbers up to 2 147 483-64-7

    Examples
    --------
    >>> checkCAS('7732-18-5')
    True
    >>> checkCAS('77332-18-5')
    False
    """
    try:
        check = CASRN[-1]
        CASRN = CASRN[::-1][1:]
        productsum = 0
        i = 1
        for num in CASRN:
            if num != '-':
                productsum += i*int(num)
                i += 1
        return (productsum % 10 == int(check))
    except:
        return False


class ChemicalMetadata:
    __slots__ = ['pubchemid', 'formula', 'MW', 'smiles', 'InChI', 'InChI_key',
                 'iupac_name', 'common_name', 'all_names', 'CAS', '_charge']
        
    @property
    def charge(self):
        """Charge of the species as an integer. Computed as a property as most
        species do not have a charge and so storing it would be a waste of 
        memory.
        """
        try:
            return self._charge
        except AttributeError:
            self._charge = charge_from_formula(self.formula)
            return self._charge
        
    @property
    def CASs(self):
        return int2CAS(self.CAS)
    
    def __init__(self, pubchemid, CAS, formula, MW, smiles, InChI, InChI_key,
                 iupac_name, common_name):
        self.pubchemid = pubchemid
        self.CAS = CAS
        self.formula = formula
        self.MW = MW
        self.smiles = smiles
        self.InChI = InChI
        self.InChI_key = InChI_key
        self.iupac_name = iupac_name
        self.common_name = common_name
        
    def __repr__(self):
        return f"<{type(self).__name__}: {self.CASs}>"
        

class ChemicalMetadataDB:
    __slots__ = ('pubchem_index',
                 'smiles_index',
                 'InChI_index',
                 'InChI_key_index',
                 'name_index',
                 'CAS_index',
                 'formula_index',
                 'unloaded_files',
    )
    def __init__(self, 
                 files=[os.path.join(folder, 'Anion db.tsv'),
                        os.path.join(folder, 'Cation db.tsv'),
                        os.path.join(folder, 'Inorganic db.tsv'),
                        os.path.join(folder, 'chemical identifiers example user db.tsv'),
                        os.path.join(folder, 'chemical identifiers.tsv')]):                
        self.pubchem_index = {}
        self.smiles_index = {}
        self.InChI_index = {}
        self.InChI_key_index = {}
        self.name_index = {}
        self.CAS_index = {}
        self.formula_index = {}
        self.unloaded_files = files
        self.load_elements()
        
    def load_elements(self):
        InChI_key_index = self.InChI_key_index
        CAS_index = self.CAS_index
        pubchem_index = self.pubchem_index
        smiles_index = self.smiles_index
        InChI_index = self.InChI_index
        formula_index = self.formula_index
        name_index = self.name_index
        for ele in periodic_table:
            name = ele.name.lower()
            CAS = int(ele.CAS.replace('-', '')) # Store as int for easier lookup
            obj = ChemicalMetadata(pubchemid=ele.PubChem, CAS=CAS, 
                                   formula=ele.symbol, MW=ele.MW, smiles=ele.smiles,
                                   InChI=ele.InChI, InChI_key=ele.InChI_key,
                                   iupac_name=name, 
                                   common_name=name)
            InChI_key_index[ele.InChI_key] = obj
            CAS_index[CAS] = obj
            pubchem_index[ele.PubChem] = obj
            smiles_index[ele.smiles] = obj
            InChI_index[ele.InChI] = obj
            formula_index[obj.formula] = obj
            if ele.number in homonuclear_elemental_gases:
                name_index['monatomic ' + name] = obj    
            else:
                name_index[name] = obj    

    def load(self, file_name):
        CAS_index = self.CAS_index
        name_index = self.name_index
        pubchem_index = self.pubchem_index
        smiles_index = self.smiles_index
        InChI_index = self.InChI_index
        InChI_key_index = self.InChI_key_index
        formula_index = self.formula_index
        f = open(file_name)
        for line in f:
            # This is effectively the documentation for the file format of the file
            values = line.rstrip('\n').split('\t')
            (pubchemid, CAS, formula, MW, smiles, InChI, InChI_key, iupac_name, common_name) = values[0:9]
            CAS = int(CAS.replace('-', '')) # Store as int for easier lookup
            all_names = values[7:]
            pubchemid = int(pubchemid)
            if CAS in CAS_index:
                obj = CAS_index[CAS]
            else:
                obj = ChemicalMetadata(pubchemid, CAS, formula, float(MW), smiles,
                                       InChI, InChI_key, iupac_name, common_name)
                CAS_index[CAS] = obj
                pubchem_index[pubchemid] = obj
                smiles_index[smiles] = obj
                InChI_index[InChI] = obj
                InChI_key_index[InChI_key] = obj
                formula_index[formula] = obj
            for name in all_names: 
                name = name.lower()
                if name not in name_index: name_index[name] = obj
        f.close()
    
    def search_index(self, index, key):
        try: return index[key]
        except: 
            files = self.unloaded_files
            if files:
                self.load(files.pop())
                self.search_index(index, key)
    
    def search_pubchem(self, pubchem):
        pubchem = int(pubchem)
        return self.search_index(self.pubchem_index, pubchem)
        
    def search_CAS(self, CAS):
        CAS = CAS2int(CAS)
        return self.search_index(self.CAS_index, CAS)

    def search_smiles(self, smiles):
        return self.search_index(self.smiles_index, smiles)

    def search_InChI(self, InChI):
        return self.search_index(self.InChI_index, InChI)

    def search_InChI_key(self, InChI_key):
        return self.search_index(self.InChI_key_index, InChI_key)

    def search_name(self, name):
        return self.search_index(self.name_index, name)
    
    def search_formula(self, formula):
        return self.search_index(self.formula_index, formula)


pubchem_db = ChemicalMetadataDB()

def chemical_metadata_from_any(ID):
    """
    Looks up metadata of a chemical by searching and testing for the
    string being any of the following types of chemical identifiers:
    
    * Name, in IUPAC form or common form or a synonym registered in PubChem
    * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
    * InChI key, prefixed by 'InChIKey='
    * PubChem CID, prefixed by 'PubChem='
    * SMILES (prefix with 'SMILES=' to ensure smiles parsing; ex.
      'C' will return Carbon as it is an element whereas the SMILES 
      interpretation for 'C' is methane)
    * CAS number (obsolete numbers may point to the current number)    

    If the input is an ID representing an element, the following additional 
    inputs may be specified as 
    
    * Atomic symbol (ex 'Na')
    * Atomic number (as a string)

    Parameters
    ----------
    ID : str
        One of the name formats described above

    Returns
    -------
    metadata : ChemicalMetadata

    Notes
    -----
    A LookupError is raised if the name cannot be identified. The PubChem 
    database includes a wide variety of other synonyms, but these may not be
    present for all chemcials.

    Examples
    --------
    >>> chemical_metadata_from_any('water')
    <ChemicalMetadata: 7732-18-5>
    >>> chemical_metadata_from_any('InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3')
    <ChemicalMetadata: 64-17-5>
    >>> chemical_metadata_from_any('CCCCCCCCCC')
    <ChemicalMetadata: 124-18-5>
    >>> chemical_metadata_from_any('InChIKey=LFQSCWFLJHTTHZ-UHFFFAOYSA-N')
    <ChemicalMetadata: 64-17-5>
    >>> chemical_metadata_from_any('pubchem=702')
    <ChemicalMetadata: 64-17-5>
    >>> chemical_metadata_from_any('O') # only elements can be specified by symbol
    <ChemicalMetadata: 17778-80-2>
    """
    if not ID: raise ValueError('ID cannot be empty')
    ID = ID.replace('_', ' ')
    ID_lower = ID.lower()
    
    ID_len = len(ID)
    if ID_len > 9:
        inchi_search = False
        # normal upper case is 'InChI=1S/'
        if ID_lower[0:9] == 'inchi=1s/':
            inchi_search = ID[9:]
        elif ID_lower[0:8] == 'inchi=1/':
            inchi_search = ID[8:]
        if inchi_search:
            inchi_lookup = pubchem_db.search_InChI(inchi_search)
            if inchi_lookup: return inchi_lookup
            raise LookupError('A valid InChI name was recognized, but it is not in the database')
        if ID_lower[0:9] == 'inchikey=':
            inchi_key_lookup = pubchem_db.search_InChI_key(ID[9:])
            if inchi_key_lookup: return inchi_key_lookup
            raise LookupError('A valid InChI Key was recognized, but it is not in the database')
    if ID_len > 8:
        if ID_lower[0:8] == 'pubchem=':
            pubchem_lookup = pubchem_db.search_pubchem(ID[8:])
            if pubchem_lookup: return pubchem_lookup
            raise LookupError('A PubChem integer identifier was recognized, but it is not in the database.')
    if ID_len > 7:
        if ID_lower[0:7] == 'smiles=':
            smiles_lookup = pubchem_db.search_smiles(ID[7:])
            if smiles_lookup: return smiles_lookup
            raise LookupError('A SMILES identifier was recognized, but it is not in the database.')
    
    # Permutate through various name options
    ID_search = spaceout_words(ID).lower()
    for name in (ID_lower, ID_search):
        name_lookup = pubchem_db.search_name(name)
        if name_lookup: return name_lookup
    
    if checkCAS(ID):
        CAS_lookup = pubchem_db.search_CAS(ID)
        if CAS_lookup: return CAS_lookup
        
        # Handle the case of synonyms
        CAS_alternate_loopup = pubchem_db.search_name(ID)
        if CAS_alternate_loopup: return CAS_alternate_loopup
        
        raise LookupError('a valid CAS number was recognized, but its not in the database')
    
    try: formula = serialize_formula(ID)
    except: pass
    else:
        formula_query = pubchem_db.search_formula(formula)
        if formula_query: return formula_query
    
    raise LookupError(f'chemical {repr(ID)} not recognized')


def PubChem(CASRN):
    """
    Given a CASRN in the database, obtain the PubChem database
    number of the compound.

    Parameters
    ----------
    CASRN : string
        Valid CAS number in PubChem database [-]

    Returns
    -------
    pubchem : int
        PubChem database id, as an integer [-]

    Notes
    -----
    CASRN must be an indexing key in the pubchem database.

    Examples
    --------
    >>> PubChem('7732-18-5')
    962

    References
    ----------
    .. [1] Pubchem.
    """
    return pubchem_db.search_CAS(CASRN).pubchemid



def MW(CASRN):
    """
    Given a CASRN in the database, obtain the Molecular weight of the
    compound, if it is in the database.

    Parameters
    ----------
    CASRN : string
        Valid CAS number in PubChem database

    Returns
    -------
    MolecularWeight : float

    Notes
    -----
    CASRN must be an indexing key in the pubchem database. No MW Calculation is
    performed; nor are any historical isotopic corrections applied.

    Examples
    --------
    >>> MW('7732-18-5')
    18.01528

    References
    ----------
    .. [1] Pubchem.
    """
    return pubchem_db.search_CAS(CASRN).MW


def formula(CASRN):
    """
    >>> formula('7732-18-5')
    'H2O'
    """
    return pubchem_db.search_CAS(CASRN).formula


def smiles(CASRN):
    """
    >>> smiles('7732-18-5')
    'O'
    """
    return pubchem_db.search_CAS(CASRN).smiles


def InChI(CASRN):
    """
    >>> InChI('7732-18-5')
    'H2O/h1H2'
    """
    return pubchem_db.search_CAS(CASRN).InChI


def InChI_Key(CASRN):
    """
    >>> InChI_Key('7732-18-5')
    'XLYOFNOQVPJJNP-UHFFFAOYSA-N'
    """
    return pubchem_db.search_CAS(CASRN).InChI_key


def IUPAC_name(CASRN):
    """
    >>> IUPAC_name('7732-18-5')
    'oxidane'
    """
    return pubchem_db.search_CAS(CASRN).iupac_name

def name(CASRN):
    """
    >>> name('7732-18-5')
    'water'
    """
    return pubchem_db.search_CAS(CASRN).common_name


def synonyms(CASRN):
    """
    >>> synonyms('98-00-0')
    ['furan-2-ylmethanol', 'furfuryl alcohol', '2-furanmethanol', '2-furancarbinol', '2-furylmethanol', '2-furylcarbinol', '98-00-0', '2-furanylmethanol', 'furfuranol', 'furan-2-ylmethanol', '2-furfuryl alcohol', '5-hydroxymethylfuran', 'furfural alcohol', 'alpha-furylcarbinol', '2-hydroxymethylfuran', 'furfuralcohol', 'furylcarbinol', 'furyl alcohol', '2-(hydroxymethyl)furan', 'furan-2-yl-methanol', 'furfurylalcohol', 'furfurylcarb', 'methanol, (2-furyl)-', '2-furfurylalkohol', 'furan-2-methanol', '2-furane-methanol', '2-furanmethanol, homopolymer', '(2-furyl)methanol', '2-hydroxymethylfurane', 'furylcarbinol (van)', '2-furylmethan-1-ol', '25212-86-6', '93793-62-5', 'furanmethanol', 'polyfurfuryl alcohol', 'pffa', 'poly(furfurylalcohol)', 'poly-furfuryl alcohol', '(fur-2-yl)methanol', '.alpha.-furylcarbinol', '2-hydroxymethyl-furan', 'poly(furfuryl alcohol)', '.alpha.-furfuryl alcohol', 'agn-pc-04y237', 'h159', 'omega-hydroxypoly(furan-2,5-diylmethylene)', '(2-furyl)-methanol (furfurylalcohol)', '40795-25-3', '88161-36-8']
    """
    return pubchem_db.search_CAS(CASRN).all_names

