# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the identifiers module from the chemicals's library:
# https://github.com/CalebBell/chemicals
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.
import re
import os
import chemicals
from chemicals.elements import (
    periodic_table, 
    homonuclear_elemental_gases,  
    serialize_formula
)
from chemicals.identifiers import (
    CAS_to_int,
    ChemicalMetadata,
    check_CAS,
)
from ..utils import forward
from chemicals import identifiers
folder = identifiers.folder
searchable_format = re.compile(r"\B([A-Z])")

@forward(identifiers)
def spaceout_words(ID):
    return searchable_format.sub(r" \1", ID)

@forward(identifiers)
def to_searchable_format(ID):    
    return spaceout_words(ID).replace('_', ' ')

@forward(identifiers)
def CAS_from_any(ID):
    return pubchem_db.search(ID).CASs

@forward(identifiers)
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
    
    cache = {}
    
    def __init__(self, 
                 files=[os.path.join(folder, 'chemical identifiers pubchem large.tsv'),
                        os.path.join(folder, 'Inorganic db.tsv'),
                        os.path.join(folder, 'Anion db.tsv'),
                        os.path.join(folder, 'Cation db.tsv'),
                        os.path.join(folder, 'chemical identifiers example user db.tsv'),
                        os.path.join(folder, 'chemical identifiers pubchem small.tsv')]):                
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
                                   common_name=name,
                                   synonyms=())
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
            synonyms = values[7:]
            pubchemid = int(pubchemid)
            if CAS in CAS_index:
                obj = CAS_index[CAS]
            else:
                obj = ChemicalMetadata(pubchemid, CAS, formula, float(MW), smiles,
                                       InChI, InChI_key, iupac_name, common_name, synonyms)
                CAS_index[CAS] = obj
                pubchem_index[pubchemid] = obj
                smiles_index[smiles] = obj
                InChI_index[InChI] = obj
                InChI_key_index[InChI_key] = obj
                formula_index[formula] = obj
            for name in synonyms: 
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
        return self.search_index(self.pubchem_index, int(pubchem))
        
    def search_CAS(self, CAS):
        return self.search_index(self.CAS_index, CAS_to_int(CAS))

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

    def search(self, ID):
        cache = self.cache
        if ID in cache:
            return cache[ID]
        else:
            if len(cache) > 100: cache.clear()
            cache[ID] = obj = self._search(ID)
            return obj

    def _search(self, ID):
        if not ID: raise ValueError('ID cannot be empty')
        
        if check_CAS(ID):
            CAS_lookup = self.search_CAS(ID)
            if CAS_lookup: return CAS_lookup
            
            # Handle the case of synonyms
            CAS_alternate_loopup = self.search_name(ID)
            if CAS_alternate_loopup: return CAS_alternate_loopup
            
            raise LookupError('a valid CAS number was recognized, but its not in the database')
        
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
                inchi_lookup = self.search_InChI(inchi_search)
                if inchi_lookup: return inchi_lookup
                raise LookupError('A valid InChI name was recognized, but it is not in the database')
            if ID_lower[0:9] == 'inchikey=':
                inchi_key_lookup = self.search_InChI_key(ID[9:])
                if inchi_key_lookup: return inchi_key_lookup
                raise LookupError('A valid InChI Key was recognized, but it is not in the database')
        if ID_len > 8:
            if ID_lower[0:8] == 'pubchem=':
                pubchem_lookup = self.search_pubchem(ID[8:])
                if pubchem_lookup: return pubchem_lookup
                raise LookupError('A PubChem integer identifier was recognized, but it is not in the database.')
        if ID_len > 7:
            if ID_lower[0:7] == 'smiles=':
                smiles_lookup = self.search_smiles(ID[7:])
                if smiles_lookup: return smiles_lookup
                raise LookupError('A SMILES identifier was recognized, but it is not in the database.')
        
        # Permutate through various name options
        ID_search = spaceout_words(ID).lower()
        for name in (ID_lower, ID_search):
            name_lookup = self.search_name(name)
            if name_lookup: return name_lookup
        
        try: formula = serialize_formula(ID)
        except: pass
        else:
            formula_query = self.search_formula(formula)
            if formula_query: return formula_query
        
        raise LookupError(f'chemical {repr(ID)} not recognized')

chemicals.CAS_from_any = CAS_from_any
chemicals.pubchem_db = identifiers.pubchem_db = pubchem_db = ChemicalMetadataDB()
identifiers._pubchem_db_loaded = True
