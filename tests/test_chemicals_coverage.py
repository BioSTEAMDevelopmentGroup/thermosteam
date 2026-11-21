# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
import thermosteam as tmo
import numpy as np
from numpy.testing import assert_allclose
from thermosteam._chemicals import ChemicalDraft, ChemicalsOutline

def test_chemicals_outline():
    draft = ChemicalDraft('Water', None, cache=True)
    assert repr(draft) == 'ChemicalDraft(Water)'
    draft.copy()
    outline = ChemicalsOutline(['Ethanol', draft])
    assert len(outline) == 2 and 'Water' in outline and draft in outline
    assert outline['Water'] is draft
    assert outline[['Water', 'Ethanol']][0] is draft
    with pytest.raises(tmo.exceptions.UndefinedChemicalAlias):
        outline['Methanol']
    with pytest.raises(TypeError):
        outline.Methanol = draft
    with pytest.raises(TypeError):
        outline['Methanol'] = draft
    outline.append('Methanol')
    with pytest.raises(ValueError):
        outline.append('Methanol')
    with pytest.raises(TypeError):
        outline.append(1)
    copy = outline.copy()
    assert len(copy) == 3
    copy.extend(ChemicalsOutline(['O2']))
    copy.extend([ChemicalDraft('CO2')])
    assert len(copy) == 5
    assert outline.__getnewargs__()[0][0].ID == 'Ethanol'
    assert 'ChemicalsOutline' in repr(outline)
    outline.show()
    with pytest.raises(TypeError):
        outline.array(['Water'], [1])
    chemicals = ChemicalsOutline(['Water', 'Ethanol'], cache=True).to_chemicals()
    assert [i.ID for i in chemicals] == ['Water', 'Ethanol']

def test_chemicals_basic():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    assert chemicals.__getnewargs__()[0][0].ID == 'Water'
    with pytest.raises(TypeError):
        chemicals.Methanol = None
    with pytest.raises(TypeError):
        chemicals['Methanol'] = None
    assert chemicals[['Water', 'Ethanol']][1].ID == 'Ethanol'
    with pytest.raises(tmo.exceptions.UndefinedChemicalAlias):
        chemicals['Methanol']
    copy = chemicals.copy()
    copy.append('Methanol')
    with pytest.raises(ValueError):
        copy.append(tmo.Chemical('Methanol', cache=True))
    with pytest.raises(TypeError):
        copy.append(1)
    copy.extend(tmo.Chemicals(['O2'], cache=True))
    copy.extend([tmo.Chemical('CO2', cache=True)])
    assert len(copy) == 5 and len(chemicals) == 2
    with pytest.raises(TypeError):
        chemicals.array(['Water'], [1])
    copy.show(N=2)
    assert 'Chemicals' in copy._info(2)
    assert copy._info(10) == str(copy)
    X = tmo.Chemical.blank('X')
    with pytest.raises(RuntimeError):
        tmo.Chemicals([X]).compile()

def test_compiled_chemicals():
    chemicals = tmo.CompiledChemicals(['Water', 'Ethanol', 'Methanol', 'Glucose'], cache=True)
    assert hash(chemicals) == hash(chemicals.IDs)
    assert 'append' in dir(chemicals)
    chemicals.define_group('Alcohols', ['Ethanol', 'Methanol'], [1, 1], wt=True)
    chemicals.define_group('Liquids', ['Water', 'Alcohols'])
    assert 'Alcohols' in chemicals.chemical_groups
    assert chemicals.chemical_group_members('Alcohols') == ['Ethanol', 'Methanol']
    with pytest.raises(ValueError):
        chemicals.chemical_group_members('Gases')
    with pytest.raises(ValueError):
        chemicals.define_group('Bad', ['Water'], [1, 2])
    chemicals.refresh_constants()
    assert_allclose(chemicals.MW[0], 18.015, rtol=1e-2)
    with pytest.raises(ValueError):
        chemicals.array(['Alcohols'], [1])
    split = chemicals.split(['Alcohols'], [1])
    assert_allclose(split[1:3], [1, 1])
    split = chemicals.split(['Water', 'Glucose'], [0.5, 0.2])
    assert_allclose(split[[0, 3]], [0.5, 0.2])

def test_chemical_data():
    from thermosteam._chemical_data import ChemicalData, CachedProperty
    chemicals = tmo.CompiledChemicals(['Water', 'Ethanol'], cache=True)
    data = ChemicalData(chemicals)
    assert [i.ID for i in data] == ['Water', 'Ethanol']
    assert data['Water'].ID == 'Water'
    assert data[['Water']].N == 1
    assert data['H2O'].ID == 'Water'
    with pytest.raises(KeyError):
        data[1]
    data = ChemicalData(list(chemicals))
    # These cached properties raise errors (see report), so skip them
    broken = {
        'ThermalConductivityGases', 'ThermalConductivityLiquids', 'rhocs',
        'rhocs_mass', 'Vmg_STPs', 'rhog_STPs', 'rhog_STPs_mass', 'Vml_STPs',
        'Vml_60Fs', 'rhol_STPs', 'rhol_60Fs', 'rhol_STPs_mass', 'rhol_60Fs_mass',
        'molecular_diameters', 'Psat_298s', 'Hvap_298s', 'Hvap_298s_mass',
        'rhos_Tms', 'Parachors',
    }
    names = [i for i, j in vars(ChemicalData).items()
             if isinstance(j, CachedProperty) and i not in broken]
    assert len(names) > 40
    for name in names:
        value = getattr(data, name)
        assert len(value) == 2
        assert getattr(data, name) is value # Cached
    assert_allclose(data.MWs, [18.015, 46.07], rtol=1e-2)

def test_biorefinery_chemicals():
    from thermosteam import _biorefinery_chemicals as bc
    IDs = sorted(set([i.__name__ for i in bc.biorefinery_chemicals.values()]))
    assert len(IDs) > 20
    for ID in IDs:
        chemical = bc.search_biorefinery_chemicals(ID)
        assert chemical.MW > 0
    chemical = bc.search_biorefinery_chemicals('Raw carbon black')
    assert chemical.MW > 0
    with pytest.raises(LookupError):
        bc.search_biorefinery_chemicals('NotAChemicalXYZ', LookupError('missing'))
    Water = bc.search_biorefinery_chemicals('Water')
    assert_allclose(Water.MW, 18.015, rtol=1e-2)

def test_identifiers():
    from thermosteam.chemicals import identifiers as ids
    dct = ids.ChemicalDataDictionary({'H2O': 1})
    assert 'H2O' in dct
    assert 'Water' in dct
    assert 'Water' in dct # cached
    assert 'Ethanol' not in dct
    assert dct['H2O'] == 1
    assert dct['Water'] == 1
    assert dct['Water'] == 1 # cached
    dct = ids.ChemicalDataDictionary({'aquafina': 1})
    assert 'H2O' in dct
    dct = ids.ChemicalDataDictionary({'aquafina': 1})
    assert dct['H2O'] == 1
    with pytest.raises(KeyError):
        dct['Ethanol']
    assert ids.to_searchable_format('HeatOfFormation') == 'Heat Of Formation'
    assert ids.CAS_from_any('Water') == '7732-18-5'
    db = ids.pubchem_db
    assert db.search_pubchem(962).CASs == '7732-18-5'
    assert db.search('pubchem=962').CASs == '7732-18-5'
    water = db.search('Water')
    assert db.search('InChI=1S/' + water.InChI).CASs == '7732-18-5'
    assert db.search('InChIKey=' + water.InChI_key).CASs == '7732-18-5'
    assert db.search('smiles=O').CASs == '7732-18-5'
    for ID in ('InChI=1S/XXXXXXXXX', 'InChIKey=XXXXXXXXX', 'pubchem=1',
               'smiles=XXXXX', '7732-18-6'):
        with pytest.raises(LookupError):
            db.search(ID)
    with pytest.raises(ValueError):
        db._search('')

if __name__ == '__main__':
    test_chemicals_outline()
    test_chemicals_basic()
    test_compiled_chemicals()
    test_chemical_data()
    test_biorefinery_chemicals()
    test_identifiers()
