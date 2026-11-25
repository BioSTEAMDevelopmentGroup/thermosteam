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
'''
UNIFAC groups and group assignments (data from thermo).
'''
from thermo.unifac import (
    UNIFAC_subgroup, UFSG, DOUFSG, NISTUFSG, PSRKSG,
    DDBST_UNIFAC_assignments,
    DDBST_MODIFIED_UNIFAC_assignments,
    DDBST_PSRK_assignments,
    load_group_assignments_DDBST,
)

__all__ = ('GroupCounts',
           'UNIFACGroupCounts',
           'DortmundGroupCounts',
           'NISTGroupCounts',
           'PSRKGroupCounts',
           'get_group_counts')

# Subgroups missing in thermo
DOUFSG[179] = UNIFAC_subgroup(179, 'BTI-', 85, 'BTI', 5.8504, 5.7513)
NISTUFSG[186] = UNIFAC_subgroup(186, 'CH(O)2', 74, 'Acetal', 0.9347, 0.708)


class GroupCounts(dict):
    __slots__ = ()

    def __init_subclass__(cls):
        cls.ids_by_group_name = {subgroup.group: key
                                 for key, subgroup in cls.subgroups_by_id.items()}

    def copy(self):
        return self.__class__(self)

    def set_group_counts_by_name(self, group_counts: dict, reset=True):
        if reset: self.clear()
        ids_by_group_name = self.ids_by_group_name
        for name, count in group_counts.items():
            key = ids_by_group_name[name]
            self[key] = self.get(key, 0) + count

    def _show_group_counts(self):
        if self:
            subgroups_by_id = self.subgroups_by_id
            return ', '.join([str(count) + subgroups_by_id[key].group
                              for key, count in self.items()])
        else:
            return 'Empty'

    def __str__(self):
        return f"<{self._show_group_counts()}>"

    def __repr__(self):
        return f"<{type(self).__name__}: {self._show_group_counts()}>"


class UNIFACGroupCounts(GroupCounts):
    __slots__ = ()
    subgroups_by_id = UFSG

class DortmundGroupCounts(GroupCounts):
    __slots__ = ()
    subgroups_by_id = DOUFSG

class NISTGroupCounts(GroupCounts):
    __slots__ = ()
    subgroups_by_id = NISTUFSG

class PSRKGroupCounts(GroupCounts):
    __slots__ = ()
    subgroups_by_id = PSRKSG


group_counts_classes = {
    'UNIFAC': UNIFACGroupCounts,
    'Dortmund': DortmundGroupCounts,
    'NIST': NISTGroupCounts,
    'PSRK': PSRKGroupCounts,
}
group_assignments = { # TODO: Add NIST UNIFAC group assignments
    'UNIFAC': DDBST_UNIFAC_assignments,
    'Dortmund': DDBST_MODIFIED_UNIFAC_assignments,
    'PSRK': DDBST_PSRK_assignments,
}

def load_group_assignments():
    if DDBST_UNIFAC_assignments: return # Already loaded
    load_group_assignments_DDBST()
    # Group assignments missing in thermo
    DDBST_MODIFIED_UNIFAC_assignments['PEDCQBHIVMGVHV-UHFFFAOYSA-N'] = {2: 2, 3: 1, 14: 2, 81: 1} # Glycerol

def get_group_counts(name, InChI_key):
    """
    Return group counts of a chemical (by InChI key) for the given group
    contribution method (i.e., 'UNIFAC', 'Dortmund', 'PSRK', or 'NIST').
    Group assignments are loaded from thermo on first request. Returns
    empty group counts if not available.
    """
    GroupCounts = group_counts_classes[name]
    if InChI_key is None or name not in group_assignments: return GroupCounts()
    load_group_assignments()
    return GroupCounts(group_assignments[name].get(InChI_key, ()))
