# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Binary interaction parameters (kij) of cubic equations of state (EOS).

Peng-Robinson (PR) and Soave-Redlich-Kwong (SRK) interaction parameters are
from ChemSep (DECHEMA parameters; Artistic License 2.0, see
data/ChemSep_Artistic_license_2_0.txt). Missing PR interaction parameters are
estimated with the E-PPR78 group contribution method [2]_ using ChemSep's
mapping of UNIFAC subgroups to PPR78 groups (data/chemsep_fac2ppr78.txt).
Missing SRK interaction parameters are estimated from PR interaction
parameters using the PR2SRK relationship [1]_.

References
----------
.. [1] Jaubert, J.-N.; Privat, R. Relationship between the Binary Interaction
   Parameters (kij) of the Peng-Robinson and Those of a Soave-Redlich-Kwong
   Equation of State: Application to the Definition of the PR2SRK Model.
   Fluid Phase Equilib. 2010, 295 (1), 26-37.
.. [2] Qian, J.-W.; Privat, R.; Jaubert, J.-N. Predicting the Phase
   Equilibria, Critical Phenomena, and Mixing Enthalpies of Binary Aqueous
   Systems Containing Alkanes, Cycloalkanes, Aromatics, Alkenes, and Gases
   (N2, CO2, H2S, H2) with the PPR78 Equation of State. Ind. Eng. Chem. Res.
   2013, 52 (46), 16457-16490.

"""
import os
import re
from math import log, sqrt
from thermo.interaction_parameters import IPDB
from thermo.eos import PR, SRK
from thermo.group_contribution.ppr78 import PPR78_kij
from chemicals.identifiers import sorted_CAS_key

__all__ = ('PR2SRK_kij', 'PPR78_groups', 'PR_kij', 'SRK_kij',
           'PR_kijs', 'SRK_kijs', 'get_kijs')

data_folder = os.path.join(os.path.dirname(__file__), 'data')

IPDB.load_json(os.path.join(data_folder, 'chemsep_srk.json'), 'ChemSep SRK')

def load_UNIFAC_to_PPR78():
    # Return a dictionary of UNIFAC subgroup numbers (as in thermo) to
    # PPR78 group counts and whether the groups change in rings.
    # ChemSep numbers the C=C subgroup as 9, so subgroups numbered 9-69
    # elsewhere are numbered 10-70 in ChemSep.
    UNIFAC_to_PPR78 = {}
    with open(os.path.join(data_folder, 'chemsep_fac2ppr78.txt')) as file:
        next(file) # Header
        for line in file:
            if line.startswith('#'): continue
            subgroup, *values = line.split()
            subgroup = int(subgroup)
            if values[0] == '0': continue # No PPR78 groups
            if values[0] == 'same':
                groups = {subgroup: 1}
            else: # Pairs of group counts and PPR78 group numbers
                counts = [int(i) for i in values if not i.startswith('fix')]
                groups = dict(zip(counts[1::2], counts[::2]))
            cyclic = values[-1].startswith('fix_cyclic')
            if subgroup == 9:
                subgroup = 70
            elif 9 < subgroup <= 70:
                subgroup -= 1
            UNIFAC_to_PPR78[subgroup] = (groups, cyclic)
    return UNIFAC_to_PPR78

UNIFAC_to_PPR78 = load_UNIFAC_to_PPR78()

#: PPR78 groups of molecules by CAS number.
PPR78_molecules = {
    '74-82-8': 5, # Methane
    '74-84-0': 6, # Ethane
    '124-38-9': 12, # Carbon dioxide
    '7727-37-9': 13, # Nitrogen
    '7783-06-4': 14, # Hydrogen sulfide
    '7732-18-5': 16, # Water
    '74-85-1': 17, # Ethylene
    '1333-74-0': 21, # Hydrogen
    '630-08-0': 28, # Carbon monoxide
    '7440-59-7': 29, # Helium
    '7440-37-1': 30, # Argon
    '7446-09-5': 31, # Sulfur dioxide
    '7782-44-7': 32, # Oxygen
    '10102-43-9': 33, # Nitric oxide
    '463-58-1': 34, # Carbonyl sulfide
    '7664-41-7': 35, # Ammonia
    '10024-97-2': 37, # Nitrous oxide
    '74-86-2': 38, # Acetylene
}

#: Constants of the excess Gibbs energy at infinite pressure of cubic EOS
#: with van der Waals one-fluid mixing rules:
#: Lambda = ln[(1 + d1)/(1 + d2)] / (d1 - d2), where V^2 + (d1 + d2)bV + d1*d2*b^2
#: is the denominator of the attractive term.
Lambda_PR = log((2. + sqrt(2.)) / (2. - sqrt(2.))) / (2. * sqrt(2.))
Lambda_SRK = log(2.)

#: Default temperature [K] at which PR interaction parameters are estimated
#: and converted when the temperature of the data is not available.
T_default = 298.15

def PR2SRK_kij(kij, T, Tc1, Pc1, omega1, Tc2, Pc2, omega2):
    r"""
    Return the binary interaction parameter of the SRK EOS which gives the
    same excess Gibbs energy at infinite pressure as the PR EOS with the
    given binary interaction parameter (PR2SRK) [1]_:

    .. math::

        k_{ij}^{SRK} = \frac{E_{ij}^{SRK} - (\delta_i^{SRK} - \delta_j^{SRK})^2}
        {2\delta_i^{SRK}\delta_j^{SRK}}

        E_{ij}^{SRK} = \frac{\Lambda^{PR}\Omega_b^{PR}}{\Lambda^{SRK}\Omega_b^{SRK}}
        \left[(\delta_i^{PR} - \delta_j^{PR})^2 + 2\delta_i^{PR}\delta_j^{PR} k_{ij}^{PR}\right]

        \delta_i = \frac{\sqrt{a_i(T)}}{b_i}

    Parameters
    ----------
    kij : float
        PR binary interaction parameter.
    T : float
        Temperature [K].
    Tc1, Tc2 : float
        Critical temperatures [K].
    Pc1, Pc2 : float
        Critical pressures [Pa].
    omega1, omega2 : float
        Acentric factors.

    Examples
    --------
    >>> from thermosteam.equilibrium import PR2SRK_kij
    >>> # Nitrogen/ethane at 200 K
    >>> round(PR2SRK_kij(0.0533, 200., 126.2, 3394387., 0.0372, 305.32, 4872000., 0.0995), 4)
    0.0427

    """
    deltas = {}
    for EOS in (PR, SRK):
        eos1 = EOS(Tc=Tc1, Pc=Pc1, omega=omega1, T=T, P=101325.)
        eos2 = EOS(Tc=Tc2, Pc=Pc2, omega=omega2, T=T, P=101325.)
        deltas[EOS] = (sqrt(eos1.a_alpha) / eos1.b, sqrt(eos2.a_alpha) / eos2.b)
    d1, d2 = deltas[PR]
    E = (d1 - d2) ** 2 + 2. * d1 * d2 * kij
    E *= (Lambda_PR * PR.c2) / (Lambda_SRK * SRK.c2)
    d1, d2 = deltas[SRK]
    return (E - (d1 - d2) ** 2) / (2. * d1 * d2)

def has_ring(smiles):
    # Ring closures are digits outside brackets
    return any([i.isdigit() for i in re.sub(r'\[[^\]]*\]', '', smiles)])

def PPR78_groups(chemical):
    """
    Return PPR78 group counts (by group number) of a chemical using ChemSep's
    mapping of UNIFAC subgroups to PPR78 groups, or None if not available.
    Chemicals with groups that change in rings (e.g., cycloalkanes) are not
    supported.

    Examples
    --------
    >>> import thermosteam as tmo
    >>> from thermosteam.equilibrium import PPR78_groups
    >>> PPR78_groups(tmo.Chemical('Toluene'))
    {7: 5, 8: 1, 1: 1}
    >>> PPR78_groups(tmo.Chemical('CO2'))
    {12: 1}

    """
    if chemical.CAS in PPR78_molecules: return {PPR78_molecules[chemical.CAS]: 1}
    UNIFAC = chemical.UNIFAC
    if not UNIFAC: return None
    ring = has_ring(chemical.smiles) if chemical.smiles else True
    groups = {}
    for subgroup, count in UNIFAC.items():
        if subgroup not in UNIFAC_to_PPR78: return None
        PPR78_counts, cyclic = UNIFAC_to_PPR78[subgroup]
        if cyclic and ring: return None
        for group, n in PPR78_counts.items():
            groups[group] = groups.get(group, 0) + n * count
    return groups

def PR_kij(chemical1, chemical2):
    """
    Return the PR binary interaction parameter of two chemicals and the
    temperature [K] it applies to, or None if not available. The interaction
    parameter is taken from ChemSep PR data (at the temperature of the data,
    or 298.15 K if not available), or estimated with E-PPR78 (at 298.15 K).

    Parameters
    ----------
    chemical1, chemical2 : :class:`~thermosteam.Chemical`

    """
    key = ' '.join(sorted_CAS_key([chemical1.CAS, chemical2.CAS]))
    PR_data = IPDB.tables['ChemSep PR']
    if key in PR_data:
        data = PR_data[key]
        return data['kij'], data['T'] or data['Tmin'] or T_default
    groups1 = PPR78_groups(chemical1)
    groups2 = PPR78_groups(chemical2)
    if groups1 and groups2:
        a, b = chemical1, chemical2
        kij = PPR78_kij(T_default, groups1, groups2, a.Tc, a.Pc, a.omega,
                        b.Tc, b.Pc, b.omega, version='extended', string=False)
        return kij, T_default

def SRK_kij(chemical1, chemical2):
    """
    Return the SRK binary interaction parameter of two chemicals. The
    interaction parameter is taken from ChemSep SRK data if available;
    otherwise, it is estimated from the PR interaction parameter (see
    :func:`~thermosteam.equilibrium.PR_kij`) using the PR2SRK relationship.
    Return zero if none of these are available.

    Parameters
    ----------
    chemical1, chemical2 : :class:`~thermosteam.Chemical`

    """
    key = ' '.join(sorted_CAS_key([chemical1.CAS, chemical2.CAS]))
    SRK_data = IPDB.tables['ChemSep SRK']
    if key in SRK_data: return SRK_data[key]['kij']
    PR = PR_kij(chemical1, chemical2)
    if PR is None: return 0.
    kij, T = PR
    a, b = chemical1, chemical2
    return PR2SRK_kij(kij, T, a.Tc, a.Pc, a.omega, b.Tc, b.Pc, b.omega)

def PR_kijs(chemicals):
    """
    Return a matrix of PR binary interaction parameters (see
    :func:`~thermosteam.equilibrium.PR_kij`). Missing interaction parameters
    are zero.

    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    """
    chemicals = tuple(chemicals)
    N = len(chemicals)
    kijs = [[0.] * N for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            PR = PR_kij(chemicals[i], chemicals[j])
            if PR is None: continue
            kij, T = PR
            kijs[i][j] = kijs[j][i] = kij
    return kijs

def SRK_kijs(chemicals):
    """
    Return a matrix of SRK binary interaction parameters (see
    :func:`~thermosteam.equilibrium.SRK_kij`).

    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    """
    chemicals = tuple(chemicals)
    N = len(chemicals)
    kijs = [[0.] * N for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            kijs[i][j] = kijs[j][i] = SRK_kij(chemicals[i], chemicals[j])
    return kijs

def get_kijs(name, chemicals):
    """
    Return a matrix of binary interaction parameters from a ChemSep database
    (e.g., 'ChemSep PR' or 'ChemSep SRK').

    Parameters
    ----------
    name : str
        Name of database.
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    """
    if name == 'ChemSep PR':
        return PR_kijs(chemicals)
    elif name == 'ChemSep SRK':
        return SRK_kijs(chemicals)
    else:
        return IPDB.get_ip_asymmetric_matrix(name, [i.CAS for i in chemicals], 'kij')
