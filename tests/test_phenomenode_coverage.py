# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Tests for phenomenode graphs of a single flash drum (s1) at specified
temperature and pressure, solved with a phenomena-based simulation loop.

Units: flows [mol/s], temperature [K], enthalpy [J/mol], heat capacity
[J/mol/K], duty [W], K = y/x [-], B = V/L [-]. All coefficients are
plugged-in values.

"""
import pytest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from thermosteam import phenomenode as pn

graphviz_available = shutil.which('dot') is not None
requires_graphviz = pytest.mark.skipif(
    not graphviz_available, reason='Graphviz executables are not available'
)

chemicals = ('Ethanol', 'Water')

class Flash:
    """
    Flash drum with specified feed, temperature, pressure, and K.

    * vle_phenomenode_s1: x_1 = (1 - K_2) / (K_1 - K_2); y = K x
    * bulk_material_balance_s1: lever rule, V = F (z_1 - x_1) / (y_1 - x_1); L = F - V; B = V / L
    * material_balance_s1: l = f / (1 + K B); v = f - l
    * energy_balance_s1: Q = H_vapor + H_liquid - H_feed

    """
    feed = np.array([40., 60.]) # [mol/s]
    T_feed = 340. # [K]
    T = 363.15 # [K]
    T_ref = 298.15 # [K]
    K = np.array([1.57, 0.65]) # [-]
    Cp_l = np.array([112., 75.3]) # [J/mol/K]
    Cp_g = np.array([65.4, 33.6]) # [J/mol/K]
    Hvap = np.array([38.6e3, 40.7e3]) # [J/mol]

    def __init__(self):
        F = self.feed.sum()
        z = self.feed / F
        # Initial guess
        self.values = values = {
            'x_s1': z,
            'y_s1': z,
            'B_s1': 1.,
            'V_s1': F / 2,
            'L_s1': F / 2,
            'vapor_s1': self.feed / 2,
            'liquid_s1': self.feed / 2,
            'Q_s1': 0.,
        }
        v = {i: pn.VariableNode(i, lambda i=i: values[i]) for i in values}
        self.variables = v
        self.equations = eqs = {i: pn.EquationNode(i) for i in self.sequence}
        # Feed, temperature, and K are specified, so they have no variable nodes
        eqs['vle_phenomenode_s1'].set_equations(
            outputs=[v['x_s1'], v['y_s1']],
        )
        eqs['bulk_material_balance_s1'].set_equations(
            inputs=[v['x_s1'], v['y_s1']],
            outputs=[v['V_s1'], v['L_s1'], v['B_s1']],
            tracked_outputs=[v['B_s1']], # V and L are only intermediates
        )
        eqs['material_balance_s1'].set_equations(
            inputs=[None, v['B_s1']],
            outputs=[v['vapor_s1'], v['liquid_s1']],
        )
        eqs['energy_balance_s1'].set_equations(
            inputs=[v['vapor_s1'], v['liquid_s1']], outputs=[v['Q_s1']],
        )

    #: Order in which equations are solved
    sequence = (
        'vle_phenomenode_s1',
        'bulk_material_balance_s1',
        'material_balance_s1',
        'energy_balance_s1',
    )

    def vle_phenomenode_s1(self):
        values = self.values
        K1, K2 = self.K
        x1 = (1 - K2) / (K1 - K2)
        values['x_s1'] = x = np.array([x1, 1 - x1])
        values['y_s1'] = self.K * x

    def bulk_material_balance_s1(self):
        values = self.values
        x1 = values['x_s1'][0]
        y1 = values['y_s1'][0]
        F = self.feed.sum()
        z1 = self.feed[0] / F
        values['V_s1'] = V = F * (z1 - x1) / (y1 - x1)
        values['L_s1'] = L = F - V
        values['B_s1'] = V / L

    def material_balance_s1(self):
        values = self.values
        values['liquid_s1'] = liquid = self.feed / (1 + self.K * values['B_s1'])
        values['vapor_s1'] = self.feed - liquid

    def energy_balance_s1(self):
        values = self.values
        dT = self.T - self.T_ref
        H_feed = (self.feed * self.Cp_l * (self.T_feed - self.T_ref)).sum()
        H_liquid = (values['liquid_s1'] * self.Cp_l * dT).sum()
        H_vapor = (values['vapor_s1'] * (self.Hvap + self.Cp_g * dT)).sum()
        values['Q_s1'] = H_vapor + H_liquid - H_feed

    def simulate(self):
        # Record the initial guess, then solve one phenomenon at a time (one iteration)
        get_values = lambda: [np.copy(i.get_value()) for i in self.variables.values()]
        history = [get_values()]
        for name in self.sequence: getattr(self, name)()
        history.append(get_values())
        self.time = np.arange(len(history), dtype=float)
        columns = [
            (name, j) for name, value in zip(self.variables, history[-1])
            for j in (chemicals if np.ndim(value) else ['-'])
        ]
        self.variable_profiles = pd.DataFrame(
            [np.hstack(i) for i in history],
            columns=pd.MultiIndex.from_tuples(columns),
        )

def create_flash():
    flash = Flash()
    flash.simulate()
    return flash

def test_simulation():
    # One iteration of the simulation loop gives a solution that satisfies
    # the mass, bulk, VLE, and energy equations.
    flash = create_flash()
    values = flash.values
    vapor = values['vapor_s1']
    liquid = values['liquid_s1']
    V = values['V_s1']
    L = values['L_s1']
    assert_allclose(vapor + liquid, flash.feed) # Material balance
    assert_allclose(V / L, values['B_s1']) # Bulk balance
    assert_allclose(V + L, flash.feed.sum())
    assert_allclose(liquid / L, values['x_s1']) # VLE
    assert_allclose(vapor / V, values['y_s1'])
    assert_allclose(flash.K * values['x_s1'], values['y_s1'])
    assert 0 < V < flash.feed.sum() # Two phases
    assert values['Q_s1'] > 0 # Heat is needed to vaporize

def test_nodes():
    # Equations are correctly connected through the variables they share.
    flash = Flash()
    variables, equations = flash.variables, flash.equations
    with pytest.raises(ValueError): pn.VariableNode('x', 1)
    var = variables['B_s1']
    assert np.isnan(var.last_value)
    assert var.get_value() == 1. and var.last_value == 1.
    assert repr(var) == 'B_s1'
    eq = equations['material_balance_s1']
    assert repr(eq) == 'material_balance_s1'
    assert eq.category == 'material'
    assert equations['bulk_material_balance_s1'].category == 'material'
    assert equations['energy_balance_s1'].category == 'energy'
    assert equations['vle_phenomenode_s1'].category == 'vle'
    with pytest.raises(AttributeError): pn.EquationNode('unknown').category
    assert eq.inputs == [variables['B_s1']] # None filtered out
    assert eq.tracked_outputs == eq.outputs
    bulk = equations['bulk_material_balance_s1']
    assert bulk.tracked_outputs == [variables['B_s1']]
    T = pn.VariableNode('T', lambda: 300.)
    duplicate_eq = pn.EquationNode('energy_balance_x')
    duplicate_eq.set_equations(inputs=[T, T])
    assert duplicate_eq.inputs == [T] # Duplicates filtered out
    assert eq.neighbors == set([eq, bulk, equations['energy_balance_s1']])
    assert len(eq.get_edges()) == 3
    assert len(eq.get_edges(outputs=False)) == 1
    assert len(eq.get_edges(inputs=False)) == 2
    assert len(eq.get_edges(False, False)) == 0
    assert eq.get_edges()[0].name == ('material_balance_s1', 'B_s1')
    with pytest.raises(ValueError):
        pn.EquationNode('material_x').set_equations(inputs=[eq])
    assert pn.get_order([3., 1., 1., 2.]) == [1, 1, 3, 0]
    criteria = pn.Criteria('phenomena')
    assert criteria.names == ['phenomen']
    assert criteria.match('vle_phenomenode_s1')
    selection = pn.CriteriaSelection([('material_balance',), ('energy_balance',)])
    assert selection.get_criteria('bulk_material_balance_s1').names == ['material_balance']
    with pytest.raises(ValueError): selection.get_criteria('reaction_s1')
    dotfile = pn.DotFile('graph')
    assert dotfile.file == 'graph.dot'
    assert pn.DotFile('graph.dot').file == 'graph.dot'
    assert dotfile.subfile('a') == 'graph_a.dot'
    assert 'graph.dot' in repr(dotfile)

@requires_graphviz
def test_phenome_graph():
    # Equations are grouped by phenomena (material, energy, VLE) into a graph.
    flash = create_flash()
    equations = flash.equations
    criteria_selection = [
        ('material_balance',), ('energy_balance',), ('phenomena', 'vle'),
    ]
    graph = pn.PhenomeGraph(
        list(equations.values()), criteria_selection, ['s1'],
        flash.variable_profiles, flash.time,
    )
    # Component and bulk material balances are grouped together
    nodes = {i.category: i for i in graph.phenomenodes}
    assert len(graph.phenomenodes) == len(nodes) == 3
    assert nodes['material'].equations == set([
        equations['material_balance_s1'], equations['bulk_material_balance_s1'],
    ])
    assert len(graph.edges) == 2 # VLE and energy only connect through material balances
    for profile in graph.profiles:
        assert_allclose(profile, [100, 0]) # Solved in one iteration
    node = graph.phenomenodes[0]
    assert repr(node) == node.name
    assert node.size == 1
    with tempfile.TemporaryDirectory() as folder:
        file = os.path.join(folder, 'graph.dot')
        graph.load(file) # File does not exist; it is created
        assert os.path.exists(file)
        graph.load(file) # File exists
        graph.write(os.path.join(folder, 'graph.png'))
        assert os.path.exists(os.path.join(folder, 'graph.png'))
    graph.load(maxiter='10', damping='0.5', K='0.3')
    graph.display()

def create_bipartite_graph(name='graph', file=None, **kwargs):
    flash = create_flash()
    equations = list(flash.equations.values())
    variables = list(flash.variables.values())
    edges = list(pn.BipartitePhenomeGraph.get_node_edges(equations))
    # Normalized error of each variable over the simulation
    profiles = flash.variable_profiles
    errors = {
        i.name: np.abs(profiles[i.name] - profiles[i.name].iloc[-1]).values.sum(axis=1)
        for i in variables
    }
    errors = {i: j / j.max() for i, j in errors.items()}
    equation_profiles = [
        sum([errors[j.name] for j in i.outputs]) / len(i.outputs)
        for i in equations
    ]
    graph = pn.BipartitePhenomeGraph(
        name, equations, variables, edges,
        equation_profiles,
        [errors[i.name] for i in variables],
        [errors[i.variable.name] for i in edges],
        flash.time, file, **kwargs,
    )
    return graph

@requires_graphviz
def test_bipartite_phenome_graph():
    # Equations and variables are drawn as a graph.
    graph = create_bipartite_graph(maxiter='10')
    assert repr(graph) == 'graph'
    assert pn.PhenomenaGraph is pn.BipartitePhenomeGraph
    # Subgraph inherits positions from parent
    equations = [i for i in graph.equations if 'material' in i.name]
    variables = [i for i in graph.variables if any([i in j.outputs for j in equations])]
    edges = [i for i in pn.BipartitePhenomeGraph.get_node_edges(equations) if i.variable in variables]
    graph.subgraph('material', equations, variables, edges, None, None, None)
    assert len(graph.subgraphs) == 1
    assert graph.subgraphs[0].pydot is not None
    # Clustering by unit
    assert graph.create_digraph(subgraph_units=['s1']) is not None
    with tempfile.TemporaryDirectory() as folder:
        file = os.path.join(folder, 'bipartite.dot')
        graph.load(file) # File does not exist; it is created
        assert os.path.exists(file)
        graph.load(file) # File exists
        png = os.path.join(folder, 'bipartite.png')
        graph.write(png)
        assert os.path.exists(png)
        graph.write(png, pydot=False)
        assert os.path.exists(png)
    graph.display()

def temporary_directory_without_zeros():
    # Frame file names are created by replacing '0' in the path with the frame number
    for i in range(100):
        folder = tempfile.TemporaryDirectory()
        if '0' not in folder.name: return folder
        folder.cleanup()
    raise RuntimeError('could not create temporary directory')

@requires_graphviz
def test_bipartite_convergence_gif():
    # Convergence of the equations is animated as a gif.
    graph = create_bipartite_graph(maxiter='10')
    reference = {j.name: i for i, j in enumerate(graph.equations)}
    original_folder = pn.folder
    with temporary_directory_without_zeros() as folder:
        pn.folder = folder
        try:
            # Only render the first and last iterations to keep the test fast
            profiles = [i[[0, -1]] for i in graph.equation_profiles]
            file = os.path.join(folder, 'convergence')
            graph.convergence_gif(reference, graph.time, profiles, file, total_duration=1)
            assert os.path.exists(file + '.gif')
            profiles = [i[[1]] for i in graph.equation_profiles]
            graph.convergence_gif(reference, graph.time, profiles, fps=2)
            assert os.path.exists(os.path.join(folder, 'temp.gif'))
            graph.convergence_gif(reference, graph.time, profiles, file + '.gif')
            with pytest.raises(NotImplementedError):
                graph.convergence_gif(reference, graph.time, profiles, interpolate=True)
        finally:
            pn.folder = original_folder

if __name__ == '__main__':
    test_simulation()
    test_nodes()
    if graphviz_available: test_phenome_graph()
    if graphviz_available: test_bipartite_phenome_graph()
    if graphviz_available: test_bipartite_convergence_gif()
