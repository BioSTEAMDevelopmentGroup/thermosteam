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

def create_network():
    values = {'mol_s1': 1., 'T_s1': 300., 'mol_s2': 2., 'T_s2': 310.}
    variables = {i: pn.VariableNode(i, lambda i=i: values[i]) for i in values}
    names = (
        'material_balance_s1', 'energy_balance_s1',
        'phenomena_vle_s1', 'material_balance_s2',
        'energy_balance_s2',
    )
    equations = {i: pn.EquationNode(i) for i in names}
    v = variables
    equations['material_balance_s1'].set_equations(
        inputs=[v['T_s1'], None, v['T_s1']], outputs=[v['mol_s1']],
    )
    equations['energy_balance_s1'].set_equations(
        inputs=[v['mol_s1']], outputs=[v['T_s1']], tracked_outputs=[v['T_s1']],
    )
    equations['phenomena_vle_s1'].set_equations(
        inputs=[v['T_s1'], v['mol_s1']], outputs=[v['mol_s2']],
    )
    equations['material_balance_s2'].set_equations(
        inputs=[v['mol_s1']], outputs=[], tracked_outputs=[v['mol_s2']],
    )
    equations['energy_balance_s2'].set_equations(
        inputs=[v['mol_s2']], outputs=[v['T_s2']],
    )
    return variables, equations

def test_nodes():
    variables, equations = create_network()
    with pytest.raises(ValueError): pn.VariableNode('x', 1)
    var = variables['mol_s1']
    assert var.get_value() == 1. and var.last_value == 1.
    assert repr(var) == 'mol_s1'
    eq = equations['material_balance_s1']
    assert repr(eq) == 'material_balance_s1'
    assert eq.category == 'material'
    assert equations['phenomena_vle_s1'].category == 'vle'
    with pytest.raises(AttributeError): pn.EquationNode('unknown').category
    assert eq.variables == (variables['T_s1'], variables['mol_s1'])
    assert equations['energy_balance_s1'] in eq.neighbors
    assert len(eq.get_edges()) == 2
    assert len(eq.get_edges(outputs=False)) == 1
    assert len(eq.get_edges(inputs=False)) == 1
    assert len(eq.get_edges(False, False)) == 0
    edge = eq.get_edges()[0]
    assert edge.name == ('material_balance_s1', 'T_s1')
    with pytest.raises(ValueError):
        pn.EquationNode('material_x').set_equations(inputs=[eq])
    assert pn.get_order([3., 1., 1., 2.]) == [1, 1, 3, 0]
    criteria = pn.Criteria('phenomena')
    assert criteria.names == ['phenomen']
    assert criteria.match('phenomenode_vle')
    selection = pn.CriteriaSelection([('material_balance',), ('energy_balance',)])
    assert selection.get_criteria('energy_balance_s1').names == ['energy_balance']
    with pytest.raises(ValueError): selection.get_criteria('reaction_s1')
    dotfile = pn.DotFile('graph')
    assert dotfile.file == 'graph.dot'
    assert pn.DotFile('graph.dot').file == 'graph.dot'
    assert dotfile.subfile('a') == 'graph_a.dot'
    assert 'graph.dot' in repr(dotfile)

@requires_graphviz
def test_phenome_graph():
    variables, equations = create_network()
    criteria_selection = [
        ('material_balance',), ('energy_balance',), ('phenomena', 'vle'),
    ]
    time = np.array([0., 1., 2.])
    variable_profiles = pd.DataFrame(
        [[1.5, 320., 2.5, 330.], [1.1, 305, 2.2, 315], [1., 300., 2., 310.]],
        columns=['mol_s1', 'T_s1', 'mol_s2', 'T_s2'],
    )
    graph = pn.PhenomeGraph(
        list(equations.values()), criteria_selection, ['s1', 's2'],
        variable_profiles, time,
    )
    assert len(graph.phenomenodes) == 4
    assert len(graph.profiles) == 4
    for profile in graph.profiles:
        assert np.isfinite(profile).all()
        assert (0 <= profile).all() and (profile <= 100).all()
    node = graph.phenomenodes[0]
    assert repr(node) == node.name
    assert node.size >= 1
    assert node.output_variables()
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
    variables, equations = create_network()
    equations = list(equations.values())
    variables = [i for i in variables.values() if any([i in j.outputs for j in equations])]
    edges = pn.BipartitePhenomeGraph.get_node_edges(equations)
    edges = [i for i in edges if i.variable in variables]
    time = np.array([0., 1.])
    equation_profiles = [np.array([1., 0.]) for i in equations]
    variable_profiles = [np.array([1., 0.]) for i in variables]
    edge_profiles = [np.array([1., 0.]) for i in edges]
    graph = pn.BipartitePhenomeGraph(
        name, equations, variables, edges, equation_profiles,
        variable_profiles, edge_profiles, time, file, **kwargs,
    )
    return graph

@requires_graphviz
def test_bipartite_phenome_graph():
    graph = create_bipartite_graph(maxiter='10')
    assert repr(graph) == 'graph'
    assert pn.PhenomenaGraph is pn.BipartitePhenomeGraph
    # Subgraph inherits positions from parent
    equations = [i for i in graph.equations if 's1' in i.name]
    variables = [i for i in graph.variables if any([i in j.outputs for j in equations])]
    edges = [i for i in pn.BipartitePhenomeGraph.get_node_edges(equations) if i.variable in variables]
    graph.subgraph('s1', equations, variables, edges, None, None, None)
    assert len(graph.subgraphs) == 1
    assert graph.subgraphs[0].pydot is not None
    # Clustering by unit
    digraph = graph.create_digraph(subgraph_units=['s1', 's2'])
    assert digraph is not None
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
    graph = create_bipartite_graph(maxiter='10')
    reference = {j.name: i for i, j in enumerate(graph.equations)}
    original_folder = pn.folder
    with temporary_directory_without_zeros() as folder:
        pn.folder = folder
        try:
            profiles = [np.array([1., 0.]) for i in graph.equations]
            file = os.path.join(folder, 'convergence')
            graph.convergence_gif(reference, graph.time, profiles, file, total_duration=1)
            assert os.path.exists(file + '.gif')
            profiles = [np.array([0.5]) for i in graph.equations]
            graph.convergence_gif(reference, graph.time, profiles, fps=2)
            assert os.path.exists(os.path.join(folder, 'temp.gif'))
            graph.convergence_gif(reference, graph.time, profiles, file + '.gif')
            with pytest.raises(NotImplementedError):
                graph.convergence_gif(reference, graph.time, profiles, interpolate=True)
        finally:
            pn.folder = original_folder

if __name__ == '__main__':
    test_nodes()
    if graphviz_available: test_phenome_graph()
    if graphviz_available: test_bipartite_phenome_graph()
    if graphviz_available: test_bipartite_convergence_gif()
