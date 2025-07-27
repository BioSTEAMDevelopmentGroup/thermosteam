# -*- coding: utf-8 -*-
"""
"""
from typing import NamedTuple
from warnings import warn
from graphviz import Digraph, Source
from IPython import display as _display
from collections import deque
from colorpalette import Color
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from itertools import product
from PIL import Image
import contextlib
import pydot
import numpy as np
import imageio
import os
from matplotlib import colormaps
from matplotlib.colors import rgb2hex
import shutil

viridis = colormaps['viridis']

__all__ = (
    'PhenomeNode',
    'EquationNode',
    'VariableNode',   
    'PhenomeGraph',
    'BipartitePhenomeGraph',
    'PhenomenaGraph',
    'DotFile',
    'Edge',
    'all_subgraphs',
)

#: [dict] Default variable node settings
variable_node_options = {
    'gradientangle': '0',
    'width': '0.15',
    'height': '0.15',
    'orientation': '0.0',
    'peripheries': '1',
    'penwidth': '3',
    'margin': '0',
    'fillcolor': 'none',
    'shape': 'circle',
    'style': 'filled',
    # 'color': '#b4b1ae',
    'label': '',
    'pin': 'true',
}

#: [dict] Default equation node settings
equation_node_options = {
    'shape': 'circle',
    'style': 'filled',
    'gradientangle': '0',
    'width': '0.2',
    'height': '0.2',
    'orientation': '0.0',
    'peripheries': '1',
    'margin': '0',
    'label': '',
    'pin': 'true',
}

#: [dict] Default phenomenode settings
phenomenode_options = {
    'shape': 'box',
    'style': 'filled',
    'gradientangle': '0',
    'width': '0.2',
    'orientation': '0.0',
    'peripheries': '0',
    'margin': '0',
    'label': '',
    'pin': 'true',
}

#: [dict] Default phenomenode edge settings
phenomenode_edge_options = dict(
    label='', 
    arrowtail='none', 
    arrowhead='none', 
    minlen='1',
    dir='none',
    penwidth='3',
    style='solid',
    pin='true',
)

#: [dict] Default equation node colors by category
colors = {
    'material': '#a280b9', # purple
    'energy': '#ed586f', # red
    'vle': '#60c1cf', # blue
    'lle': '#79bf82', # green
    'reaction': '#dd7440', # orange
    'shortcut': '#00a996', # darker blue 
    'compression': '#5c5763', # black
}
edge_options = dict(
    label='', 
    arrowtail='none', 
    arrowhead='none', 
    headport='c', 
    tailport='c', 
    len='0.5',
    # minlen='0.5',
    dir='none',
    penwidth='3',
    style='solid',
    # pin='true',
)
folder = os.path.dirname(__file__)
folder = os.path.join(folder, 'temporary_images')
             
all_subgraphs = ('material_balance', 'energy_balance', 'phenomena')

def filter_nodes(nodes, cls):
    filtered_nodes = []
    past_nodes = set()
    for i in nodes:
        if i is None: continue
        if i in past_nodes: continue    
        if not isinstance(i, cls):
            raise ValueError('equation nodes can only be connected to variable nodes')
        filtered_nodes.append(i)
        past_nodes.add(i)
    return filtered_nodes

def group_equations(equations, criteria, all_equations, all_neighbors):
    for eq in equations:
        if eq in all_equations or eq in all_neighbors: continue
        if criteria.match(eq.name): 
            all_equations.add(eq)
            group_equations(eq.neighbors, criteria, all_equations, all_neighbors)
        else:
            all_neighbors.add(eq)

def get_order(values):
    N = len(values)
    order = [*range(N)]
    order = sorted(order, key=lambda x: values[x])
    past_values = {}
    for n, i in enumerate(order):
        value = values[i]
        if value in past_values:
            order[n] = past_values[value]
        else:
            past_values[value] = i
    return order

class EquationNode:
    __slots__ = ('name', 'inputs', 'outputs', 'tracked_outputs')
    
    def __init__(self, name):
        self.name = name    
    
    @property
    def category(self):
        name = self.name
        for category in colors:
            if category in name: return category
        raise AttributeError('no category')
    
    def set_equations(self, inputs=(), outputs=(), tracked_outputs=None):
        self.inputs = filter_nodes(inputs, VariableNode)
        self.outputs = filter_nodes(outputs, VariableNode)
        if tracked_outputs is None: 
            self.tracked_outputs = self.outputs
        else:
            self.tracked_outputs = filter_nodes(tracked_outputs, VariableNode)
        for i in self.variables: i.equations.append(self)
        
    @property
    def neighbors(self):
        return set([j for i in self.variables for j in i.equations])
        
    @property
    def variables(self):
        return (*self.inputs, *self.outputs)
    
    def get_edges(self, inputs=True, outputs=True):
        if inputs:
            nodes = self.inputs.copy()
            if outputs:
                nodes += self.outputs
        elif outputs:
            nodes = self.outputs
        else:
            nodes = ()
        return [Edge(self, i) for i in nodes]
        
    def __repr__(self):
        return self.name
    
    
class VariableNode:
    __slots__ = ('name', 'getter', 'equations', 'last_value')
    
    def __init__(self, name, getter):
        if not callable(getter):
            raise ValueError('getter must be function')
        self.name = name
        self.getter = getter
        self.equations = []
        self.last_value = np.nan
        
    def get_value(self):
        self.last_value = value = self.getter()
        return value
        
    def __repr__(self):
        return self.name


class Edge(NamedTuple):
    equation: EquationNode
    variable: VariableNode
    
    @property
    def name(self):
        return (self.equation.name, self.variable.name)
    

class Criteria:
    __slots__ = ('names',)
    
    def __init__(self, names):
        if isinstance(names, str): names = [names]
        self.names = names
        
    def match(self, full_name):
        return all([i in full_name for i in self.names])


class CriteriaSelection:
    __slots__ = ('criterias',)
    def __init__(self, selection):
        self.criterias = [Criteria(i) for i in selection]

    def get_criteria(self, full_name):
        for criteria in self.criterias:
            if criteria.match(full_name): return criteria

class PhenomeNode:
    __slots__ = ('name', 'size', 'equations', 'neighbors', 'criteria')
    
    def __init__(self, equation, criteria_selection):
        self.name = equation.name # Good enough
        self.equations = equations = set([equation])
        self.neighbors = neighbors = set()
        self.criteria = criteria = criteria_selection.get_criteria(equation.name)
        group_equations(equation.neighbors, criteria, equations, neighbors)
        
    category = EquationNode.category
        
    def finalize(self, stage_tags):
        name = '_'.join([i.name for i in self.equations])
        self.size = sum([i in name for i in stage_tags])
        
    def output_variables(self):
        return set([j for i in self.equations for j in i.tracked_outputs]) 
        
    def __repr__(self):
        return self.name


class PhenomeGraph:
    __slots__ = ('phenomenodes', 'edges', 'pydot', 'profiles', 'time')
    
    def __init__(self, equations, criteria_selection, stage_tags, variable_profiles, time, file=None):
        phenomenodes = []
        collected_equations = set()
        if not isinstance(criteria_selection, CriteriaSelection): criteria_selection = CriteriaSelection(criteria_selection)
        for eq in equations:
            if eq in collected_equations: continue
            phenomenode = PhenomeNode(eq, criteria_selection)
            phenomenodes.append(phenomenode)
            collected_equations.update(phenomenode.equations)
        reference = {
            eq: node for node in phenomenodes
            for eq in node.equations
        }
        edges = set([
            frozenset([node, reference[eq]]) for node in phenomenodes
            for eq in node.neighbors
        ])
        for i in phenomenodes: i.finalize(stage_tags)
        errors = np.abs(variable_profiles - variable_profiles.iloc[-1])
        def normalized_profile(errors):
            order = np.array(get_order(errors.sum(axis=1)))
            order_norm = 100 * order / order.max()
            # breakpoint()
            return order_norm
            # MSE = errors.sum(axis=1)
            # # MSE = (errors * errors).mean(axis=1)
            # mask = np.isnan(MSE)
            # MSE[mask] = MSE[~mask].max() # Maximum error
            # MSE[MSE < 1e-16] = 1e-16 # Minimum error
            # MSE = np.log(MSE)
            # MSE_zeroed = MSE - MSE.min()
            # MSE_norm = 100 * MSE_zeroed / MSE_zeroed.max()
            # return MSE_norm
        
        assert variable_profiles.shape[0] == len(time)
        profiles = [
            normalized_profile(errors[[i.name for i in phenomenode.output_variables()]].values)
            for phenomenode in phenomenodes
        ]
        self.time = time
        assert len(profiles[0]) == len(time)
        self.phenomenodes = phenomenodes
        self.edges = edges
        self.profiles = profiles
        self.load_pydot()
        
    def load_pydot(self, 
            file=None,
            maxiter=None,
            damping=None, 
            K=None,
        ):
        if file:
            try:
                self.pydot = pydot.graph_from_dot_file(file)[0]
            except:
                self.load_pydot(None, maxiter, damping, K)
                self.write(file)
            return
        if maxiter is None: maxiter = '20000'
        if damping is None: damping = '0.3'
        if K is None: K = '0.2'
        # Create a digraph and set direction left to right
        digraph = Digraph(format='png', strict=True)
        digraph.attr(
            'graph',
            rankdir='LR', 
            maxiter=maxiter, 
            Damping=damping, 
            K=K,
            penwidth='0', 
            color='none', 
            bgcolor='transparent',
            nodesep='0.02', 
            ranksep='0.02', 
            layout='dot', 
            splines='curved', 
            outputorder='edgesfirst', 
            dpi='300',
            # nodesep='0.2', 
            # ranksep='0.2',
            overlap='compress', 
            dir='none',
        )
        dct = {}
        for i in self.phenomenodes:
            names = i.criteria.names
            if len(names) == 1:
                key, = names
                if key in dct: dct[key].append(i)
                else: dct[key] = [i]
            else:
                outer, inner = names
                if outer in dct: subdct = dct[outer]
                else: dct[outer] = subdct = {}
                if inner in subdct: subdct[inner].append(i)
                else: subdct[inner] = [i]
            
        for m, subdct in enumerate(dct.values()):
            if isinstance(subdct, dict):
                with digraph.subgraph(name=f'cluster_{m}') as subgraph:
                    subgraph.attr('graph', color='none', bgcolor='none', shape='box', label='')
                    for n, phenomenodes in enumerate(subdct.values()): 
                        with subgraph.subgraph(name=f'cluster_{m}_{n}') as sub2:
                            sub2.attr('graph', color='none', bgcolor='none', shape='box', label='')
                            for i in phenomenodes:
                                color = colors[i.category]
                                sub2.node(
                                    name=i.name, 
                                    color='black',
                                    fillcolor=color,
                                    height=str(i.size * 0.2),
                                    **phenomenode_options
                                )
            else:
                n = m
                phenomenodes = subdct
                with digraph.subgraph(name='cluster_' + str(n)) as subgraph:
                    subgraph.attr('graph', color='none', bgcolor='none', shape='box', label='')
                    for i in phenomenodes:
                        color = colors[i.category]
                        subgraph.node(
                            name=i.name, 
                            color='black',
                            fillcolor=color,
                            height=str(i.size * 0.2),
                            **phenomenode_options
                        )
                                
        # Set attributes for graph and edges
        digraph.attr(
            'edge', 
            concentrate='true',
            dir='none'
        )
        for left, right in self.edges:
            if right.size == 1:
                color = colors[right.category]
            elif left.size == 1:
                color = colors[right.category]
            else:
                color = 'black'
            digraph.attr('edge', label='', taillabel='', headlabel='')
            digraph.edge(
                left.name, right.name, 
                color='black',
                **phenomenode_edge_options,
            )
        dot = digraph.pipe(format='dot') # Pin nodes
        self.pydot = pydot.graph_from_dot_data(dot.decode('utf-8'))[0]
        
    def display(self): # TODO: this does not work, maybe revert back to graphviz  
        source = Source(self.pydot.create_dot().decode('utf-8'))
        # img = source.pipe('svg')
        # x = _display.SVG(img)
        x = _display.Image(source.pipe(format='png'))
        return _display.display(x)
        
    def write(self, file):
        format = file[file.index('.')+1:]
        method = 'write_' + format
        getattr(self.pydot, method)(file)
        
    # TODO: plots/gifs
    # def plot_convergence_profile(self, file=None):
    #     equation_profiles = self.equation_profiles
    #     profiles_arr = np.abs(equation_profiles.values)
    #     equations = sorted(self.equations, key=lambda x: x.category)
    #     index = [equation_profiles.columns.get_loc(i.name) for i in equations]
    #     # ps = [*range(0, 101, 1)]
    #     # percentiles = np.percentile(profiles_arr, ps, axis=0)
    #     # fs = [interp1d(i, ps) for i in percentiles.T]
    #     M, N = profiles_arr.shape
    #     image = np.zeros([M, N, 3])
    #     # errors_percentiles = np.zeros([M, N])
    #     # for i in range(M):
    #     #     for j in index: 
    #     #         errors_percentiles[i, j] = fs[j](profiles_arr[i, j])
    #     # errors_percentiles -= errors_percentiles.min(axis=0, keepdims=True)
    #     profiles_arr[:] = np.log(profiles_arr + 1e-6)
    #     profiles_arr -= profiles_arr.min(axis=0, keepdims=True)
    #     max_errors = profiles_arr.max(axis=0, keepdims=True)
    #     max_errors[max_errors < 1e-9] = 1
    #     profiles_arr /= max_errors
    #     for i in range(M):
    #         for k, (j, eq) in enumerate(zip(index, equations)): 
    #             # error = errors_percentiles[i, j]
    #             # if error < 20: error = 0.
    #             error = 100 * profiles_arr[i, j]
    #             color = colors[eq.category]
    #             image[i, k, :] = Color(fg=color).shade(error).RGBn        
    #     ax = plt.imshow(image)
    #     plt.axis('off')
    #     if file:
    #         for i in ('svg', 'png'):
    #             plt.savefig(file, dpi=900, transparent=True)
    #     return ax
        
    def convergence_gif(
            self, 
            file=None, 
            total_duration=30, # Seconds
            fps=3,
            **kwargs
        ):
        digraph = self.pydot
        digraph.set_maxiter('10') # This speeds up image creation.
        digraph.set_dpi('100') # Pydot does not seem to save dpi from piped graphviz
        # breakpoint()
        output_file = file
        input_file = os.path.join(folder, 'temp0.png')
        profiles = self.profiles
        total_frames = total_duration * fps
        time = self.time
        normalized_time = time / time[-1] * total_duration
        frame_duration = 1 / fps 
        interpolators = [interp1d(normalized_time, profiles[name]) for name in profiles]
        phenomenodes = self.phenomenodes
        nodes_dct = {}
        for subgraph in digraph.get_subgraph_list():
            if self.depth == 1:
                for i in subgraph.get_nodes():
                    nodes_dct[i.get_name().strip('"')] = i
            elif self.depth == 2:
                for subgraph in subgraph.get_subgraph_list():
                    for i in subgraph.get_nodes():
                        nodes_dct[i.get_name().strip('"')] = i
        print('total frames', total_frames)
        t = 0
        for n in range(total_frames):
            print('Frame #', n)
            t += frame_duration
            for j, node in enumerate(phenomenodes): 
                error = 0.8 * interpolators[j](t) # Leave at least 20% of color in
                color = Color(fg=colors[node.category]).shade(error).HEX
                node = nodes_dct[node.name]
                node.set_fillcolor(color)
            digraph.write_png(input_file.replace('0', str(n)))
            
        if output_file is None: output_file = os.path.join(folder, 'temp.gif')
        elif '.' not in output_file: output_file += '.gif'
        
        with contextlib.ExitStack() as stack:
            # lazily load images
            imgs = (
                stack.enter_context(
                    Image.open(
                        input_file.replace('0', str(i))
                    )
                ) for i in range(total_frames)
            )
        
            # extract  first image from iterator
            img = next(imgs)
        
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=output_file, format='GIF', append_images=imgs,
                     save_all=True, duration=frame_duration * 1000, optimize=False, loop=1)
            

class BipartitePhenomeGraph:
    __slots__ = ('name', 'equations', 'variables', 'edges', 
                 'variable_profiles', 'equation_profiles', 'edge_profiles',
                 'subgraphs', 'pydot', 'time',)
    
    @staticmethod
    def get_node_edges(equation_nodes, inputs=True, outputs=True):
        return set(sum([i.get_edges(inputs, outputs) for i in equation_nodes], []))
    
    def __init__(self, 
            name, equations, variables, edges, 
            equation_profiles, variable_profiles, 
            edge_profiles, time, file=None,
            **kwargs,
        ):
        self.name = name
        self.equations = equations
        self.variables = variables
        self.edges = edges
        self.equation_profiles = equation_profiles
        self.variable_profiles = variable_profiles
        self.edge_profiles = edge_profiles
        self.time = time
        self.subgraphs = []
        self.load_pydot(file, **kwargs)
       
    def subgraph(self, 
            name, equations, variables, edges, 
            equation_profiles, variable_profiles, 
            edge_profiles, file=None, **kwargs,
        ):
        subgraph = BipartitePhenomeGraph.__new__(BipartitePhenomeGraph)
        subgraph.name = name
        subgraph.equations = equations
        subgraph.variables = variables
        subgraph.edges = edges
        subgraph.equation_profiles = equation_profiles
        subgraph.variable_profiles = variable_profiles
        subgraph.edge_profiles = edge_profiles
        subgraph.time = self.time
        self.subgraphs.append(subgraph)
        subgraph.load_pydot(file, parent=self.pydot, **kwargs)
       
    def load_pydot(self, 
            file=None,
            maxiter=None,
            damping=None, 
            K=None,
            subgraph_units=None,
            parent=None,
        ):
        if file:
            try:
                self.pydot = pydot.graph_from_dot_file(file)[0]
            except:
                self.load_pydot(None, maxiter, damping, K, subgraph_units, parent)
                self.write(file)
            return
        if maxiter is None: maxiter = '10000'
        if damping is None: damping = '0.2'
        if K is None: K = '0.2'
        # Create a digraph and set direction left to right
        digraph = Digraph(format='png', strict=True)
        digraph.attr(
            'graph',
            rankdir='LR', 
            maxiter=maxiter, 
            Damping=damping, 
            K=K,
            sep='1',
            penwidth='0', 
            color='none', 
            bgcolor='transparent',
            nodesep='0.02', 
            ranksep='0.02', 
            layout='fdp', 
            splines='curved', 
            outputorder='edgesfirst', 
            dpi='300',
            # nodesep='0.2', 
            # ranksep='0.2',
            overlap='compress', 
            dir='none',
        )
        variables = self.variables
        equations = self.equations
        edges = self.edges
        inner_edges = []
        past_nodes = set()
        subgraph_units = False
        if subgraph_units:
            for name in subgraph_units:
                # with digraph.subgraph(name=name) as subgraph:
                # subgraph.attr('graph', color='#b4b1ae', bgcolor='#b4b1ae', shape='box', layout='fdp')
                subgraph_equations = [i for i in equations if name in i.name]
                subgraph_variables = set(sum([i.outputs for i in subgraph_equations], []))
                subgraph_variables = set([i for i in subgraph_variables if name in i.name])
                assert subgraph_variables.isdisjoint(past_nodes)
                past_nodes.update(subgraph_variables)
                subgraph_edges = BipartitePhenomeGraph.get_node_edges(subgraph_equations)
                subgraph_edges = tuple([i for i in subgraph_edges if i.variable in subgraph_variables])
                subgraph_variables = tuple(subgraph_variables)
                inner_edges.extend(subgraph_edges)
                self.fill_digraph_nodes(digraph, subgraph_variables, subgraph_equations)
                self.fill_digraph_edges(digraph, subgraph_edges)
            self.fill_digraph_edges(digraph, set(edges).difference(inner_edges))
        else:
            self.fill_digraph_nodes(digraph, variables, equations)
            self.fill_digraph_edges(digraph, edges)
        # img = digraph.pipe(format='png') # Pin nodes
        # f = open(self.name + '.png', 'wb')
        # f.write(img)
        # f.close()
        dot = digraph.pipe(format='dot') # Pin nodes
        self.pydot = pydot.graph_from_dot_data(dot.decode('utf-8'))[0]
        # for node in self.pydot.get_nodes(): node.set_pin('false')
        if parent:
            nodes_dct = {i.get_name().strip('"'): i for i in self.pydot.get_nodes()}
            parent_nodes_dct = {i.get_name().strip('"'): i for i in parent.get_nodes()}
            edges_dct = {
                (i.get_source().replace(':c', '').strip('"'), i.get_destination().replace(':c', '').strip('"')): i
                for i in self.pydot.get_edges()
            }
            parent_edges_dct = {
                (i.get_source().replace(':c', '').strip('"'), i.get_destination().replace(':c', '').strip('"')): i
                for i in parent.get_edges()
            }
            for name, node in nodes_dct.items():
                pos = parent_nodes_dct[name].get_pos()
                if pos: node.set_pos(pos)
            for key, edge in edges_dct.items():
                pos = parent_edges_dct[key].get_pos()
                if pos: edge.set_pos(pos)
        
    def fill_digraph_nodes(self, digraph, variables, equations):
        variable_colors = {}
        for i in equations: 
            color = colors[i.category]
            digraph.node(
                name=i.name, 
                color=color,
                fillcolor=color,
                **equation_node_options
            )
            for var in i.outputs: variable_colors[var.name] = color
        for i in variables: 
            digraph.node(
                name=i.name, 
                color=variable_colors[i.name],
                **variable_node_options
            )
    
    def fill_digraph_edges(self, digraph, edges):
        # Set attributes for graph and edges
        digraph.attr(
            'edge', 
            dir='none'
        )
        equations = set(self.equations)
        variables = set(self.variables)
        for edge in edges:
            equation = edge.equation
            variable = edge.variable
            assert equation in equations
            assert variable in variables
            digraph.attr('edge', label='', taillabel='', headlabel='')
            color = colors[equation.category]
            digraph.edge(
                equation.name, variable.name, 
                color=color, name='_'.join(edge.name),
                **edge_options,
            )
        
    def display(self): # TODO: this does not work, maybe revert back to graphviz  
        source = Source(self.pydot.create_dot().decode('utf-8'))
        # img = source.pipe('svg')
        # x = _display.SVG(img)
        x = _display.Image(source.pipe(format='png'))
        return _display.display(x)
        
    def write(self, file):
        format = file[file.index('.')+1:]
        method = 'write_' + format
        getattr(self.pydot, method)(file)
        
    # def plot_convergence_profile(self, file=None):
    #     equation_profiles = self.equation_profiles
    #     profiles_arr = np.abs(equation_profiles.values)
    #     equations = sorted(self.equations, key=lambda x: x.category)
    #     index = [equation_profiles.columns.get_loc(i.name) for i in equations]
    #     # ps = [*range(0, 101, 1)]
    #     # percentiles = np.percentile(profiles_arr, ps, axis=0)
    #     # fs = [interp1d(i, ps) for i in percentiles.T]
    #     M, N = profiles_arr.shape
    #     image = np.zeros([M, N, 3])
    #     # errors_percentiles = np.zeros([M, N])
    #     # for i in range(M):
    #     #     for j in index: 
    #     #         errors_percentiles[i, j] = fs[j](profiles_arr[i, j])
    #     # errors_percentiles -= errors_percentiles.min(axis=0, keepdims=True)
    #     profiles_arr[:] = np.log(profiles_arr + 1e-6)
    #     profiles_arr -= profiles_arr.min(axis=0, keepdims=True)
    #     max_errors = profiles_arr.max(axis=0, keepdims=True)
    #     max_errors[max_errors < 1e-9] = 1
    #     profiles_arr /= max_errors
    #     for i in range(M):
    #         for k, (j, eq) in enumerate(zip(index, equations)): 
    #             # error = errors_percentiles[i, j]
    #             # if error < 20: error = 0.
    #             error = 100 * profiles_arr[i, j]
    #             color = colors[eq.category]
    #             image[i, k, :] = Color(fg=color).shade(error).RGBn        
    #     ax = plt.imshow(image)
    #     plt.axis('off')
    #     if file:
    #         for i in ('svg', 'png'):
    #             plt.savefig(file, dpi=900, transparent=True)
    #     return ax
        
    def convergence_gif(self, 
            reference, time, profiles, 
            file=None, total_duration=None, fps=None, 
            categorize=True, interpolate=False,
            inverse=False, **kwargs
        ):
        digraph = self.pydot
        equations = self.equations
        digraph.set_maxiter('10') # This speeds up image creation.
        digraph.set_dpi('100') # Pydot does not seem to save dpi from piped graphviz
        # breakpoint()
        output_file = file
        input_file = os.path.join(folder, 'temp0.png')
        time = self.time
        # profiles = self.variable_profiles
        # reference = {i: n for n, i in enumerate(profiles)}
        # profiles = profiles.values
        # profiles = np.abs(profiles - profiles[-1])
        # profiles -= profiles.min(axis=0, keepdims=True)
        # profiles *= 100 / profiles.max()
        pulses = (False, True, False)
        if interpolate:
            raise NotImplementedError('not implemented in BioSTEAM yet')
            if fps is None: fps = 3
            total_frames = total_duration * fps
            time -= time[0]
            normalized_time = time / time[-1] * total_duration
            frame_duration = 1 / fps 
            interpolators = [interp1d(normalized_time, i) for i in profiles]
            duration_ms = 1000 * frame_duration
            # print(duration_ms)
        else:
            # time = time[:total_frames + 1] - time[0]
            # normalized_time = time / time[-1] * total_duration
            # duration_ms = 1000 * np.diff(normalized_time) # in ms
            # duration_ms = [*duration_ms]
            # duration_ms = 1 / fps * 1000
            # print(duration_ms)
            # breakpoint()
            total_frames = len(profiles[0])
            if total_duration is not None:
                duration_ms = 1000 * total_duration / total_frames
                fps = duration_ms / 1000
            elif fps is not None:
                total_duration = total_frames / fps
            else:
                total_duration = 16
                fps = total_duration / total_frames
            duration_ms = 1000 / (fps * len(pulses))
            print('total_duration', total_duration)
            # profiles = [i[:total_frames] for i in profiles]
            # profiles = [i - i.min() for i in profiles]
            # profiles = [100 * i / (i.max() or 1) for i in profiles]
        total_frames = len(profiles[0])
        print('total frames', total_frames)
        variables = set(self.variables)
        nodes_dct = {i.get_name().strip('"'): i for i in digraph.get_nodes()}
        equation_names = set([i.name for i in equations])
        variable_names = set([i.name for i in variables])
        edges_dct = {
            (i.get_source().replace(':c', '').strip('"'), i.get_destination().replace(':c', '').strip('"')): i
            for i in digraph.get_edges()
        }
        if interpolate: 
            t = 0
            dt = total_duration / (total_frames - 1)
        filenames = []
        done = set()
        for n in range(total_frames):
            print('Frame #', n)
            if interpolate: 
                t += dt
                if t > total_duration: t = total_duration # floating point error
            nfile = input_file.replace('0', str(n))
            for pulse in pulses: 
                pulse_nfile = nfile.replace('.png', '_pulse.png')
                if (n, pulse) in done:
                    if pulse: 
                        filenames.append(pulse_nfile)
                    else:
                        filenames.append(nfile)
                    continue
                for eq in equations:
                    index = reference[eq.name]
                    if interpolate: 
                        baseline = interpolators[index](t) 
                        if pulse and n:
                            last = interpolators[index](t - dt) 
                            error = max(1.5 * baseline - 0.5 * last, 0) # Leave at least 20% of color in
                        else:
                            error = baseline
                    else:
                        baseline = profiles[index][n]
                        if pulse and n:
                            last = profiles[index][n - 1]
                            error = max(1.5 * baseline - 0.5 * last, 0) # Leave at least 20% of color in
                        else:
                            error = baseline
                    error = 0.8 * error # Leave at least 20% of color in
                    color = colors[eq.category]
                    color = Color(fg=color).shade(error).HEX
                    node = nodes_dct[eq.name]
                    node.set_color(color)
                    node.set_fillcolor(color)
                    edges = [i for i in eq.get_edges() if i.variable in variables]
                    for edge in edges:
                        _edge = edges_dct[edge.equation.name, edge.variable.name]
                        _edge.set_color(color)
                    for var in eq.outputs:
                        var = nodes_dct[var.name]
                        var.set_color(color)
                if pulse: 
                    digraph.write_png(pulse_nfile)
                    filenames.append(pulse_nfile)
                else:
                    digraph.write_png(nfile)
                    filenames.append(nfile)
                done.add((n, pulse))
                
        if output_file is None: output_file = os.path.join(folder, 'temp.gif')
        elif '.' not in output_file: output_file += '.gif'
        with contextlib.ExitStack() as stack:
            # lazily load images
            imgs = (
                stack.enter_context(
                    Image.open(i)
                ) for i in filenames
            )
        
            # extract  first image from iterator
            img = next(imgs)
        
            duration = [duration_ms] * len(filenames)
            duration[-1] += 5000
            print(sum(duration) / 1000)
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=output_file, format='GIF', append_images=imgs, loop=0,
                     save_all=True, duration=duration, optimize=False)
            
    def __repr__(self):
        return self.name
    
PhenomenaGraph = BipartitePhenomeGraph # For backwards compatibility

class DotFile:
    __slots__ = ('file', 'subfiles')
    
    def __init__(self, file):
        self.file = file if file.endswith('.dot') else f'{file}.dot'
         
    def subfile(self, subgraph):
        return self.file.replace('.dot', f'_{subgraph}.dot')
        
    def __repr__(self):
        return f"{type(self).__name__}(file={self.file!r})"