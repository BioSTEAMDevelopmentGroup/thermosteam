# -*- coding: utf-8 -*-
"""
"""
from typing import NamedTuple

__all__ = (
    'EquationNode',
    'VariableNode',   
    'PhenomenaGraph',
    'Edge',
)

def filter_nodes(nodes, cls):
    filtered_nodes = []
    for i in nodes:
        if i is None: continue
        if not isinstance(i, cls):
            raise ValueError('equation nodes can only be connected to variable nodes')
        filtered_nodes.append(i)
    return tuple(filtered_nodes)

class EquationNode:
    __slots__ = ('name', 'inputs', 'outputs', 'variables')
    
    def __init__(self, name):
        self.name = name    
    
    def set_equations(self, inputs, outputs):
        self.inputs = filter_nodes(inputs, VariableNode)
        self.outputs = filter_nodes(outputs, VariableNode)
        self.variables = (*self.inputs, *self.outputs)
    
    def get_edges(self, inputs=True, outputs=True):
        if inputs:
            nodes = self.inputs
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
    __slots__ = ('name', 'value', 'equations')
    
    def __init__(self, name, value, *equations):
        self.name = name
        self.value = value
        self.equations = filter_nodes(equations, EquationNode)
        
    def __repr__(self):
        return self.name


class Edge(NamedTuple):
    variable_node: VariableNode
    equation_node: EquationNode
    
    @property
    def name(self):
        return (self.variable_node.name, self.equation_node.name)


class PhenomenaGraph:
    __slots__ = ('name', 'equations', 'variables', 'edges', 
                 'variable_profiles', 'equation_profiles', 'edge_profiles',
                 'subgraphs')
    
    def __init__(self, name, equations, variables, edges, equation_profiles, variable_profiles, edge_profiles, subgraphs=()):
        self.name = name
        self.equations = equations
        self.variables = variables
        self.edges = edges
        self.equation_profiles = equation_profiles
        self.variable_profiles = variable_profiles
        self.edge_profiles = edge_profiles
        self.subgraphs = subgraphs
        
    def __repr__(self):
        return self.name