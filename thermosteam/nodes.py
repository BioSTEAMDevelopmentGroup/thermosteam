# -*- coding: utf-8 -*-
"""
"""

__all__ = (
    'EquationNode',
    'VariableNode',   
    'PhenomenaGraph'
)

def filter_nodes(nodes, cls):
    filtered_nodes = []
    for i in nodes:
        if i is None: continue
        if not isinstance(i, cls):
            raise ValueError('equation nodes can only be connected to variable nodes')
        filtered_nodes.append(i)
    return tuple(filtered_nodes)

class VariableEquationConnection:
    __slots__ = ('variable_node', 'equation_node')
    
    def __init__(self, variable_node, equation_node):
        self.variable_node = variable_node
        self.equation_node = equation_node
        
    def __repr__(self):
        return f'{self.variable_node}--{self.equation_node}'


class EquationNode:
    __slots__ = ('name', 'inputs', 'outputs', 'variables')
    
    def __init__(self, name):
        self.name = name    
    
    def set_equations(self, inputs, outputs):
        self.inputs = filter_nodes(inputs, VariableNode)
        self.outputs = filter_nodes(outputs, VariableNode)
        self.variables = (*self.inputs, *self.outputs)
    
    def get_connections(self, inputs=True, outputs=True):
        if inputs:
            nodes = self.inputs
            if outputs:
                nodes += self.outputs
        elif outputs:
            nodes = self.outputs
        else:
            nodes = ()
        return [VariableEquationConnection(self, i) for i in nodes]
        
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


class PhenomenaGraph:
    __slots__ = ('name', 'equations', 'variables', 'connections', 
                 'variable_profiles', 'equation_profiles', 'subgraphs')
    
    def __init__(self, name, equations, variables, connections, equation_profiles, variable_profiles, subgraphs=()):
        self.name = name
        self.equations = equations
        self.variables = variables
        self.connections = connections
        self.equation_profiles = equation_profiles
        self.variable_profiles = variable_profiles
        self.subgraphs = subgraphs
        
    def __repr__(self):
        return self.name