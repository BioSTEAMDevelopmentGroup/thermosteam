# -*- coding: utf-8 -*-
"""
"""

__all__ = (
    'EquationNode',
    'VariableNode',    
)

class VariableEquationConnection:
    __slots__ = ('variable_node', 'equation_node')
    
    def __init__(self, variable_node, equation_node):
        self.variable_node = variable_node
        self.equation_node = equation_node
        
    def __repr__(self):
        return f'{self.variable_node}--{self.equation_node}'


class EquationNode:
    __slots__ = ('name', 'nodes',)
    
    def __init__(self, name):
        self.name = name
        
    def set_equations(self, *nodes):
        filtered_nodes = []
        for i in nodes:
            if i is None: continue
            if not isinstance(i, VariableNode):
                raise ValueError('equation nodes can only be connected to variable nodes')
            filtered_nodes.append(i)
        self.nodes = tuple(filtered_nodes)
        
    def get_connections(self):
        return [VariableEquationConnection(self, i) for i in self.nodes]
        
    def __repr__(self):
        return self.name
    
    
class VariableNode:
    __slots__ = ('name', 'value', 'nodes',)
    
    def __init__(self, name, value, *nodes):
        filtered_nodes = []
        for i in nodes:
            if i is None: continue
            if not isinstance(i, EquationNode):
                raise ValueError('variable nodes can only be connected to equation nodes')
            filtered_nodes.append(i)
        self.name = name
        self.value = value
        self.nodes = tuple(filtered_nodes)
        
    def get_connections(self):
        return [VariableEquationConnection(self, i) for i in self.nodes]
        
    def __repr__(self):
        return self.name
