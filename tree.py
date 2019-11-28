# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 10:58:19 2019

@author: yoelr
"""
import thermodynamics as tm 
from graphviz import Digraph
from IPython import display

water = tm.Chemical('Water')
ethanol = tm.Chemical('Ethanol')
methanol = tm.Chemical('Methanol')
ideal_mixture = tm.IdealMixture(chemicals=(water, ethanol, methanol))
digraph = Digraph(format='svg')

istreelike = lambda x: hasattr(x, '__slots__') or hasattr(x, '__dict__')
def allslots(x):
    try: return sum([i.__slots__ for i in type(x).mro() if hasattr(i, '__slots__')], ())
    except: return ()
#allslots = lambda x: sum([i.__slots__ for i in type(x).mro() if hasattr(i, '__slots__')], ())
modname = lambda child_name, parent_name: parent_name + "." + child_name if child_name in ('s', 'l', 'g') else child_name

past_names = set()

def branch_out(digraph, parent_name, child_name, child, stop=False):
    if istreelike(child):
        child_name = modname(child_name, parent_name)
        if not child_name or child_name in past_names: return
        past_names.add(child_name)
        digraph.node(child_name)
        digraph.edge(parent_name, child_name)
        tree(digraph, child_name, child)
    elif isinstance(child, list):
        for i in child:
            if isinstance(i, tm.ThermoModel):
                branch_out(digraph, parent_name, i.name, i)

def tree(digraph, parent_name, parent):
    if hasattr(parent, '__name__') or hasattr(parent, 'indomain'): return
    if hasattr(parent, '__slots__'):
        for child_name in allslots(parent):
            child = getattr(parent, child_name)
            branch_out(digraph, parent_name, child_name, child)
    elif hasattr(parent, '__dict__'):
        for child_name, child in parent.__dict__.items():
            branch_out(digraph, parent_name, child_name, child)
    
        

tree(digraph, 'Ideal Mixture', ideal_mixture)
display.SVG(digraph.pipe(format='svg'))
        
