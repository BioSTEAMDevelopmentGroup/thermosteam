# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import os
from warnings import warn 
import thermosteam as tmo

file_path = os.path.dirname(__file__)

__all__ = ('UnitGraphics',
           'box_graphics',
           'mixer_graphics',
           'splitter_graphics',
           'vertical_column_graphics',
           'vertical_vessel_graphics',
           'utility_heat_exchanger_graphics',
           'process_heat_exchanger_graphics',
           'process_specification_graphics',
           'system_unit',
           'stream_unit',
           'junction_graphics',
           'scaler_graphics',
           'compressor_graphics',
           'turbine_graphics',
           'valve_graphics')

# %% Base class for unit graphics

class UnitGraphics:
    """Create a UnitGraphics object that contains specifications for 
    Graphviz node and edge styles."""
    __slots__ = ('node', 'edge_in', 'edge_out', 'tailor_node_to_unit')
    
    def __init__(self, edge_in, edge_out, node, tailor_node_to_unit=None):
        # [dict] Input stream edge settings
        self.edge_in = edge_in
        
        # [dict] Output stream edge settings
        self.edge_out = edge_out
        
        #: [dict] Node settings
        self.node = node
        
        # [function(node, unit)] tailor node to unit.
        self.tailor_node_to_unit = tailor_node_to_unit
    
    def get_inlet_options(self, sink, sink_index):
        edge_in = self.edge_in
        try:
            options = edge_in[sink_index]
        except IndexError:
            if sink._ins_size_is_fixed: 
                N_inlets = len(edge_in)
                warn(f'inlet #{sink_index} at {repr(sink)} missing graphics options; '
                     f'expected at most {N_inlets} inlet' + ('' if N_inlets == 1 else 's'),
                      RuntimeWarning)
            options = {'headport': 'c'}
        return options
    
    def get_outlet_options(self, source, source_index):
        edge_out = self.edge_out
        try:
            options = edge_out[source_index]
        except IndexError:
            if source._outs_size_is_fixed: 
                N_outlets = len(edge_out)
                warn(f'outlet #{source_index} at {repr(source)} missing graphics options; '
                     f'expected at most {N_outlets} outlet' + ('' if N_outlets == 1 else 's'),
                      RuntimeWarning)
            options = {'tailport': 'c'}
        return options
    
    @classmethod
    def box(cls, N_ins, N_outs):
        edge_in = [{'headport': 'c'} for i in range(N_ins)]
        edge_out = [{'tailport': 'c'} for i in range(N_outs)]
        return cls(edge_in, edge_out, box_node)
    
    def get_minimal_node(self, unit):
        """Return minmal node (a single dot)."""
        minode = dict(
            name = str(hash(unit.ID)),
            label = unit.ID,
            width = '0.1',
            shape = 'oval',
            style = 'filled',
            fillcolor = tmo.preferences.unit_color,
            fontcolor = tmo.preferences.unit_label_color,
        )
        return minode
    
    def get_node_tailored_to_unit(self, unit): # pragma: no coverage
        """Return node tailored to unit specifications"""
        node = self.node.copy()
        if getattr(unit, '_owner', None):
            owner = unit._owner
            ID = unit.ID
            sID, *_ = ID.split('[')
            if sID in owner.parallel:
                N = owner.parallel[sID]
            elif 'self' in owner.parallel:
                N = owner.parallel['self']
            else:
                N = None
            label = '\n'.join([tmo.utils.format_title(i) for i in ID.split('.')])
            label = f"{owner.ID}\n{label}"
            if N is not None and N > 1: label = f"{label}\n1 of {N}"
            node['label'] = label
        else:
            node['label'] = '\n'.join([unit.ID, unit.line]) if unit.line else unit.ID
        tailor_node_to_unit = self.tailor_node_to_unit
        if 'fillcolor' not in node:
            node['fillcolor'] = tmo.preferences.unit_color
        if 'fontcolor' not in node:
            node['fontcolor'] = tmo.preferences.unit_label_color
        if 'color' not in node:
            node['color'] = tmo.preferences.unit_periphery_color
        if tailor_node_to_unit:
            tailor_node_to_unit(node, unit)
        node['name'] = str(hash(unit))
        return node
        
    def __repr__(self): # pragma: no coverage
        return f'{type(self).__name__}(node={self.node}, edge_in={self.edge_in}, edge_out={self.edge_out})'


# %% UnitGraphics components

single_edge_in = ({'headport': 'c'},)
single_edge_out = ({'tailport': 'c'},)
multi_edge_in = 20 * single_edge_in
multi_edge_out = 20 * single_edge_out
right_edge_out = ({'tailport': 'e'},)
left_edge_in = ({'headport': 'w'},)
top_bottom_edge_out = ({'tailport': 'n'}, {'tailport': 's'}, {'tailport': 'c'})

box_node = {'shape': 'box',
            'style': 'filled',
            'gradientangle': '0',
            'width': '0.6',
            'height': '0.6',
            'orientation': '0.0',
            'peripheries': '1',
            'margin': 'default',
            'fontname': 'Arial'}

box_graphics = UnitGraphics(single_edge_in, single_edge_out, box_node)


# %% All graphics objects used in BioSTEAM

# Create mixer graphics
node = box_node.copy()
node['shape'] = 'triangle'
node['orientation'] = '270'
mixer_graphics = UnitGraphics(multi_edge_in, right_edge_out, node)

# Create splitter graphics
node = box_node.copy()
node['shape'] = 'triangle'
node['orientation'] = '90'
# node['fillcolor'] = "#bfbfbf:white"
splitter_graphics = UnitGraphics(left_edge_in, 6 * single_edge_out, node)

# Create distillation column graphics
node = box_node.copy()
node['width'] = '1'
node['height'] = '1.2'
vertical_column_graphics = UnitGraphics(single_edge_in, top_bottom_edge_out, node)

# Create flash column graphics
node = box_node.copy()
node['height'] = '1.1'
vertical_vessel_graphics = UnitGraphics(single_edge_in, top_bottom_edge_out, node)

# Mixer-Settler graphics
node = box_node.copy()
node['width'] = '1.2'
mixer_settler_graphics = UnitGraphics(multi_edge_in, top_bottom_edge_out, node)

# Single stream heat exchanger node
node = box_node.copy()
node['shape'] = 'circle'
node['margin'] = '0'
def tailor_utility_heat_exchanger_node(node, unit): # pragma: no coverage
    try:
        si = unit.ins[0]
        so = unit.outs[0]
        H_in = si.H
        H_out = so.H
        if H_in > H_out + 1e-6:
            node['color'] = 'none'
            node['fillcolor'] = '#60c1cf'
            node['fontcolor'] = 'white'
            node['gradientangle'] = '0'
            line = 'Cooling'
        elif H_in < H_out - 1e-6:
            node['color'] = 'none'
            node['gradientangle'] = '0'
            node['fillcolor'] = '#ed5a6a'
            node['fontcolor'] = 'white'
            line = 'Heating'
        else:
            line = 'Heat exchanger'
    except:
        line = 'Heat exchanger'
    if unit.owner is unit:
        node['label'] = '\n'.join([unit.ID, line])

utility_heat_exchanger_graphics = UnitGraphics(single_edge_in, single_edge_out, node,
                                               tailor_utility_heat_exchanger_node)

# Process heat exchanger network
node = node.copy()
node['shape'] = 'circle'
node['margin'] = '0'
node['gradientangle'] = '90'
node['fillcolor'] = '#60c1cf:#ed5a6a'
def tailor_process_heat_exchanger_node(node, unit): # pragma: no coverage
    node['fontcolor'] = 'white'
    node['color'] = 'none'

process_heat_exchanger_graphics = UnitGraphics(2 * single_edge_in, 2 *single_edge_out, node,
                                               tailor_process_heat_exchanger_node)

# Process specification graphics
node = box_node.copy()
node['fillcolor'] = "#f98f60"
node['color'] = '#de7e55'
node['fontcolor'] = 'white'
node['shape'] = 'note'
node['margin'] = '0.2'
def tailor_process_specification_node(node, unit): # pragma: no coverage    
    node['label'] = (f"{unit.ID} - {unit.description}\n"
                     f"{unit.line}")

process_specification_graphics = UnitGraphics(single_edge_in, single_edge_out, node,
                                              tailor_process_specification_node)


# System unit for creating diagrams
node = box_node.copy()
node['peripheries'] = '1'    
system_unit = UnitGraphics(multi_edge_in, multi_edge_out, node)

node = box_node.copy()
node['fillcolor'] = 'none'
node['color'] = 'none'
def tailor_stream_node(node, unit):
    node['fontcolor'] = tmo.preferences.label_color
    
stream_unit = UnitGraphics(multi_edge_in, multi_edge_out, node,
                           tailor_stream_node)

node = box_node.copy()
def tailor_junction_node(node, unit): # pragma: no coverage
    if not any(unit._ins + unit._outs):
        node['fontsize'] = '18'
        node['shape'] = 'plaintext'
        node['fillcolor'] = 'none'
    else:
        node['width'] = '0.1'
        node['shape'] = 'point'
        node['fillcolor'] = tmo.preferences.stream_color
    node['color'] = 'none'

junction_graphics = UnitGraphics(single_edge_in, single_edge_out, node,
                                 tailor_junction_node)


node = box_node.copy()
def tailor_scalar_node(node, unit): # pragma: no coverage
    node['label'] = f"x{unit.scale}"
    node['width'] = '0.1'
    node['shape'] = 'oval'
    node['style'] = 'filled'
    node['fillcolor'] = tmo.preferences.unit_color
    node['fontcolor'] = tmo.preferences.unit_label_color
    
scaler_graphics = UnitGraphics(single_edge_in, single_edge_out, node,
                               tailor_scalar_node)

# Compressor graphics
node = box_node.copy()
node['shape'] = 'trapezium'
node['orientation'] = '270'
node['height'] = '1.5'
node['margin'] = '0'
compressor_graphics = UnitGraphics(single_edge_in, single_edge_out, node)

# Turbine graphics
node = box_node.copy()
node['shape'] = 'trapezium'
node['orientation'] = '90'
node['height'] = '1.5'
node['margin'] = '0'
turbine_graphics = UnitGraphics(single_edge_in, single_edge_out, node)

# Valve graphics
node = box_node.copy()
node['peripheries'] = '0'
def tailor_valve_node(node, unit): # pragma: no coverage
    node['label'] = ''
    if tmo.preferences.graphviz_format == 'svg':
        node['fillcolor'] = tmo.preferences.unit_color
        node['fontcolor'] = tmo.preferences.unit_label_color
        node['color'] = tmo.preferences.unit_periphery_color
    else:
        if tmo.preferences.unit_color == "#555f69":
            filename = "graphics/valve_dark.png"
        elif tmo.preferences.unit_color == "white:#CDCDCD":
            filename = "graphics/valve_light.png"
        else:
            filename = "graphics/valve_dark.png"
        node['fillcolor'] = node['color'] = 'none'
        node['image'] = os.path.join(file_path, filename)
        node['xlabel'] = unit.ID + "\nValve"
        node['width'] = '0.7738'
        node['height'] = '0.5'
        node['margin'] = '0'
        node['fixedsize'] = 'true'
        node['fontcolor'] = tmo.preferences.label_color
valve_graphics = UnitGraphics(single_edge_in, single_edge_out, node, tailor_valve_node)