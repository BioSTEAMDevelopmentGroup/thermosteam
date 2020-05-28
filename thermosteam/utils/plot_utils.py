# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import matplotlib.pyplot as plt

__all__ = ('fix_axis_style',

)

def fix_axis_style(ax=None, xticks=None, yticks=None):
    if ax is None:
        ax = plt.gca()
    if xticks is None:
        xticks, _ = plt.xticks()
    if yticks is None:
        yticks, _ = plt.yticks()
    plt.xticks(xticks)
    plt.yticks(yticks)
    xlim = plt.xlim()
    ylim = plt.ylim()
    ax.tick_params(axis="x", top=False, direction="inout", length=4)
    ax.tick_params(axis="y", right=False, direction="inout", length=4)
    x_twin = ax.twinx()
    plt.sca(x_twin)
    x_twin.tick_params(axis='y', right=True, direction="in", length=4)
    plt.ylim(ylim)
    plt.yticks(yticks, ())
    y_twin = ax.twiny()
    plt.sca(y_twin)
    y_twin.tick_params(axis='x', top=True, direction="in", length=4)
    plt.xlim(xlim)
    plt.xticks(xticks, ())