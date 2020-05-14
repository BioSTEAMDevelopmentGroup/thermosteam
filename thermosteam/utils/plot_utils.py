# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:12:48 2020

@author: yoelr
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