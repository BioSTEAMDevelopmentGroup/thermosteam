# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
This module includes classes and functions relating string coloring and format.

"""
from colorpalette import Color, Palette
import numpy as np

__all__ = ('colors', 'CABBI_colors')

# %% Classes for coloring

# BioSTEAM console
ansicolor = Color.from_ansi
colors = Palette()
colors.dim = ansicolor('dim', '\x1b[37m\x1b[22m')
colors.reset = ansicolor('reset', '\x1b[0m')
colors.exception = ansicolor('bright red', '\x1b[31m\x1b[1m')
colors.next = colors.info = ansicolor('bright blue', '\x1b[34m\x1b[1m')
colors.note = ansicolor('bright cyan', '\x1b[36m\x1b[1m')
colors.dim = ansicolor('dim', '\x1b[37m\x1b[22m')
colors.violet = Color('strong purple', '#e53fe5')

# Main colors
colors.blue = Color('blue', np.array([0.376, 0.757, 0.812]) * 255)
colors.blue_tint = colors.blue.tint(50)
colors.blue_shade = Color('special blue shade', [53, 118, 127])
colors.blue_dark = colors.blue.shade(70)
colors.orange = Color('orange', 255*np.array([0.976, 0.561, 0.376]))
colors.orange_tint = colors.orange.tint(50)
colors.orange_shade = colors.orange.shade(50)
colors.red = Color('red', (241, 119, 127))
colors.red_tint = colors.red.tint(50)
colors.red_shade = colors.red.shade(25)
colors.red_dark = colors.red.shade(45)
colors.green = Color('green', '#79bf82')
colors.green_tint = colors.green.tint(50)
colors.green_shade = colors.green.shade(25)
colors.green_dark = colors.green.shade(45)
colors.yellow = Color('yellow', '#f3c354')
colors.yellow_tint = colors.yellow.tint(45)
colors.yellow_shade = colors.yellow.shade(35)
colors.yellow_dark = colors.yellow.shade(55)
                     
# Contrast colors
colors.green = Color('green', '#33CC33')
colors.green_tint =colors. green.tint(20).shade(15)
colors.green_shade = colors.green.shade(50).tint(40).shade(30)
colors.purple = Color('purple', 255*np.array([0.635, 0.502, 0.725]))
colors.purple_tint = colors.purple.tint(60)
colors.purple_shade = colors.purple.shade(25)
colors.purple_dark = colors.purple.shade(75)

# Neutral colors
colors.brown = Color('brown', [97, 30, 9])
colors.brown_tint = colors.brown.tint(60)
colors.brown_shade = colors.brown.shade(25)
colors.grey = Color('grey', 255*np.array([0.565, 0.569, 0.557]))
colors.grey_tint = colors.grey.tint(25)
colors.grey_shade = colors.grey.shade(25)
colors.grey_dark = colors.grey.shade(50)
colors.neutral = Color('neutral', 255*np.array([0.4824, 0.5216, 0.502]))
colors.neutral_tint = colors.neutral.tint(50)
colors.neutral_shade = colors.neutral.shade(50)
colors.neutral_dark = colors.neutral.shade(75)

# CABBI
colors.CABBI_blue_light = Color('CABBI_blue_light', '#b2e0e5')
colors.CABBI_blue = Color('CABBI_blue', '#82cfd0')
colors.CABBI_teal = Color('CABBI_teal', '#00a996')
colors.CABBI_teal_green = Color('CABBI_teal_green', '#1a8476')
colors.CABBI_green_soft = Color('CABBI_green_soft', '#3ba459')
colors.CABBI_green_dirty = Color('CABBI_green_dirty', '#8ead3e')
colors.CABBI_green = Color('CABBI_green', '#007f3d')
colors.CABBI_yellow = Color('CABBI_yellow', '#ffdd50')
colors.CABBI_orange = Color('CABBI_orange', '#fcb813')
colors.CABBI_grey = Color('CABBI_grey', '#e1deda')
colors.CABBI_brown = Color('CABBI_brown', '#98876e')
colors.CABBI_black = Color('CABBI_black', '#403a48')

CABBI_colors = Palette(
    blue_light = colors.CABBI_blue_light,
    blue = colors.CABBI_blue,
    teal = colors.CABBI_teal,
    teal_green = colors.CABBI_teal_green,
    green_soft = colors.CABBI_green_soft,
    green_dirty = colors.CABBI_green_dirty,
    green = colors.CABBI_green,
    yellow = colors.CABBI_yellow,
    orange = colors.CABBI_orange,
    grey = colors.CABBI_grey,
    brown = colors.CABBI_brown,
    black = colors.CABBI_black,
)
