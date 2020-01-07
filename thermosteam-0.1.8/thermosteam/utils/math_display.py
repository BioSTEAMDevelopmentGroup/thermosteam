# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 05:05:15 2020

@author: yoelr
"""

from IPython.display import Image, display
from IPython.lib.latextools import latex_to_png

__all__ = ('MathString', 'MathSection')

class MathSection:
    __slots__ = ('math_strings',)
    def __init__(self, math_strings):
        self.math_strings = tuple([MathString(i) for i in math_strings])
        
    def show(self):
        for i in self.math_strings: i.show()
    _ipython_display_ = show
    
    def __repr__(self):
        return f"{type(self).__name__}({self.math_stings})"


class MathString(str):
    
    def __new__(cls, string):
        return super().__new__(MathString, string)
    
    def __repr__(self):
        return super().__repr__()
        
    def show(self):
        from thermosteam import settings
        color = 'white' if settings.dark_mode else 'black'
        png = latex_to_png("$$" + self + "$$", color=color)
        image = Image(png)
        display(image)
    _ipython_display_ = show