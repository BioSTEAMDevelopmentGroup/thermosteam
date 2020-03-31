# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:17:00 2017

@author: Yoel Cortes-Pena
"""
from setuptools import setup
#import numpy

setup(
    name='thermosteam',
    packages=['thermosteam'],
    license='MIT',
    version='0.7.2',
    description="BioSTEAM's Premier Thermodynamic Engine",
    long_description=open('README.rst').read(),
    author='Yoel Cortes-Pena',
    install_requires=['pint>=0.9', 'ht>=0.1.52', 'fluids>=0.1.74',
                      'numba>=0.48.0', 'scipy>=1.3.1', 'IPython>=7.9.0', 
                      'colorpalette>=0.3.0', 'biosteam>=2.7.7',
                      'pandas>=0.25.2', 'graphviz>=0.8.3', 'matplotlib>=3.1.1',
                      'coolprop>=6.3.0', 'numpy>=1.18.1', 'xlrd==1.2.0',
                      'openpyxl>=3.0.0', 'free_properties>=0.2.3',
                      'flexsolve>=0.2.3', 'pyglet', 'sympy'],
    python_requires=">=3.6",
    package_data=
        {'thermosteam': ('base/*', 'equilibrium/*', 'equilibrium/Data/*',
                         'functors/*', 'functors/Data/*', 
                         'reaction/*', 'utils/*', 'utils/decorator_utils/*', 
                         'functors/Data/Critical Properties/*',
                         'functors/Data/Density/*', 
                         'functors/Data/Electrolytes/*', 
                         'functors/Data/Environment/*', 
                         'functors/Data/Heat Capacity/*', 
                         'functors/Data/Identifiers/*',
                         'functors/Data/Law/*', 
                         'functors/Data/Misc/*', 
                         'functors/Data/Misc/element.txt',
                         'functors/Data/Phase Change/*', 
                         'functors/Data/Reactions/*', 
                         'functors/Data/Safety/*', 
                         'functors/Data/Solubility/*', 
                         'functors/Data/Interface/*', 
                         'functors/Data/Triple Properties/*', 
                         'functors/Data/Thermal Conductivity/*', 
                         'functors/Data/Vapor Pressure/*', 
                         'functors/Data/Viscosity/*',
                         'base/units_of_measure.txt', 
                      )},
    platforms=['Windows', 'Mac', 'Linux'],
    author_email='yoelcortes@gmail.com',
    url='https://github.com/BioSTEAMDevelopmentGroup/thermosteam',
    download_url='https://github.com/BioSTEAMDevelopmentGroup/thermosteam.git',
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Mathematics'],
    keywords='thermodynamics chemical engineering mass energy balance material properties phase equilibrium',
)