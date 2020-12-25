# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
from distutils.core import setup

setup(
    name='thermosteam',
    packages=['thermosteam'],
    license='MIT',
    version='0.22.7',
    description="BioSTEAM's Premier Thermodynamic Engine",
    long_description=open('README.rst').read(),
    author='Yoel Cortes-Pena',
    install_requires=['pint>=0.9',
                      'scipy>=1.4', 
                      'chemicals>=0.1.5',
                      'IPython>=7.9.0', 
                      'colorpalette>=0.3.1', 
                      'biosteam>=2.21.10',
                      'pandas>=0.25.2', 
                      'matplotlib>=3.1.1',
                      'numpy>=1.18.1', 
                      'xlrd==1.2.0',
                      'openpyxl>=3.0.0', 
                      'free_properties>=0.2.5',
                      'flexsolve==0.3.32', 
                      'pyglet',
                      'numba>=0.50.0'],
    extras_require={ # Not in current PyPI version, but will be available in the next
        'dev': [
            'sympy',
            'sphinx', 
            'sphinx_rtd_theme', 
            'pyyaml',
            'pytest-cov',
            'coveralls',
        ]
    }, 
    package_data={
        'thermosteam': [
            'base/*', 
            'equilibrium/*', 
            'reaction/*',
            'utils/*',
            'utils/decorators/*', 
            'mixture/*',
            'chemicals/*',
            'chemicals/Reaction/*',
            'equilibrium/UNIFAC/*',
            'units_of_measure.txt', 
        ]
    },
    platforms=['Windows', 'Mac', 'Linux'],
    author_email='yoelcortes@gmail.com',
    url='https://github.com/BioSTEAMDevelopmentGroup/thermosteam',
    download_url='https://github.com/BioSTEAMDevelopmentGroup/thermosteam.git',
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: OSI Approved :: University of Illinois/NCSA Open Source License',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Manufacturing',
                 'Intended Audience :: Science/Research',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: POSIX :: BSD',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: Implementation :: CPython',
                 'Topic :: Education'],
    keywords=['thermodynamics', 'chemical engineering', 'mass and energy balance', 'material properties', 'phase equilibrium'],
)