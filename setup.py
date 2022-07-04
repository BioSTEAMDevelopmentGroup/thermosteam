# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
from setuptools import setup

setup(
    name='thermosteam',
    packages=['thermosteam'],
    license='MIT',
    version='0.28.9',
    description="BioSTEAM's Premier Thermodynamic Engine",
    long_description=open('README.rst', encoding='utf-8').read(),
    author='Yoel Cortes-Pena',
    install_requires=['pint>=0.9',
                      'scipy>=1.5',
                      'thermo>=0.2.9',
                      'IPython>=7.9.0',
                      'colorpalette>=0.3.3',
                      'pandas>=0.25.2',
                      'matplotlib>=3.1.1',
                      'numpy>=1.18.1,<=1.20',
                      'xlrd>=1.2.0',
                      'openpyxl>=3.0.0',
                      'free_properties>=0.3.4',
                      'flexsolve>=0.5.3',
                      'numba>=0.53.1',
                      'pyglet'],
    extras_require={
        'dev': [
            'biorefineries>=2.23.16',
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
            'thermo/*',
            'units_of_measure.txt',
        ]
    },
    python_requires='>=3.8',
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