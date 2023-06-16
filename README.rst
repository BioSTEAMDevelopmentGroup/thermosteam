====================================================
ThermoSTEAM: BioSTEAM's Premier Thermodynamic Engine 
====================================================

.. image:: http://img.shields.io/pypi/v/thermosteam.svg?style=flat
   :target: https://pypi.python.org/pypi/thermosteam
   :alt: Version_status
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://biosteam.readthedocs.io/en/latest/API/thermosteam/index.html
   :alt: Documentation
.. image:: https://img.shields.io/pypi/pyversions/thermosteam.svg
   :target: https://pypi.python.org/pypi/thermosteam
   :alt: Supported_versions
.. image:: http://img.shields.io/badge/license-UIUC-blue.svg?style=flat
   :target: https://github.com/BioSTEAMDevelopmentGroup/thermosteam/blob/master/LICENSE.txt
   :alt: license
.. image:: https://coveralls.io/repos/github/BioSTEAMDevelopmentGroup/thermosteam/badge.svg?branch=master
   :target: https://coveralls.io/github/BioSTEAMDevelopmentGroup/thermosteam?branch=master
   :alt: Coverage
.. image:: https://joss.theoj.org/papers/10.21105/joss.02814/status.svg
   :target: https://doi.org/10.21105/joss.02814

.. contents::

Workshops
---------
Join us on Friday, Jan 20, 9:15-10:15am CST, for a BioSTEAM workshop! 
Email biosteamdevelopmentgroup@gmail.com for details.

What is ThermoSTEAM?
--------------------

ThermoSTEAM is a standalone thermodynamic engine capable of estimating mixture 
properties, solving thermodynamic phase equilibria, and modeling stoichiometric 
reactions. ThermoSTEAM builds upon `chemicals <https://github.com/CalebBell/chemicals>`_, 
the chemical properties component of the Chemical Engineering Design Library, 
with a robust and flexible framework that facilitates the creation of property packages.  
`The Biorefinery Simulation and Techno-Economic Analysis Modules (BioSTEAM) <https://biosteam.readthedocs.io/en/latest/>`_ 
is dependent on ThermoSTEAM for the simulation of unit operations.

Installation
------------

Get the latest version of ThermoSTEAM from `PyPI <https://pypi.python.org/pypi/thermosteam/>`_.
If you have an installation of Python with pip, simple install it with::

    $ pip install thermosteam

To get the git version and install it, run::

    $ git clone --depth 100 git://github.com/BioSTEAMDevelopmentGroup/thermosteam
    $ cd thermosteam
    $ pip install .

We use the `depth` option to clone only the last 100 commits. Thermosteam has a 
long history, so cloning the whole repository (without using the depth option)
may take over 30 min.

If you would like to clone all branches, add the "--no-single-branch" flag as such::

    $ git clone --depth 100 --no-single-branch git://github.com/BioSTEAMDevelopmentGroup/thermosteam

Documentation
-------------

ThermoSTEAM's documentation is available on the web:

    https://biosteam.readthedocs.io/en/latest/API/thermosteam/index.html

Bug reports
-----------

To report bugs, please use the ThermoSTEAM's Bug Tracker at:

    https://github.com/BioSTEAMDevelopmentGroup/thermosteam


License information
-------------------

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the ThermoSTEAM license, if it is convenient for you,
please cite ThermoSTEAM if used in your work. Please also consider contributing
any changes you make back, and benefit the community.


Citation
--------

To cite ThermoSTEAM in publications use::

    Cortes-Pena, Y., (2020). ThermoSTEAM: BioSTEAM's Premier Thermodynamic Engine. 
    Journal of Open Source Software, 5(56), 2814. doi.org/10.21105/joss.02814
