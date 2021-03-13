Contributing to Thermosteam
===========================

General Process
---------------

Here’s the short summary of how to contribute using git bash:

#. If you are a first-time contributor:

   * Go to https://github.com/BioSTEAMDevelopmentGroup/thermosteam and click the “fork” button to create your own copy of the project.

   * Clone the project to your local computer::
    
        git clone --depth 100 https://github.com/your-username/thermosteam.git
    
   * Change the directory::
    
        cd thermosteam
    
   * Add the upstream repository::
    
        git remote add upstream https://github.com/BioSTEAMDevelopmentGroup/thermosteam.git
    
   * Now, git remote -v will show two remote repositories named "upstream" (which refers to the thermosteam repository), and "origin" (which refers to your personal fork).

#. Develop your contribution:

   * Pull the latest changes from upstream::

       git checkout master
       git pull upstream master

   * Create a branch for the feature you want to work on. Since the branch name will appear in the merge message, use a sensible name such as "Chemical-properties-enhancement"::

       git checkout -b Chemical-properties-enhancement

   * Commit locally as you progress (git add and git commit) Use a properly formatted commit message, write tests that fail before your change and pass afterward, run all the tests locally. Be sure to document any changed behavior in docstrings, keeping to the NumPy docstring standard.

#. To submit your contribution:

   * Push your changes back to your fork on GitHub::

       git push origin Chemical-properties-enhancement

   * Enter your GitHub username and password (repeat contributors or advanced users can remove this step by connecting to GitHub with SSH).

   * Go to GitHub. The new branch will show up with a green Pull Request button. Make sure the title and message are clear, concise, and self- explanatory. Then click the button to submit it.

   * If your commit introduces a new feature or changes functionality, post in https://github.com/BioSTEAMDevelopmentGroup/thermosteam/issues to explain your changes. For bug fixes, documentation updates, etc., this is generally not necessary, though if you do not get any reaction, do feel free to ask for a review.

Testing
-------

First install the developer version of thermosteam:

.. code-block:: bash

   $ cd thermosteam
   $ pip install -e .[dev]

This installs `pytest <https://docs.pytest.org/en/stable/>`__ and other
dependencies you need to run the tests locally. You can run tests by going
to your local thermosteam directory and running the following:

.. code-block:: bash
    
   $ pytest
    
This runs all the `doctests <https://docs.python.org/3.6/library/doctest.html>`__
in thermosteam, which covers most of the API. If any test is marked with a 
letter F, that test has failed. Pytest will point you to the location of the 
error, the values that were expected, and the values that were generated.


Documentation
-------------

Concise and thorough documentation is required for any contribution. Make sure to:

* Use NumPy style docstrings.
* Document all functions and classes.
* Document short functions in one line if possible.
* Mention and reference any equations or methods used and make sure to include the chapter and page number if it is a book or a long document.
* Preview the docs before making a pull request (open your cmd/terminal in the "docs" folder, run "make html", and open "docs/_build/html/index.html").


Authorship
----------

Authorship must be acknowledged for anyone contributing code, significant 
expertise, and/or other impactful efforts. Additionally, authorship should be 
included at the module-level, with a short description of the general contribution. 

If any code or implementation was copied from a third party, it should be rightfully
noted in the module-level documentation along with the corresponding copyright.

Any third-party code copied to the BioSTEAM software must be strictly open-source 
(not copy-left nor open-access). Additionally, if the license is different, 
the module should add the third-party license as an option (dual licensing is OK).


Best practices
--------------

Please refer to the following guides for best practices to make software designs more understandable, flexible, and maintainable:
    
* `PEP 8 style guide <https://www.python.org/dev/peps/pep-0008/>`__.
* `PEP 257 docstring guide <https://www.python.org/dev/peps/pep-0257/>`__.
* `Zen of Python philosophy <https://www.python.org/dev/peps/pep-0020/>`__.
* `SOLID programing principles <https://en.wikipedia.org/wiki/SOLID>`__.
