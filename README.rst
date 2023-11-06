lsstdesc-diffsky
================

lsstdesc-diffsky is a python library based on JAX
for producing mock catalogs tailored to LSST DESC science requirements.


Installation
------------
The latest release of lsstdesc-diffsky is available for installation with pip. 
To do so, we recommend starting from a base environment of jax managed by conda-forge::

    $ conda create -c conda-forge -n diffsky_env python=3.9 numpy jax pytest ipython jupyter matplotlib scipy h5py 
    $ conda activate diffsky_env

To install the latest release::

    $ pip install lsstdesc-diffsky

To install the current tip of the main branch, you will need to clone and install the source code::

    $ cd /path/to/root/lsstdesc-diffsky
    $ pip install .


Documentation
-------------
Online documentation for lsstdesc-diffsky is available at
`lsstdesc-diffsky.readthedocs.io <https://lsstdesc-diffsky.readthedocs.io/en/latest/>`_


Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/lsstdesc-diffsky
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

    
Available Catalogs
------------------
A mock catalalog is available at NERSC and is called roman_rubin_2023_v1.1.0.


Working at NERSC
~~~~~~~~~~~~~~~~

If you want to use the desc-python environment at NERSC, 
you will need to add lsstdesc-diffsky and its dependencies yourself.
The directions for cloning and adding packages to the environment can be found
`here <https://github.com/LSSTDESC/desc-python/wiki/Add-Packages-to-the-desc-python-environment>`_. 
Once you have built the basic environment, you will need to activate it and install 
lsstdesc-diffsky and its dependencies. Note that the appropriate version of the 
diffsky codes can vary with the mock being used.

If you wish to use the code in a jupyter notebook, 
you will also need to install a python kernel as follows::

    $ python -m ipykernel install --user --name=my_kernel_name

`demo_load_catalog <https://github.com/LSSTDESC/lsstdesc-diffsky/tree/main/notebooks/demo_load_catalog.ipynb>`_
is a demo notebook showing how to load the catalog from a notebook running at NERSC 
and perform some basic operations.
