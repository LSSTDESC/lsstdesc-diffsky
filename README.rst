lsstdesc_diffsky
================

Code for producing mock catalogs tailored to LSST DESC requirements.

Installation
------------
To install lsstdesc_diffsky into your environment from the source code::

    $ cd /path/to/root/lsstdesc_diffsky
    $ python setup.py install

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/lsstdesc_diffsky
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

Documentation
-------------
Online documentation for lsstdesc-diffsky is available at
`lsstdesc-diffsky.readthedocs.io <https://lsstdesc-diffsky.readthedocs.io/en/latest/>`_
    
Available Catalogs
------------------
A prototype catalalog is available at NERSC and is called
skysim_v3.1.0

Until lsstdesc-diffsky is part of the desc-python standard environment,
you will need to add the package and its dependencies to desc-python.
The directions for cloning and adding packages to the environment can be found
`here <https://github.com/LSSTDESC/desc-python/wiki/Add-Packages-to-the-desc-python-environment>`_
Once you have built the basic environment, you will need to activate it and install the
`DSPS <https://github.com/ArgonneCPAC/dsps), [diffsky](https://github.com/ArgonneCPAC/diffsky>`_
and the lsstdesc-diffsky packages as described above. If you wish to
use the code in a jupyter notebook, you will also need to install a python kernel as follows::

    $ python -m ipykernel install --user --name=my_kernel_name

`demo_load_catalog <https://github.com/LSSTDESC/lsstdesc-diffsky/tree/main/notebooks/demo_load_catalog.ipynb>`_
is a demo notebook showing how to load the catalog and perform some basic operations.
