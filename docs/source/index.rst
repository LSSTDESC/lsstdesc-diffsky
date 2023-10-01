.. lsstdesc_diffsky documentation master file, created by
   sphinx-quickstart on Mon Apr  3 15:19:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the lsstdesc_diffsky documentation
=============================================
This package contains the code to generate LSST-specific mock galaxy catalogs
from selected gravity-only N-body simulations using a forward modeling approach. 
Currently, the pipeline begins with halo-lightcone healpixels generated from the simulation. 
Each halo is matched by mass to a corresponding halo in UniverseMachine, 
whose galaxy content is then transferred to the simulated halo. 
Each galaxy has an associated parametrized star-formation history.
The code then uses the `DSPS package <https://dsps.readthedocs.io/en/latest/>`__.
to generate SEDs for each galaxy, using a star formation history determined by its 
`diffsky <https://diffsky.readthedocs.io/en/latest/>`__ parameters.
The source code is publicly available 
`on GitHub <https://github.com/LSSTDESC/lsstdesc-diffsky/>`__.


User Guide
----------

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation.rst

.. toctree::
    :maxdepth: 1

    reference.rst

.. toctree::
    :maxdepth: 1

    citation.rst
