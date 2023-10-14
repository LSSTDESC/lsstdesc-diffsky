Installation instructions
=========================

To install lsstdesc-diffsky into an existing environment with conda-forge::

    $ conda install -c conda-forge lsstdesc-diffsky
    

You can alternatively install with pip. 
To do so starting from a base environment of jax managed by conda-forge::

    $ conda create -c conda-forge -n diffsky_env python=3.9 numpy jax pytest ipython jupyter matplotlib scipy h5py 
    $ conda activate diffsky_env
    $ pip install lsstdesc-diffsky


Managing dependencies
---------------------

The above commands will install all the latest releases of the diffsky dependency chain. 
This includes `numpy <https://numpy.org/>`__ and 
`jax <https://jax.readthedocs.io/en/latest/>`__, 
and also a collection of libraries implementing 
the differentiable modeling ingredients: 
`Diffmah <https://github.com/ArgonneCPAC/diffmah>`_, 
`Diffstar <https://github.com/ArgonneCPAC/diffstar>`_, 
`DSPS <https://github.com/ArgonneCPAC/dsps>`_, 
and `Diffsky <https://github.com/ArgonneCPAC/diffsky>`_.

Depending on your analysis, you may need to install a specific branch 
of lsstdesc-diffsky and/one of its dependencies. You can do this by cloning 
the GitHub repo of the code for which you need a custom version, 
checking out the appropriate version, and running::

    $ pip install . --no-deps
