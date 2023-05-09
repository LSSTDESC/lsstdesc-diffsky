Installation instructions
=========================

Dependencies
------------

``lsstdesc-diffsky`` requires `numpy <https://numpy.org/>`__ 
and `jax <https://jax.readthedocs.io/en/latest/>`__, 
and also a collection of libraries implementing 
the differentiable modeling ingredients: 
`Diffmah <https://github.com/ArgonneCPAC/diffmah>`_, 
`Diffstar <https://github.com/ArgonneCPAC/diffstar>`_, 
`DSPS <https://github.com/ArgonneCPAC/dsps>`_,
and `Diffsky <https://github.com/ArgonneCPAC/diffsky>`_.

Installation
------------

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Beginning from an environment that includes numpy and a recent version of jax, 
you can install the additional dependencies available on conda-forge as follows:

       conda install -c conda-forge diffmah diffstar dsps

Depending on the mock you are making, 
Note that for ``diffsky`` and depending on the mock you are making,
for one or more of the above dependencies, 
you may need to install the main branch of the code.
You can do this by cloning the GitHub repository and running the standard::

       python setup.py install

command or its usual variants, such as::

       python setup.py install --prefix=/PATH/TO/INSTALL/DIRECTORY

or::

       pip install .

Installing lsstdesc-diffsky
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have the dependencies installed, 
``lsstdesc-diffsky`` can be installed as usual::

       python setup.py install
