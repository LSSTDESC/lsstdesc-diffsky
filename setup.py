import os
from setuptools import setup, find_packages


PACKAGENAME = "lsstdesc_diffsky"
__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "lsstdesc_diffsky", "_version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())


setup(
    name=PACKAGENAME,
    version=__version__,
    author=("Andrew Hearin", "Eve Kovacs"),
    author_email=("ahearin@anl.gov", "kovacs@anl.gov"),
    description="Parallel differentiable generation of synthetic galaxy catalogs",
    long_description="Parallel differentiable generation of synthetic galaxy catalogs",
    install_requires=("numpy", "jax"),
    packages=find_packages(),
    url="https://github.com/LSSTDESC/lsstdesc-diffsky",
    package_data={
        "lsstdesc_diffsky": ("tests/testing_data/*.dat", "param_data/*/*.txt")
    },
)
