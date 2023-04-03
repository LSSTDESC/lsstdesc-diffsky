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
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Some package",
    long_description="Just some package",
    install_requires=["numpy"],
    packages=find_packages(),
    url="https://github.com/aphearin/lsstdesc_diffsky",
    package_data={"lsstdesc_diffsky": ("tests/testing_data/*.dat",)},
)
