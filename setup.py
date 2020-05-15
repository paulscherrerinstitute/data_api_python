import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="data_api",
    version="0.7.6",
    author="Paul Scherrer Institute",
    author_email="daq@psi.ch",
    description=("Interface to PSI's Data API"),
    license="GPLv3",
    keywords="",
    url="https://github.com/paulscherrerinstitute/data_api_python",
    packages=find_packages(),
    long_description=read('Readme.md'),
    entry_points={
        'console_scripts': ['data_api=data_api2.cli:main']
    }
)
