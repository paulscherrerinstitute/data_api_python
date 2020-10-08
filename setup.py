import os
import pathlib
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def version():
    p = pathlib.Path(__file__).parent.joinpath("package_version.txt")
    with open(p, "r") as f1:
        return f1.read()[:-1]

setup(
    name="data_api",
    version=version(),
    author="Paul Scherrer Institute",
    author_email="daq@psi.ch",
    description=("Interface to PSI's DAQ data- and imagebuffer"),
    license="GPLv3",
    keywords="",
    url="https://github.com/paulscherrerinstitute/data_api_python",
    packages=find_packages(),
    long_description=read('Readme.md'),
    entry_points={
        'console_scripts': ['data_api=data_api2.cli:main']
    },
    data_files = [
        ('', ["package_version.txt"])
    ]
)
