import pathlib
import pkg_resources
from pkg_resources import resource_stream, Requirement

def version():
    return resource_stream(__name__, "package_version.txt").read()[:-1].decode()
