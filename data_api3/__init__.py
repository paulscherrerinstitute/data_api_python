import pathlib
import importlib.resources


def resource_stream(package, resource):
    return importlib.resources.open_binary(package, resource)


def version():
    return resource_stream(__name__, "package_version.txt").read()[:-1].decode()
