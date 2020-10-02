import pathlib

def version():
    p = pathlib.Path(__file__).parent.parent.joinpath("package_version.txt")
    with open(p, "r") as f1:
        return f1.read()[:-1]
