from setuptools import setup, find_packages

def install_requires(filename):
    with open(filename, "r") as f:
        requirements = f.read().splitlines()
    return requirements

setup(name='cdest',
      version='0.0.1',
      package_dir={'': 'src'},
      install_requires=install_requires("requirements.txt"))
