from setuptools import setup, find_packages

def install_requires(filename):
    with open(filename, "r") as f:
        requirements = f.read().splitlines()
    return requirements

setup(name='cde',
      version='0.0.1',
      packages=find_packages(),
      install_requires=install_requires("requirements.txt"))
