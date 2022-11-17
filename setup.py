"""A Python implementation of SAFE"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

from io import open

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SAFEpython',  # Required
    version='0.0.0',  # Required
    description='A Python implementation of of the SAFE toolbox for sensitivity analysis',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://www.safetoolbox.info',  # Optional
    author='Francesca Pianosi, Fanny Sarrazin , Thorsten Wagener',  # Optional
    author_email='fanny.sarrazin@bristol.ac.uk',  # Optional
    packages=find_packages(exclude=['workflows', 'data']),  # Required
#    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=[
        "numpy>=1.13.0",
        "scipy>=0.19.1",
        "numba",
        "matplotlib",
    ],
)
