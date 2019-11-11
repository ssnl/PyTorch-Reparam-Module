#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.md').read()

VERSION = '0.0.1'

setup(
    # Metadata
    name='torchreparam',
    version=VERSION,
    author='Tongzhou Wang',
    author_email='tongzhou.wang.1994@gmail.com',
    url='https://github.com/SsnL/PyTorch-Reparam-Module',
    description='Reparameterize your PyTorch modules',
    long_description=readme,
    license='MIT',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
)
