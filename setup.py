#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup process."""

from io import open
from os import path

from setuptools import find_packages, setup

with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
) as f:
    long_description = f.read()

setup(
    # Basic project information
    name="ttctext",
    version="0.0.1",
    # Authorship and online reference
    author="Satyajit Ghana",
    author_email="satyajitghana7@gmail.com",
    url="https://github.com/extensive-nlp/ttc_nlp",
    # Detailled description
    description="TTC NLP Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="sample setuptools development",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # Package configuration
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">= 3.6",
    install_requires=[
        "torch>=1.9.0",
        "torchtext>=0.10.0",
        "torchmetrics>=0.4.1",
        "omegaconf>=2.1.0",
        "pytorch-lightning>=1.3.8",
        "gdown>=3.13.0",
        "spacy>=3.1.0",
        "pandas~=1.1.0",
        "seaborn>=0.11.1",
        "matplotlib>=3.1.3",
        "tqdm>=4.61.2",
        "scikit-learn~=0.24.2",
    ],
    # Licensing and copyright
    license="Apache 2.0",
)
