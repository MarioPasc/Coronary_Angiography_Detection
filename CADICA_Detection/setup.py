# setup.py
from setuptools import setup, find_packages
import os 

setup(
    name='CADICA_Detection',
    version='0.1',
    author='Mario Pascual González',
    author_email='mpascualg02@gmail.com',
    description='',
    url='https://github.com/yourusername/CADICA_Detection',
    packages=find_packages(),  # Finds all submodules automatically
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "tqdm",
        "Scikit-Learn"

    ],
    python_requires='>=3.9',
)
