# CADICA_Detection/setup.py
from setuptools import setup, find_packages

setup(
    name='CADICA_Detection',
    version='0.1',
    author='Mario Pascual González',
    author_email='mpascualg02@gmail.com',
    description='',
    url='https://github.com/yourusername/CADICA_Detection',
    packages=find_packages(include=["CADICA_Detection", "CADICA_Detection.*", "CADICA_Detection.external.ultralytics"]),
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "opencv-python-headless",
        "pyyaml",
        "scienceplots",
        "natsort"
    ],
    python_requires='>=3.9',
    package_data={
        "": ["external/ultralytics/*"]  # Adjust to include all relevant files
    }
)
