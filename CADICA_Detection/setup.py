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
    install_requires = [
        "numpy>=1.23.0",
        "numpy<2.0.0; sys_platform == 'darwin'",  # macOS-specific constraint
        "matplotlib>=3.3.0",
        "opencv-python>=4.6.0",
        "pillow>=7.1.2",
        "pyyaml>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "torch>=1.8.0",
        "torch>=1.8.0,!=2.4.0; sys_platform == 'win32'",  # Windows-specific constraint
        "torchvision>=0.9.0",
        "tqdm>=4.64.0",
        "psutil",
        "py-cpuinfo",
        "pandas>=1.1.4",
        "seaborn>=0.11.0",
        "ultralytics-thop>=2.0.0",
        "optuna",
        "nvidia-ml-py3",
        "iterative-stratification",
        "natsort",
        "Scienceplots"
    ]
    ,
    python_requires='>=3.9',
    package_data={
        "": ["external/ultralytics/*"]  # Adjust to include all relevant files
    }
)
