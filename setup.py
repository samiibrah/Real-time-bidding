from setuptools import setup, find_packages

setup(
    name="rtb_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
    ],
    python_requires=">=3.9",
)