# RTD docs build shim — real builds use maturin via pyproject.toml
from setuptools import setup, find_packages
setup(
    name="genevector",
    version="1.0.0",
    packages=find_packages(),
)
