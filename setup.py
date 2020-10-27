from setuptools import setup, find_packages

setup(
    name='compass',
    version='0.0.1',
    description='Single Cell Gene Vector Library',
    packages=find_packages(include=['compass']),
    install_requires=["scipy","numpy","sklearn","torch","pandas","scanpy","umap","tqdm","seaborn"],
)
