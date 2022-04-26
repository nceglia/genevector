from setuptools import setup, find_packages

setup(
    name='genevector',
    version='0.0.1',
    description='Single Cell Gene Vector Library',
    packages=find_packages(include=['genevector']),
    install_requires=["scipy","leidenalg","numpy==1.21","notebook","sklearn","cython","pandas","scanpy","umap-learn","tqdm","seaborn","matplotlib"],
)
