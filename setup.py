from setuptools import setup, find_packages
import versioneer

setup(
    name='genevector',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Single Cell Gene Vector Library',
    packages=find_packages(include=['genevector']),
    install_requires=["scipy","leidenalg","numpy","notebook","sklearn","cython","pandas","scanpy","umap-learn","tqdm","seaborn","matplotlib"],
)
