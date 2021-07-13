from setuptools import setup, find_packages

__version__ = '0.0.1'

setup(
    name="mlthon",
    version=__version__,
    description="This package makes machine learning easier!",
    author="Pritish Mishra",
    author_email="pritishjan@gmail.com",
    url="https://github.com/pritishmishra703/MLthon.git",
    download_url="https://github.com/pritishmishra703/MLthon/tarball/master",
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib']
)
