from setuptools import setup, find_packages

# to develop in conda
# install conda-build
# run "conda develop ."
# note conda develop works but does not appear to be maintinated
# https://github.com/conda/conda-build/issues/1992

setup(
    name="enduse",
    version="0.0.1",
    description="Enduse based stockturnover model for load forecasting",
    packages=find_packages(include=["enduse", "enduse.*"]),
)
