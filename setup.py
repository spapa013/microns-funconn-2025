#!/usr/bin/env python
from setuptools import setup, find_packages
import re
from os import path

here = path.abspath(path.dirname(__file__))


# Read version from version.py using regex
def get_version():
    version_file = path.join(here, "funconnect", "version.py")
    with open(version_file, "r") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string in %s" % version_file)


setup(
    name="funconnect",
    version=get_version(),
    description="FUNCtional CONNectomics analysis",
    author="Zhuokun Ding",
    packages=find_packages(),
    install_requires=[],
)
