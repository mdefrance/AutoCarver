from codecs import open
from os import path

from setuptools import setup,find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="AutoCarver",
    version="5.0.2",
    author="Mario DEFRANCE",
    author_email="defrancemario@gmail.com",
    description="Automatic Bucketizing of Features with Optimal Association",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdefrance/AutoCarver",
    project_urls={
        "Bug Tracker": "https://github.com/mdefrance/AutoCarver/issues"
    },
    license="MIT",
    # install_requires= # TODO,
    packages=find_packages(),
    classifiers=[
        # ou 4 - Beta ou 5 - Production/Stable
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        # "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.7",
)
