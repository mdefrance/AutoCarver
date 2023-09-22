from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="AutoCarver",
    version="5.2.1",
    author="Mario DEFRANCE",
    author_email="defrancemario@gmail.com",
    description="Automatic Discretization of Features with Optimal Target Association",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdefrance/AutoCarver",
    project_urls={
        "Documentation": "https://autocarver.readthedocs.io/en/latest/index.html",
        "Bug Tracker": "https://github.com/mdefrance/AutoCarver/issues",
    },
    license="MIT",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tqdm",
    ],
    extras_require={
        "jupyter": [
            "ipython",
        ]
    },
    packages=find_packages(),
    classifiers=[
        # ou 4 - Beta ou 5 - Production/Stable
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.9",
)
