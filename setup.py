import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='AutoCarver',
    version='4.2.1',
    author='Mario DEFRANCE',
    author_email='defrancemario@gmail.com',
    description='Automatic Bucketizing of Features with Optimal Association',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mdefrance/AutoCarver',
    project_urls = {
        "Bug Tracker": "https://github.com/mdefrance/AutoCarver/issues"
    },
    license='MIT',
    packages=['AutoCarver'],
    classifiers=[
        'Development Status :: 3 - Alpha',  # ou 4 - Beta ou 5 - Production/Stable
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    python_requires='>=3.7'
)
