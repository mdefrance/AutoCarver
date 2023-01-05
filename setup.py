import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='AutoCarver',
    version='1.0.0',
    author='Mario DEFRANCE',
    author_email='defrancemario@gmail.com',
    description='Automatic Carving of Features with Optimal Association',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mdefrance/AutoCarver',
    project_urls = {
        "Bug Tracker": "https://github.com/mdefrance/AutoCarver/issues"
    },
    license='MIT',
    packages=['AutoCarver']
)
