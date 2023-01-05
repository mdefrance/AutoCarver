import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='toolbox_dsa',
    version='0.0.1',
    author='Mario DEFRANCE',
    author_email='defrancemario@gmail.com',
    description='Release 1.0 AutoCarver',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://scm.saas.cagip.group.gca/scoad/toolbox_dsa',
    #project_urls = {
    #    "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    #},
    license='MIT',
    packages=['AutoCarver']
)
