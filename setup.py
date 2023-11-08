import setuptools
from setuptools import setup, find_packages


# Function to parse requirements from requirements.txt file
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if not line.startswith("#")]


# Use parse_requirements to get the list of requirements
requirements = parse_requirements("requirements.txt")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="competitionassay",
    version="0.0.2",
    author="Eric Schmidt",
    author_email="eric_schmidt_99@gmx.de",
    description="Software for the analysis of the competition assay data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nablaaa/CompetitionAssay",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "all": ["tensorflow==2.12.0", "napari==0.4.18", "n2v==0.3.2"],
    },
    include_package_data=True,
)
