import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="competitionassay",
    version="0.0.6",
    author="Eric Schmidt",
    author_email="eric_schmidt_99@gmx.de",
    description="Software for the analysis of the competition assay data.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Nablaaa/CompetitionAssay",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-image",
        "pandas",
    ],
    include_package_data=True,
)
