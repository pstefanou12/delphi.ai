import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-YOUR-USERNAME-HERE",
    version="0.0.1",
    author="Patroklos Stefanou",
    author_email="patstefanou@gmail.com",
    description="Package for Robust Statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pstefanou12/delphi",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "delphi"},
    packages=setuptools.find_packages(where="delphi"),
    python_requires=">=3.6",
)
