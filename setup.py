import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    README = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    REQUIREMENTS = f.readlines()

PKGS = ["SmartDiff"]

setuptools.setup(
    name="Smartdiff",
    version="0.0.12",
    author="Harvard SmartDiff Group",
    author_email="harvard.smartdiff@yahoo.com",
    description="A smart package for Automatic Differentiation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/SmartDiff/cs107-FinalProject",
    packages=PKGS,
    install_requires=REQUIREMENTS,
    include_package_data=True,
    package_data={'SmartDiff': ['SmartDiff/GUI/*.ui']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
