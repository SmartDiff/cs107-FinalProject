import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pkg-smartdiff", 
    version="0.0.1",
    #author="Name Lastname",
    #author_email="example@gmail.com",
    description="A smart package for Automatic Differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SmartDiff/cs107-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)