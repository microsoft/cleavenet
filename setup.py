import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setuptools.setup(
    name="cleavenet", # Replace with your own username
    version="0.1.1",
    description="Deep learning model for substrate cleavage prediction and generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/cleavenet",
    license='BSD 2-clause',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=parse_requirements('requirements.txt')
)
