import os
from setuptools import setup, find_packages

# os.system("pipreqs --encoding=utf8 --force")

install_requires = []
with open("requirements.txt", "r") as f:
    lines = f.readlines()
    install_requires = [line.rstrip() for line in lines]

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="RLFC-SACIDHP",
    author="C.Teirlinck",
    version="0.1",
    long_description=readme,
    packages=find_packages(),
    install_requires=install_requires,
)
