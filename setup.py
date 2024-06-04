import pathlib
import pkg_resources
from setuptools import setup
from typing import List

# Dynamically generate list of requirements from requirements.txt
def get_requirements() -> List[str]:
    with pathlib.Path('requirements.txt').open() as requirements:
        requirements_list = pkg_resources.parse_requirements(requirements)
        return [str(r) for r in requirements_list]


setup(
    name='text_anonymizer',
    version='1.1',
    description='Utility for anonymizing text data',
    author='DataHel',
    author_email='',
    packages=[
        'text_anonymizer'
    ],
    install_requires=get_requirements()
)