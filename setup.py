import pathlib
import pkg_resources
from setuptools import setup
from typing import List

# Dynamically generate list of requirements from requirements.txt, falling back to requirements.in
def get_requirements() -> List[str]:
    root = pathlib.Path(__file__).parent
    txt = root / 'requirements.txt'
    ini = root / 'requirements.in'

    source = None
    if txt.exists():
        source = txt
    elif ini.exists():
        source = ini

    if not source:
        # No requirements files present; return empty list to avoid install failure
        return []

    with source.open() as requirements:
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