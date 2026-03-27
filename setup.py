import pathlib
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
        return [
            line.strip() for line in requirements
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]


setup(
    name='text_anonymizer',
    version='1.2',
    description='Utility for anonymizing text data',
    author='DataHel',
    author_email='',
    packages=[
        'text_anonymizer'
    ],
    install_requires=get_requirements()
)