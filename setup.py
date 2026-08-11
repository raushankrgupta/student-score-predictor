from setuptools import setup, find_packages
from typing import List
HYPEN_E_DOT = "-e ."

def get_requirements(filepath:str)->List[str]:

    requirements = []
    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [requirement.replace("\n","") for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements
        



setup(
    name = "score-predictor-app",
    version = "1.0.0",
    description = "Predicts Students score",
    author = "Raushan",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)