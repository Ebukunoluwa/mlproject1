from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Return list of requirements, ignoring '-e .'"""
    with open(file_path) as f:
        return [req.strip() for req in f if req.strip() and req.strip() != HYPEN_E_DOT]

setup(
    name="mlproject",
    version="0.1.0",
    author="Ibukun",
    author_email="ibukunoluwaogundare@gmail.com",
    url="https://github.com/yourusername/your-repo",
    packages=find_packages(exclude=()),
    install_requires=get_requirements('requirements.txt'),
    extras_require={
        "dev": ["pytest", "black", "ruff"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
