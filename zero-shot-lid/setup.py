"""
Setup script for Zero-Shot Spoken Language Identification.

This package provides a complete framework for zero-shot language identification
using phonological features and deep learning.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="zero-shot-lid",
    version="1.0.0",
    author="Yaswanth Setty",
    author_email="yaswanthsetty@example.com",  # Replace with actual email
    description="Zero-Shot Spoken Language Identification using Phonological Features",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition",
    project_urls={
        "Bug Tracker": "https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition/issues",
        "Documentation": "https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition/tree/main/zero-shot-lid",
        "Source Code": "https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition/tree/main/zero-shot-lid",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zero-shot-lid=main:main",
        ],
    },
    keywords=[
        "language-identification",
        "zero-shot-learning",
        "speech-processing",
        "phonological-features",
        "deep-learning",
        "pytorch",
        "wav2vec2",
        "cross-lingual",
    ],
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
)