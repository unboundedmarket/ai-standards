"""
Setup configuration for AI-Blockchain Integration Standards toolbox
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-blockchain-standards",
    version="0.1.0",
    author="Cardano AI Standards Team",
    description="Reference implementation for AI-Blockchain integration standards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unboundedmarket/ai-standards",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-standards-api=ai_standards.api.server:main",
            "ai-standards-certify=ai_standards.certification.cli:main",
            "ai-standards-benchmark=ai_standards.benchmarking.cli:main",
            "ai-standards-consensus=ai_standards.consensus.cli:main",
        ],
    },
)

