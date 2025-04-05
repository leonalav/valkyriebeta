import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setuptools.setup(
    name="valkyrietest",
    version="0.1.0",
    author="abctest01",
    author_email="imperialgamer502@gmail.com",
    description="Valkyrie Language Learning Model with enhanced reasoning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "valkyrietest-train=valkyrie_llm.cli.train:main",
            "valkyrietest-train-fineweb=valkyrie_llm.cli.train_fineweb:main",
            "valkyrietest-inference=valkyrie_llm.cli.inference:main",
        ],
    },
)
