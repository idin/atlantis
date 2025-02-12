# Atlantis is a python package for automating data analysis and ML model building

from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='atlantis',
    version='2025.2.6.0',
    author='Idin K',
    author_email='python@idin.net',
    description='A Python package for automating data analysis and ML model building',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/idin/atlantis',
    packages=find_packages(),

    license="Conditional Freedom License (CFL-1.0)",
    install_requires=[
        'plotly>=5.0.0',  
        'pandas>=2.0.0',
        'numpy>=1.23.5',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
) 