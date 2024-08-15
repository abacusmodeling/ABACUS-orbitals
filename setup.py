from setuptools import setup, find_packages

setup(
    name="SIAB",
    version="0.2",
    author="ABACUS-AISI developers",
    author_email="huangyk@aisi.ac.cn",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy<1.10", # required by some functionalities employing the old version simpson integral
        "torch",
        "torch_optimizer",
        "torch_complex",
        "addict"
    ],
    zip_safe=False,
    classifiers=[
        # Add classifiers to help others find your project
        # Full list: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change as necessary
        "Operating System :: OS Independent",
    ],
    scripts=[
        "SIAB/SIAB_nouvelle.py",
    ],
   python_requires='<3.11', # required by SciPy 1.10
)
