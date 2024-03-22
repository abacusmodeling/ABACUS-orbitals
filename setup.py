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
        "scipy",
        "torch",
        "torch_optimizer",
        "torch_complex",
        "addict"
    ],
    # If you need to add package data
    # package_data={'your_package_name': ['data/*.data']},
    zip_safe=False,
    classifiers=[
        # Add classifiers to help others find your project
        # Full list: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change as necessary
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6', # Specify the min version of Python required
)