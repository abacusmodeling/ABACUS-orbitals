from setuptools import setup, find_packages

setup(
    name='ptg_dpsi',
    version='0.2',
    author='ABACUS-AISI developers',
    author_email='',
    description='With auto-differentiation, optimize Spillage function to get the best description of space spanned by planewave, yields atom-centred numerical atomic orbitals',
    packages=find_packages(
        include=['SIAB', 'SIAB.*']
    ),
    install_requires=[
        'numpy',
        'scipy',
    ], # temporarily removed torch dependency for development
    # find source code in SIAB and all subfolders
)
