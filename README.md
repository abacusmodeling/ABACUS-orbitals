# ABACUS-ORBGEN: v3.0 (alpha)
Welcome to the new version of ABACUS Numerical Atomic Orbital Generation code development repository. We are still working on optimizing the quality of orbital by trying new formulation, new optimization algorithm and strategy. Our goal is to publish orbitals that can be used for not only ground-state involved cases but also for some excitation calculation. Once meet problem, bug in orbital generation, we suggest submit issue on ABACUS Github repository: https://github.com/deepmodeling/abacus-develop.
## Configuration
### Shortcut: for Bohrium(R) users
We have published a configured Bohrium images for users want to save their time as much as possible. You can register in [Bohrium Platform](https://bohrium.dp.tech/), set-up one new container and use Bohrium image `registry.dp.tech/dptech/prod-16047/abacus-orbgen-workshop:20240814`. Then `conda activate orbgen`.
### General: Virtual environment
*WE STRONGLY RECOMMEND TO SET UP A NEW CONDA ENVIRONMENT/VIRTUAL ENVIRONMENT FOR ABACUS-ORBITALS BECAUSE IT CAN AUTOMATICALLY LINK YOUR Pytorch to MKL. OTHERWISE, YOU SHOULD ALWAYS ENSURE THE LINKAGE TO GET THE BEST PERFORMANCE.*
```bash
git clone git clone https://github.com/kirk0830/ABACUS-ORBGEN.git
cd ABACUS-ORBGEN
```
Option1 (the most recommended): If you prefer to use conda, then run the following commands to create a new conda environment and activate it.
```bash
conda create -n orbgen python=3.10
conda activate orbgen
```
Option2: If you prefer to use virtual environment, then run the following commands to create a new virtual environment and activate it.
```bash
python3 -m venv orbgen
source orbgen/bin/activate
```
### Installations
*PERFORMANCE NOTE: WE RECOMMEND USE CONDA TO INSTALL `pytorch` PACKAGE BY `conda install pytorch` FIRST AND INSTALL ALL OTHERS BY FOLLOWING INSTRUCTION BELOW*
*BE AWARE IF Intel-mkl IS SUCCESSFULLY LINKED TO `pytorch`*  
Once the virtual environment is activated (and mkl is ready), run the following commands to install ABACUS-orbitals.
```bash
pip install .
```
## Keywords
Since ABACUS-ORBGEN v3.0, the plaintext input format has been deprecated thoroughly. Up to now, please use JSON format input file. An example of JSON format input file is shown below.

Note: the following document cannot be directly use after copy&paste, because the comments in JSON file is not supported.
```jsonc
{
    // PROGRAM CONFIGURATION
    // *********************
    "environment": "", // you can add `module load *` commands here
    "mpi_command": "mpirun -np 8", // the number of processors used in the calculation
    "abacus_command": "abacus", // path to your ABACUS executable

    // GENERAL
    // *******
    "pseudo_dir": "/path/to/your/pp", // path to your pseudopotential file
    "element": "Si", // the element symbol
    "bessel_nao_rcut": [10],

    // ELECTRONIC STRUCTURE CALCULATION
    // ********************************
    // besides the ecutwfc, all keywords supported by ABACUS can be
    // defined here. For example, smearing_sigma will help for hard
    // -converging systems.
    "ecutwfc": 60,

    // REFERENCE GEOMETRIES
    // ********************
    // orbitals are generated based on wavefunction of specific geometry
    // calculated. One specific geometry is defined via a structral proto 
    // plus a perturbation, specified by its kind and magnitude.
    // NOTE1:
    // the number of bands (nbands), spin channels (nspin) and 
    // lmaxmax (maximal angular momentum of wavefunction) MUST be defined.
    // NOTE2:
    // ABACUS parameters can also be defined here, will overwrite those
    // defined in the general section.
    "geoms": [
        {
            "proto": "dimer",
            "pertkind": "stretch",
            "pertmags": [1.62, 1.82, 2.22, 2.72, 3.22], // unit: Angstrom
            "nbands": 20,
            "nspin": 1,
            "lmaxmax": 2
        }
    ],

    // ORBITAL DEFINITION
    // ******************
    // define the orbital by its number of zeta functions for each angular 
    // momentum.
    "orbitals": [
        {
            "nzeta": [1, 1, 0],
            "geoms": [0],
            "nbands": ["occ", "occ", "occ", "occ", "occ"],
            "checkpoint": null // this is the first orbital to generate
        },
        {
            "nzeta": [1, 1, 1],
            "geoms": [0],
            "nbands": [4, 4, 4, 4, 4],
            "checkpoint": 0 // restart from the first orbital generated
        }
    ],

    // SPILLAGE DEFINITION AND OPTIMIZATION
    // ************************************
    "fit_basis": "jy",
    "primitive_type": "reduced",
    "optimizer": "torch.swats",
    "torch.lr": 0.001,
    "max_steps": 3000,
    "spill_guess": "atomic" // always use `atomic`
}

```
### PROGRAM CONFIGURATION
In this section, user should define the executable files and the number of processors used in the calculation. The executable file of ABACUS is `abacus`. The executable file of MPI is `mpirun`. The number of processors used in the calculation is defined by `mpi_command`. For example, if the number of processors is 4, then `mpi_command` should be `mpirun -np 4`.
* `environment`: the environment configuration load commands, should be organized in one line. Conventional example is like `module load intel/2019.5.281 openmpi/3.1.4 intel-mkl/2019.5.281 intel-mpi/2019.5.281`. If the environment configuration load commands are not needed, then `environment` should be `#environment`.
* `mpi_command`: the executable file of MPI. If the executable file of MPI is in the PATH, then `mpi_command` should be `mpirun`. If the executable file of MPI is not in the PATH, then `mpi_command` should be the absolute path of the executable file of MPI. User may also need to specify the number of processors used in the calculation. For example, if the number of processors is 4, then `mpi_command` should be `mpirun -np 4`. Presently ABACUS does not support other parallelization modes.
* `abacus_command`: the executable file of ABACUS. If the executable file of ABACUS is in the PATH, then `abacus_command` should be `abacus`. If the executable file of ABACUS is not in the PATH, then `abacus_command` should be the absolute path of the executable file of ABACUS.

### GENERAL
* `pseudo_dir`: the directory of the pseudopotential file. If the pseudopotential file is in the current directory, then `pseudo_dir` should be `./`. THIS PARAMETER IS REQUIRED.
* `bessel_nao_rcut`: the realspace cutoff of numerical atomic orbitals to generate, any number of cutoffs can be defined. The unit is Bohr, e.g. `6`, `6 7 8 9 10`. THIS PARAMETER IS REQUIRED.
* `element`: the element symbol. THIS PARAMETER IS REQUIRED.

### ELECTRONIC STRUCTURE CALCULATION
In this section, user should define the parameters used in the electronic structure calculation. As long as the parameters are available in ABACUS, they can be defined in this section. Some necessary and useful parameters are listed below:
* `ecutwfc`: the energy cutoff of the plane wave basis set in Ry, e.g. 100 Ry. To get a good description of the system, the energy cutoff should be large enough. THIS PARAMETER IS REQUIRED.

*For more available keywords, please refer to ABACUS online manual: [Full List of INPUT Keywords](https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html)*

### REFERENCE GEOMETRIES
In this section, user should define the reference gemometries. Reference geometries' wavefunctions are training set of numerical atomic orbitals, therefore the quailities of numerical atomic orbitals are determined by the specifications of reference systems and learning configurations. The parameters are listed below:
* `proto`: the prototype of the reference system. The shape should be `dimer`, `trimer`, `tetramer`, but usually singly a `dimer` is enough, `trimer` is less necessary, and `tetramer` seems always not necessary. THIS PARAMETER IS REQUIRED.
* `pertkind`: the kind of perturbation. Presently only `stretch` is supported. THIS PARAMETER IS REQUIRED.
* `pertmags`: magnitude of perturbation, always should be specified as a list: `[1.62, 1.82, 2.22, 2.72, 3.22]`. For `stretch` pertkind, it is the interatomic distance in unit Angstrom. If specified as `auto`, will use default value built-in ABACUS-ORBGEN code. THIS PARAMETER IS REQUIRED.
* `nbands`: the number of bands to calculate for the reference system. THIS PARAMETER IS REQUIRED.
* `nspin`: the number of spin channels of the reference system. The number of spin channels should be the same as the number of spin channels of the system to calculate. It is always to be 1 but sometimes 2. THIS PARAMETER IS REQUIRED.
* `lmaxmax`: the maximal angular momentum over all atomtypes in present set of geometries. THIS PARAMETER IS REQUIRED.

NOTE: user can define any number of sets of geometries.

### ORBITAL DEFINITION
* `nzeta`: number of zeta functions for each angular momentum, should be provided in a list like `[1, 1, 0]`, which means `1s1p`. WARNING: the length of list should not be larger than the `lmaxmax` THIS PARAMETER IS REQUIRED.
* `geoms`: the indexes of sets of geometries whose wavefunctions will be used to calculate spillage function. Should specify as a list of int. THIS PARAMETER IS REQUIRED.
* `nbands`: the number of bands to refer for geometry. Always it is specifed as a list of int, but can also be list of str, in which the str can be `occ`, `occ+N`, `all`, `all-N`, etc., are "all occupied bands", "occupied bands plus N virtual bands", "all bands calculated in dft procedure", "all bands except N highest bands", respectively. THIS PARAMETER IS REQUIRED.
* `checkpoint`: for hierarchically generating orbitals. Each orbital is generated based on the previous orbital. If `null` is defined, all contraction coefficients will be optimized. If specified as an integer index, will restart orbital generation (spillage optimization) based on checkpoint results. THIS PARAMETER IS REQUIRED.

### SPILLAGE DEFINITION AND OPTIMIZATION
In this section, user should define the spillage function and its optimization. The parameters are listed below:
* `fit_basis`: the reference basis set, can be `pw` or `jy`. THIS PARAMETER IS REQUIRED.
* `primitive_type`: the type of primitive jy basis used to contract to numerical atomic orbitals. Options are `reduced` and `normalized`. If there is no special reason, `reduced` is the only one recommended. THIS PARAMETER IS REQUIRED.
* `optimizer`: the optimizer to use, can be `scipy.bfgs`, `torch.swats`, `torch.yogi`, `torch.adamw`, ..., etc.
* `spill_guess`: the initial guess of Spillage, can be `random` or `atomic`. For `atomic`, an additional ABACUS calculation will run to calculate reference wavefunction of isolated atom. THIS PARAMETER IS OPTIONAL.
* `max_steps`: the maximum optimization on Spillage function to perform. For `optimizer` as `torch.*`, a large number is always suggested, for `bfgs`, optimization will stop if convergence or `max_steps` is reached. THIS PARAMETER IS REQUIRED.

There are other optimizer-specific parameters, such as learning rate of torch optimizer, ..., can be set manually by keywords in the following:
* `torch.lr`: learning rate of torch optimizer. It can be understood as the stepsize. If too large value is specified, the optimization may oscilate or fail. The default value is 1e-3.
* `torch.eps`: a small parameter that add to dominator to avoid numerical instability, always being kept as default value.

## Run
### Common use 
```
orbgen -i jy-v3.0.json
```
But we do not restrict user to name the input script as `jy-v3.0.json`. Once the job is done, you will get orbitals (*.orb) files that can be directly used in ABACUS. PNG-format orbital plots are also generated for quick check on orbital quality.
