# ABACUS-ORBGEN
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
    // Environment: program run configuration
    "environment": "", // you can add `module load *` commands here
    "mpi_command": "mpirun -np 8", // the number of processors used in the calculation
    "abacus_command": "abacus", // path to your ABACUS executable

    // General
    "pseudo_dir": "/path/to/your/pp", // path to your pseudopotential file
    "element": "Si", // the element symbol
    "bessel_nao_rcut": [10],

    // ABACUS: dft calculation parameters
    // besides the ecutwfc, all keywords supported by ABACUS can be
    // defined here. For example, smearing_sigma will help for hard
    // -converging systems.
    "ecutwfc": 60,

    // Geometry: definition of structures to calculate.
    // the specific geometry is defined via a structral proto plus a
    // perturbation, specified by its kind and magnitude.
    // the number of bands, spin channels and lmaxmax MUST be defined.
    // ABACUS parameters can also be defined here, will overwrite those
    // defined in the general section.
    "geoms": [
        {
            "proto": "dimer",
            "pertkind": "stretch",
            "pertmags": [1.62, 1.82, 2.22, 2.72, 3.22],
            "nbands": 20,
            "nspin": 1,
            "lmaxmax": 2
        }
    ],

    // Spillage optimization
    "fit_basis": "jy",
    "primitive_type": "reduced",
    "optimizer": "torch.swats",
    "torch.lr": 0.001,
    "max_steps": 3000,
    "spill_guess": "atomic",

    "nthreads_rcut": 4,



    "orbitals": [
        {
            "nzeta": [1, 1, 0],
            "geoms": [0],
            "nbands": ["occ", "occ", "occ", "occ", "occ"],
            "checkpoint": null
        },
        {
            "nzeta": [1, 1, 1],
            "geoms": [0],
            "nbands": [4, 4, 4, 4, 4],
            "checkpoint": 0
        }
    ]
}

```
### PROGRAM CONFIGURATION
In this section, user should define the executable files and the number of processors used in the calculation. The executable file of ABACUS is `abacus`. The executable file of MPI is `mpirun`. The number of processors used in the calculation is defined by `mpi_command`. For example, if the number of processors is 4, then `mpi_command` should be `mpirun -np 4`.
* `environment`: the environment configuration load commands, should be organized in one line. Conventional example is like `module load intel/2019.5.281 openmpi/3.1.4 intel-mkl/2019.5.281 intel-mpi/2019.5.281`. If the environment configuration load commands are not needed, then `environment` should be `#environment`.
* `mpi_command`: the executable file of MPI. If the executable file of MPI is in the PATH, then `mpi_command` should be `mpirun`. If the executable file of MPI is not in the PATH, then `mpi_command` should be the absolute path of the executable file of MPI. User may also need to specify the number of processors used in the calculation. For example, if the number of processors is 4, then `mpi_command` should be `mpirun -np 4`. Presently ABACUS does not support other parallelization modes.
* `abacus_command`: the executable file of ABACUS. If the executable file of ABACUS is in the PATH, then `abacus_command` should be `abacus`. If the executable file of ABACUS is not in the PATH, then `abacus_command` should be the absolute path of the executable file of ABACUS.

### ELECTRONIC STRUCTURE CALCULATION
In this section, user should define the parameters used in the electronic structure calculation. As long as the parameters are available in ABACUS, they can be defined in this section. Some necessary and useful parameters are listed below:
* `pseudo_dir`: the directory of the pseudopotential file. If the pseudopotential file is in the current directory, then `pseudo_dir` should be `./`. THIS PARAMETER IS REQUIRED.
* `pesudo_name`: the name of the pseudopotential file. If the pseudopotential file is `Fe_ONCV_PBE-1.2.upf`, then `pesudo_name` should be `Fe_ONCV_PBE-1.2.upf`. THIS PARAMETER IS REQUIRED.
* `ecutwfc`: the energy cutoff of the plane wave basis set in Ry, e.g. 100 Ry. To get a good description of the system, the energy cutoff should be large enough. THIS PARAMETER IS REQUIRED.
* `bessel_nao_rcut`: the realspace cutoff of numerical atomic orbitals to generate, any number of cutoffs can be defined. The unit is Bohr, e.g. `6`, `6 7 8 9 10`. THIS PARAMETER IS REQUIRED.
* `smearing_sigma`: the smearing parameter in Ry, e.g. 0.015 Ry. This value is the default value. THIS PARAMETER IS OPTIONAL.
* `ks_solver`: the Kohn-Sham solver, can be `cg` or `dav`. THIS PARAMETER IS OPTIONAL.
* `mixing_type`: the mixing type, can be `broyden` or `pulay`. THIS PARAMETER IS OPTIONAL.
* `mixing_ndim`: the number of previous wavefunctions to mix, e.g. 8. THIS PARAMETER IS OPTIONAL.
* `mixing_beta`: the mixing parameter, e.g. 0.7. THIS PARAMETER IS OPTIONAL.

### SIAB PARAMETERS
In this section, user should define the parameters of SIAB. The parameters are listed below:
* `optimizer`: the optimizer to use, can be `pytorch.SWAT` or `bfgs`, THIS PARAMETER IS REQUIRED. *For devleopement use, it can be set as `none`, then the optmization on orbitals will be skipped, the output orbitals are jY basis, which will be significantly large compared with conventional ABACUS NAO.*
* `spill_coefs`: the coefficients of 0 and 1 order derivatives of wavefunction to include in Spillage, e.g. `0.5 0.5`. THIS PARAMETER IS REQUIRED.
* `spill_guess`: the initial guess of Spillage, can be `random`, `identity` or `atomic`. For `atomic`, an additional ABACUS calculation will run to calculate reference wavefunction of isolated atom. THIS PARAMETER IS OPTIONAL.
* `max_steps`: the maximum optimization on Spillage function to perform. For `optimizer` as `pytorch.SWAT`, a large number is always suggested, for `bfgs`, optimization will stop if convergence or `max_steps` is reached. THIS PARAMETER IS REQUIRED.
* `nthreads_rcut`: the number of threads to use for optimizing orbital for each rcut, if not set, will run SIAB in serial. This can significantly reduce the time cost when `optimizer` set as `pytorch.SWAT`. THIS PARAMETER IS OPTIONAL.

### REFERENCE SYSTEMS
In this section, user should define the reference systems. Reference systems' wavefunctions are training set of numerical atomic orbitals, therefore the quailities of numerical atomic orbitals are determined by the specifications of reference systems and learning configurations. The parameters are listed below:
* `shape`: the shape of the reference system. The shape should be `dimer`, `trimer`, `tetramer`, but usually singly a `dimer` is enough, `trimer` is less necessary, and `tetramer` seems always not necessary. THIS PARAMETER IS REQUIRED.
* `nbands`: the number of bands to calculate for the reference system. THIS PARAMETER IS REQUIRED.
* `nspin`: the number of spin channels of the reference system. The number of spin channels should be the same as the number of spin channels of the system to calculate. It is always to be 1 but sometimes 2. THIS PARAMETER IS REQUIRED.
* `bond_lengths`: the bond lengths of the reference system. The unit is Bohr, e.g. `1.8 2.0 2.3 2.8 3.8`. But if `auto` is defined, then the bond lengths will be tested automatically. THIS PARAMETER IS REQUIRED.
* `zeta_notation`: the zeta notation of the orbital to save. The zeta notation is conventionally to be `SZ` (single zeta, the minimal basis), `DZP` (double zeta with one polarization function), `TZDP` (triple zeta with double polarization functions).
* `orb_ref`: for hierarchically generating orbitals. Each orbital is generated based on the previous orbital. If `none` is defined, then the orbital is generated based on the reference system, however if `fix` is defined, then the orbital is generated based on the previous orbital, which means part of coefficients of TSBFs are fixed, whose values would be read from the previous orbital. THIS PARAMETER IS REQUIRED.
* `nbands_ref`: the number of bands to refer to in the reference system. For `optimizer` as `pytorch.SWAT`, if set to `auto`, all occupied bands will be referred to. For `optimizer` as `bfgs`, support flexible options like `occ` which means all occupied bands, `all` means all bands calculated, `occ+4` means `occ` with additional 4 bands.

## Run
### Common use
Up to your case of input, either type:  
```
SIAB_nouvelle -i SIAB_INPUT
```
if you really like the old version input, or:  
```
SIAB_nouvelle -i SIAB_INPUT.json
```
Then you will get orbitals (*.orb) in folders named in the way: \[element\]_xxx. You can also quickly check the quality of orbital by observing the profile of orbitals plot in *.png.
