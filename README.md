# ABACUS-orbitals
## Configuration
### Virtual environment
*WE STRONGLY RECOMMEND TO SETUP A NEW CONDA ENVIRONMENT/VIRTUAL ENVIRONMENT FOR ABACUS-ORBITALS.*
```bash
git clone https://github.com/kirk0830/abacus_orbital_generation.git
cd abacus_orbital_generation
```
Option1: If you prefer to use conda, then run the following commands to create a new conda environment and activate it.
```bash
conda create -n abacus_orbitals
conda activate abacus_orbitals
```
Option2: If you prefer to use virtual environment, then run the following commands to create a new virtual environment and activate it.
```bash
python3 -m venv abacus_orbitals
source abacus_orbitals/bin/activate
```
### Installations
*PERFORMANCE NOTE: WE RECOMMEND USE CONDA TO INSTALL `pytorch` PACKAGE BY `conda install pytorch` FIRST AND INSTALL ALL OTHERS BY FOLLOWING INSTRUCTION BELOW*
*BE AWARE IF Intel-mkl IS SUCCESSFULLY LINKED TO `pytorch`*  
Once the virtual environment is activated (and mkl is ready), run the following commands to install ABACUS-orbitals.
```bash
pip install -e .
```
## Tutorial (version < 0.2.0)
Find the tutorial in [Gitbook](https://mcresearch.github.io/abacus-user-guide/abacus-nac3.html) written by [Mohan Chen's Group](https://mcresearch.github.io/).
## input parameter description (version < 0.2.0)
An example of input script is shown below:
```bash
# PROGRAM CONFIGURATION
#EXE_env                                
EXE_mpi             mpirun -np 1        
EXE_pw              abacus              

# ELECTRONIC STRUCTURE CALCULATION
element             Fe                  
Ecut                100                 
Rcut                6 7 8 9 10          
Pseudo_dir          ./download/pseudopotentials/sg15_oncv_upf_2020-02-06/1.2
Pseudo_name         Fe_ONCV_PBE-1.2.upf 
smearing_sigma      0.015               

# REFERENCE SYSTEMS
# identifier     shape          nbands         lmax           nspin          bond_lengths   
  STRU1          dimer          8              3              1              1.8 2.0 2.3 2.8 3.8
  STRU2          trimer         10             3              1              1.9 2.1 2.6    

# SIAB PARAMETERS
max_steps 9000
# orb_id         stru_id        nbands_ref     orb_ref        orb_config     
  Level1         STRU1          4              none           2s1p1d         
  Level2         STRU1          4              fix            4s2p2d1f       
  Level3         STRU2          6              fix            6s3p3d2f       

# SAVE
# save_id        orb_id         zeta_notation  
  Save1          Level1         SZ             
  Save2          Level2         DZP            
  Save3          Level3         TZDP           
```
### PROGRAM CONFIGURATION
In this section, user should define the executable files and the number of processors used in the calculation. The executable file of ABACUS is `abacus`. The executable file of MPI is `mpirun`. The number of processors used in the calculation is defined by `EXE_mpi`. For example, if the number of processors is 4, then `EXE_mpi` should be `mpirun -np 4`.
* `EXE_env`: the environment configuration load commands, should be organized in one line. Conventional example is like `module load intel/2019.5.281 openmpi/3.1.4 intel-mkl/2019.5.281 intel-mpi/2019.5.281`. If the environment configuration load commands are not needed, then `EXE_env` should be `#EXE_env`.
* `EXE_mpi`: the executable file of MPI. If the executable file of MPI is in the PATH, then `EXE_mpi` should be `mpirun`. If the executable file of MPI is not in the PATH, then `EXE_mpi` should be the absolute path of the executable file of MPI. User may also need to specify the number of processors used in the calculation. For example, if the number of processors is 4, then `EXE_mpi` should be `mpirun -np 4`. Presently ABACUS does not support other parallelization modes.
* `EXE_pw`: the executable file of ABACUS. If the executable file of ABACUS is in the PATH, then `EXE_pw` should be `abacus`. If the executable file of ABACUS is not in the PATH, then `EXE_pw` should be the absolute path of the executable file of ABACUS.

### ELECTRONIC STRUCTURE CALCULATION
In this section, user should define the parameters used in the electronic structure calculation. The parameters are listed below:
* `element`: the element of the system. The element should be the same as the element in the pseudopotential file. If this parameter is not defined, then the element will be read from the pseudopotential file.
* `Ecut`: the energy cutoff of the plane wave basis set in Ry, e.g. 100 Ry. To get a good description of the system, the energy cutoff should be large enough. THIS PARAMETER IS REQUIRED.
* `Rcut`: the realspace cutoff of numerical atomic orbitals to generate, any number of cutoffs can be defined. The unit is Bohr, e.g. `6`, `6 7 8 9 10`. THIS PARAMETER IS REQUIRED.
* `Pseudo_dir`: the directory of the pseudopotential file. If the pseudopotential file is in the current directory, then `Pseudo_dir` should be `./`. THIS PARAMETER IS REQUIRED.
* `Pseudo_name`: the name of the pseudopotential file. If the pseudopotential file is `Fe_ONCV_PBE-1.2.upf`, then `Pseudo_name` should be `Fe_ONCV_PBE-1.2.upf`. THIS PARAMETER IS REQUIRED.
* `smearing_sigma`: the smearing parameter in Ry, e.g. 0.015 Ry. This value is the default value. THIS PARAMETER IS OPTIONAL.

### REFERENCE SYSTEMS
In this section, user should define the reference systems. Reference systems' wavefunctions are training set of numerical atomic orbitals, therefore the quailities of numerical atomic orbitals are determined by the specifications of reference systems and learning configurations. The parameters are listed below:
* `identifier`: the identifier of the reference system. The identifier should be unique. THIS PARAMETER IS REQUIRED.
* `shape`: the shape of the reference system. The shape should be `dimer`, `trimer`, `tetramer`, but usually singly a `dimer` is enough, `trimer` is less necessary, and `tetramer` seems always not necessary. THIS PARAMETER IS REQUIRED.
* `nbands`: the number of bands to calculate for the reference system. THIS PARAMETER IS REQUIRED.
* `lmax`: the maximum angular momentum of Truncated Spherical Bessel Functions (TSBFs) to generate. This parameter should be set according to pseudopotential valence electron configuration and zeta configuration of numerical atomic orbital wanted. THIS PARAMETER IS REQUIRED.
* `nspin`: the number of spin channels of the reference system. The number of spin channels should be the same as the number of spin channels of the system to calculate. It is always to be 1 but sometimes 2. THIS PARAMETER IS REQUIRED.
* `bond_lengths`: the bond lengths of the reference system. The unit is Bohr, e.g. `1.8 2.0 2.3 2.8 3.8`. But if `auto` is defined, then the bond lengths will be tested automatically. THIS PARAMETER IS REQUIRED.

### SIAB PARAMETERS
In this section, user should define the parameters of SIAB. The parameters are listed below:
* `max_steps`: the maximum optimization on Spillage function to perform. THIS PARAMETER IS REQUIRED.
* `orb_id`: the identifier of the orbital to generate.
* `stru_id`: the identifier of the reference system to use.
* `nbands_ref`: the number of bands to refer to in the reference system, if set to `auto`, all occupied bands will be referred to.
* `orb_ref`: for hierarchically generating orbitals. Each orbital is generated based on the previous orbital. If `none` is defined, then the orbital is generated based on the reference system, however if `fix` is defined, then the orbital is generated based on the previous orbital, which means part of coefficients of TSBFs are fixed, whose values would be read from the previous orbital. THIS PARAMETER IS REQUIRED.
* `orb_config`: orbital configuration. This keywords defines orbital configuration with more details than the zeta notation like `DZP` or `TZDP`. The value should refer to the valence electron configuration of the element in the pseudopotential file. For example, if the valence electron configuration of the element is `3s2 3p6 3d6 4s2`, then for `DZP` the value should be `4s2p2d1f`. If set to `auto`, then pseudopotential will be parsed to get the valence electron configuration.

### SAVE
In this section, user should define the orbitals to save. The parameters are listed below:
* `save_id`: the identifier of the save work. The identifier should be unique. THIS PARAMETER IS REQUIRED.
* `orb_id`: the identifier of the orbital to save.
* `zeta_notation`: the zeta notation of the orbital to save. The zeta notation is conventionally to be `SZ` (single zeta, the minimal basis), `DZP` (double zeta with one polarization function), `TZDP` (triple zeta with double polarization functions).

## input parameter description (version >= 0.2.0)
In version >= 0.2.0, many parameters are removed due to redundancy. The input script is shown below:
```json
{
    "environment": "",
    "mpi_command": "mpirun -np 8",
    "abacus_command": "abacus",

    "pseudo_dir": "/root/abacus-develop/pseudopotentials/sg15_oncv_upf_2020-02-06/1.0",
    "pseudo_name": "Si_ONCV_PBE-1.0.upf",
    "ecutwfc": 60,
    "bessel_nao_rcut": [6, 7, 8, 9, 10],
    "smearing_sigma": 0.01,

    "optimizer": "pytorch.SWAT",
    "max_steps": 1000,
    "spillage_coeff": [0.5, 0.5],
    "nthreads_rcut": 4,

    "reference_systems": [
        {
            "shape": "dimer",
            "nbands": 8,
            "nspin": 1,
            "bond_lengths": [1.62, 1.82, 2.22, 2.72, 3.22]
        },
        {
            "shape": "trimer",
            "nbands": 10,
            "nspin": 1,
            "bond_lengths": [1.9, 2.1, 2.6]
        }
    ],
    
    "orbitals": [
        {
            "zeta_notation": "Z",
            "shape": "dimer",
            "nbands_ref": 4,
            "orb_ref": "none"
        },
        {
            "zeta_notation": "DZP",
            "shape": "dimer",
            "nbands_ref": 4,
            "orb_ref": "Z"
        },
        {
            "zeta_notation": "TZDP",
            "shape": "trimer",
            "nbands_ref": 6,
            "orb_ref": "DZP"
        }
    ]
}
```
or given in plain text:
```bash
# PROGRAM CONFIGURATION
environment         module load intel/2019.5.281 openmpi/3.1.4 intel-mkl/2019.5.281 intel-mpi/2019.5.281
mpi_command         mpirun -np 1
abacus_command      abacus
# ELECTRONIC STRUCTURE CALCULATION
pseudo_dir          ./download/pseudopotentials/sg15_oncv_upf_2020-02-06/1.2
pesudo_name         Fe_ONCV_PBE-1.2.upf
ecutwfc             100
bessel_nao_rcut     6 7 8 9 10
smearing_sigma      0.015        # optional, default 0.015
ks_solver           cg           # optional, default dav
mixing_type         broyden      # optional, default broyden
mixing_ndim         8            # optional, default 8
mixing_beta         0.7          # optional, default 0.7
# SIAB PARAMETERS
optimizer           pytorch.SWAT # optimizers, can be pytorch.SWAT, SimulatedAnnealing, ...
spillage_coeff      0.5 0.5      # order of derivatives of wavefunction to include in Spillage, can be 0 or 1.
max_steps           9000
# REFERENCE SYSTEMS
# shape    nbands    nspin    bond_lengths   
  dimer    8         1        auto
  trimer   10        1        1.9 2.1 2.6
# ORBITALS
# zeta_notation    shape    nbands_ref   orb_ref
  SZ               dimer    4            none
  DZP              dimer    4            SZ
  TZDP             trimer   6            DZP
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
* `optimizer`: the optimizer to use, can be `pytorch.SWAT`, `SimulatedAnnealing`, and optimizers from `scipy.optimize`. THIS PARAMETER IS REQUIRED.
* `spillage_coeff`: the coefficients of 0 and 1 order derivatives of wavefunction to include in Spillage, e.g. `0.5 0.5`. THIS PARAMETER IS REQUIRED.
* `max_steps`: the maximum optimization on Spillage function to perform. THIS PARAMETER IS REQUIRED.
* `nthreads_rcut`: the number of threads to use for optimizing orbital for each rcut, if not set, will run SIAB in serial. THIS PARAMETER IS OPTIONAL.

### REFERENCE SYSTEMS
In this section, user should define the reference systems. Reference systems' wavefunctions are training set of numerical atomic orbitals, therefore the quailities of numerical atomic orbitals are determined by the specifications of reference systems and learning configurations. The parameters are listed below:
* `shape`: the shape of the reference system. The shape should be `dimer`, `trimer`, `tetramer`, but usually singly a `dimer` is enough, `trimer` is less necessary, and `tetramer` seems always not necessary. THIS PARAMETER IS REQUIRED.
* `nbands`: the number of bands to calculate for the reference system. THIS PARAMETER IS REQUIRED.
* `nspin`: the number of spin channels of the reference system. The number of spin channels should be the same as the number of spin channels of the system to calculate. It is always to be 1 but sometimes 2. THIS PARAMETER IS REQUIRED.
* `bond_lengths`: the bond lengths of the reference system. The unit is Bohr, e.g. `1.8 2.0 2.3 2.8 3.8`. But if `auto` is defined, then the bond lengths will be tested automatically. THIS PARAMETER IS REQUIRED.
* `zeta_notation`: the zeta notation of the orbital to save. The zeta notation is conventionally to be `SZ` (single zeta, the minimal basis), `DZP` (double zeta with one polarization function), `TZDP` (triple zeta with double polarization functions).
* `orb_ref`: for hierarchically generating orbitals. Each orbital is generated based on the previous orbital. If `none` is defined, then the orbital is generated based on the reference system, however if `fix` is defined, then the orbital is generated based on the previous orbital, which means part of coefficients of TSBFs are fixed, whose values would be read from the previous orbital. THIS PARAMETER IS REQUIRED.
* `nbands_ref`: the number of bands to refer to in the reference system, if set to `auto`, all occupied bands will be referred to.
