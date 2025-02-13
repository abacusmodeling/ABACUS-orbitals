# Development plan

latest update: 2025-02-11

it seems that the referece structure of the orbital generation task is an open question, it is not clear whether merely the dimer is enough for ensuring the orbital transferability. To support an orbital generation task in complex environments, we need to support further the following two cases:

1. multiple elements. Among which only one element is assigned to the orbital generation task.
2. periodic calculation. Present orbital generation only cares the isolated system, but the bulk system may also be of interest.

There are some codes needed to modify. But before too much technical details, the following json file may be clear and efficient for illustrating the idea:

```json
{
    "dft_calculator": {
        "abacus_command": "abacus",
        "mpi_command": "mpirun -np 8",
        "environment": "",
        "fit_basis": "jy",
        "ecutwfc": 60
    },
    "atoms": [
        {
            "element": "Si",
            "mass": 28.0855,
            "fpsp": "/root/abacus-develop/pseudopotentials/sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.0.upf",
            "forb": null,
            "ecutjy": 20,
            "rcutjy": 10,
            "lmax": 2,
            "orbgen": true
        }
    ],
    "geoms": [
        {
            "proto": "dimer",
            "pertkind": "stretch",
            "pertmags": [1.62, 1.82, 2.22, 2.72, 3.22],
            "types": [0],
            "type_map": [0, 0],
            "dftparam": {
                "nbands": 20,
                "nspin": 1,
                "lmaxmax": 2
            }
        }
    ],
    "generator": {
        "spill_guess": "atomic",
        "optimizer": "torch.swats",
        "torch.lr": 0.001,
        "max_steps": 3000,
        "nthreads_rcut": 4
    },
    "orbs": [
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

The whole orbgen run can be divided into DFT and following spillage optimization runs. In the input file, there are some calculator (ABACUS) setting, such as `mpi_command`, `abacus_command` and `environment`. The `atoms` section supports multiple elements specification, but only one element is assigned to the orbital generation task. The `geoms` section supports multiple geometries, but only one geometry is assigned to the orbital generation task. The `generator` section supports the spillage optimization setting. The `orbs` section supports multiple orbitals, but only one orbital is assigned to the orbital generation task.
