# ABACUS-orbitals

The file structure under the Orbitals directory is as follows:

```
.
|-- Ag_TZDP
    |-- Ag_gga_10au_100Ry_6s3p3d2f.orb
    |-- Ag_gga_7au_100Ry_6s3p3d2f.orb
    |-- Ag_gga_8au_100Ry_6s3p3d2f.orb
    |-- Ag_gga_9au_100Ry_6s3p3d2f.orb
    `-- info
        |-- 10
        |   |-- INPUT
        |   |-- ORBITAL_RESULTS.txt
        |   `-- SIAB_INPUT
        |-- 9

        ...
        
|-- Ag_DZP

...

```

Among them, The *.orb files are the orbital files required for ABACUS calculations, the *INPUT files under the info directory are the input files, and the numerical orbitals expressed using linear combination coefficients are found in ORBITAL_RESULTS.txt.

The files such as 'Orbitals_TZDP_E100_StandardRcut.json' include the suggested standard Rcut for each element.

The DeltaDFT_PW-LCAO* files contain the computed differences in the single crystal state equation between the PW and LCAO methods using the [DeltaDFT](https://molmod.ugent.be/deltacodesdft) approach, which can serve as an important reference for selecting parameters such as pseudopotentials or orbital radii.

