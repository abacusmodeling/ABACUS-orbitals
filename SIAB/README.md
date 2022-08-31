# ABACUS-orbitals SIAB Package


## SIAB Description

The full name of SIAB is 

**S**ystematically

**I**mprovable

**A**tomic orbital

**B**asis generator based on spillage formula.

Current, the SIAB program choose the minimization methods "PyTorch Gradient with dpsi (**PTG_dpsi**)" to optimize orbitals. 
The executable files are placed in the "opt_orb_pytorch_dpsi" directory by default.


## HOW TO USE SIAB


###  Set up dependence environment

Firstly, we set up the dependence environment for ABACUS as the followings:

```bash
$ module purge && module load abacus/2.2.2/intel-2018.update4
```
Secondly, we install **pytorch** for **PTG_dpsi** optimization. 

Take the HanHai20@USTC system for example:

```bash
$ module load gcc/9.2.0      #optional, maybe unnecessary.
$ module load anaconda3_nompi
$ module list
Currently Loaded Modulefiles:
  1) anaconda3_nompi                 3) elpa/2021.05.002/intelmpi2018   5) intelmpi/2018.update4
  2) gcc/9.2.0                       4) intel/2018.update4          
$ python3 -V
Python 3.7.4

$ conda create -n pytorch110 python=3.7
$ source activate pytorch110    #or: conda activate pytorch110
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
$ source deactivate             #or: conda deactivate

$ source activate pytorch110    #or: conda activate pytorch110
$ pip3 install --user scipy numpy
$ pip3 install --user torch_optimizer
```

### Write input file

Then, you need to write the input file such as **SIAB_INPUT in example directory "example_Si". There is a comment for each parameter.


### Run

Finally, if the Bash environment has been set up correctly, then run the command as follows:

```bash
cd example_Si
python3 ../SIAB.py ORBITAL_INPUT
```

Or, you can run the script "run.sh" or submit a job as follows:


```bash
cd example_Si

$ ./run.sh
 or submit job for LSF cluster
$ bsub -q idle -n 8 -oo running.log ./run.sh
 or submit job for SLURM cluster
$ sbatch example_Si_sbatch.sh
```


