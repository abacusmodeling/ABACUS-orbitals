#!/bin/bash
#An example for MPI job.
#SBATCH -J example_Si
#SBATCH -o example_Si_%j.log
#SBATCH -e example_Si_%j.err
#SBATCH -p test
#SBATCH --qos=testqos
##SBATCH -p CPU-Shorttime
##SBATCH --qos=qos_cpu_shorttime
#SBATCH -N 1 -n 20 --cpus-per-task=1


echo Time is `date`
echo Directory is `pwd`
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.


module purge 
module load anaconda3_nompi 
module load abacus/2.3.0/intel-2019.update5
#module load anaconda2 2>&1
#module load python/3.9.1
module list 2>&1
source activate pytorch110
#source deactivate pytorch110
#conda deactivate
which python3


MPIRUN=mpirun #Intel mpi and Open MPI
#MPIRUN=mpiexec #MPICH
MPIOPT='-env I_MPI_FABRICS shm:ofi' #Intel MPI
#MPIOPT='--mca mtl_ofi_provider_include psm2' #Open MPI
#MPIOPT='-iface ib0' #MPICH3
timeout 10 $MPIRUN $MPIOPT hostname 
export OMP_NUM_THREADS=20
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"


echo ' python3 -u ../SIAB.py SIAB_INPUT* 2>&1 '
python3 -u ../SIAB.py SIAB_INPUT* 2>&1

