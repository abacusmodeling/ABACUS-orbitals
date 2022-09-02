#!/bin/sh
module purge && module load gcc/9.2.0 elpa/2021.05.002/intelmpi2018 intelmpi/2018.update4 2>&1
module load anaconda3_nompi 
# module load anaconda2 2>&1
module list 2>&1;
source activate pytorch110
# source deactivate pytorch110
# conda deactivate
echo "pwd:`pwd`"
ls;
export OMP_NUM_THREADS=20
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "python3 ../SIAB.py SIAB_INPUT* > run.log 2>&1"
python3 ../SIAB.py SIAB_INPUT* 2>&1 | tee run.log
