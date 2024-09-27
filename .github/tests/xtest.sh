#!/usr/bin/bash

python3 ./SIAB/spillage/basistrans.py -v
python3 ./SIAB/spillage/datparse.py -v
python3 ./SIAB/spillage/index.py -v
python3 ./SIAB/spillage/inputio.py -v
python3 ./SIAB/spillage/jlzeros.py -v
python3 ./SIAB/spillage/linalg_helper.py -v
python3 ./SIAB/spillage/listmanip.py -v
python3 ./SIAB/spillage/orbio.py -v
python3 ./SIAB/spillage/orbscreen.py -v
python3 ./SIAB/spillage/radial.py -v
python3 ./SIAB/spillage/spillage.py -v
python3 ./SIAB/spillage/struio.py -v
python3 ./SIAB/spillage/lcao_wfc_analysis.py -v

python3 ./SIAB/spillage/api.py -v
