import json
# FILEPATH: Untitled-1
data = {'EXE_mpi': ['mpirun', '-np', 1], 'EXE_pw': ['abacus'], 'element': 'Si', 'Ecut': 100, 'Rcut': [6, 7], 'Pseudo_dir': ['/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf'], 'Pseudo_name': ['Si_ONCV_PBE-1.0.upf'], 'sigma': 0.01, 'STRU1': ['dimer', 8, 2, 1, 1.8, 2.0, 2.3, 2.8, 3.8], 'STRU2': ['trimer', 10, 2, 1, 1.9, 2.1, 2.6], 'max_steps': [200], 'Level1': ['STRU1', 4, 'none', '1s1p'], 'Level2': ['STRU1', 4, 'fix', '2s2p1d'], 'Level3': ['STRU2', 6, 'fix', '3s3p2d'], 'Save1': ['Level1', 'Z'], 'Save2': ['Level2', 'DZP'], 'Save3': ['Level3', 'TZDP']}

formatted_data = json.dumps(data, indent=4)
print(formatted_data)
