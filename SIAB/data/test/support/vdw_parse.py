import json
from SIAB.data.interface import PERIODIC_TABLE_TOSYMBOL, ELEMENT_FULLNAME_TOSYMBOL
covalent_radius = {}
with open("SIAB/WolframeAlpha_CovalentRadius.yaml") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line and not line[0].isnumeric():
            continue
        else:
            line = line.split()
            if len(line) <= 1:
                continue
            symbol = ELEMENT_FULLNAME_TOSYMBOL[line[2].capitalize()]
            covalent_radius.update({symbol: float(line[-2])*0.01})

with open("SIAB/covalent_radius.json", "w") as f:
    json.dump(covalent_radius, f)


exit()
zero_charge_rvdw = {}
eq_rvdw = {}
with open("SIAB/rvdw_database_1.dat") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line and not line[0].isalpha():
            continue
        else:
            line = line.split()
            if len(line) <= 1:
                continue
            zero_charge_rvdw.update({line[0]: float(line[1])})
            eq_rvdw.update({line[0]: float(line[-1])})

crystal_rvdw = {}
with open("SIAB/rvdw_database_2.dat") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line and not line[0].isalpha():
            continue
        else:
            line = line.split()
            if len(line) <= 1:
                continue
            crystal_rvdw.update({line[0]: float(line[1])})

calculate_rvdw = {}
with open("SIAB/rvdw_database_3.dat") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line and not line[0].isalpha():
            continue
        else:
            line = line.split()
            if len(line) <= 1:
                continue
            calculate_rvdw.update({line[0]: float(line[1])})

with open("SIAB/rvdw_database.json", "w") as f:
    json.dump({"zero_charge_rvdw": zero_charge_rvdw,
               "eq_rvdw": eq_rvdw,
               "crystal_rvdw": crystal_rvdw,
               "calculate_rvdw": calculate_rvdw}, f)
