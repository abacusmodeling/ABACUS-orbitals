#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
import sys
sys.stdout = Unbuffered(sys.stdout)
import os
path_thisfile = os.path.dirname(os.path.realpath(__file__))
import re
import json
import time
import copy
import math
import numpy as np
import subprocess
import string
from io import StringIO
# from string import ljust
# from _elementtree import Element
# from distutils.cygwinccompiler import get_versions
# from scipy.io.array_import import default
# #
#path_thisfile = os.path.dirname(os.path.realpath(__file__))
#sys.path.append( path_thisfile )
#sys.path.append( os.path.realpath( path_thisfile+'/../' ) )
#
##allinone avoid print buffer, instead of use sys.stdout.flush() after each print
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)
##not work: os.environ["PYTHONUNBUFFERED"] = "1"
##print " zero print_buffer " 
##print( " ", flush=True)


def get_fileString(file_fullPath):
    file = open( file_fullPath )
    data = file.read()
    return data

def get_string_linehead( headString, input_string ):
    linematch = re.search(r"^[ ]*"+headString+"[^#^\n]*", input_string, flags=re.DOTALL|re.MULTILINE)
    #localarray = re.split( r"[ ]+", linematch.group(), maxsplit=1 )
    if linematch:
        localarray = linematch.group().split(maxsplit=1)
        return localarray[1].strip()
    else:
        return ""
        print( " Missing %s"%headString )

def get_nRows_linehead( headString, input_string ):
    linematch = re.findall(r"^[ ]*"+headString+"[^#^\n]*", input_string, flags=re.DOTALL|re.MULTILINE)
    #print(len(linematch))
    return len(linematch)

def get_array_linehead( headString, input_string ):
    linematch = re.search(r"^[ ]*"+headString+"[^#^\n]*", input_string, flags=re.DOTALL|re.MULTILINE)
    #localarray = re.split( r"[ ]+", linematch.group() )
    localarray = linematch.group().split()
    return localarray[1:]

def strs_to_ints( array ):
    return [ int(ii) for ii in array ]

def strs_to_floats( array ):
    return [ float(ii) for ii in array ]

def str_to_bool(str):
    return str == "True" or str == "true"

def strs_to_bools(array):
    return [ str == "True" or str == "true" for str in array ]

def Search_Num_nearStr(dftlogfpath,theStr=r'!FINAL_ETOT_IS', NumIndex=0 ):
    if os.path.isfile(dftlogfpath) :
        dftlogfile = open( dftlogfpath )
        #print "    searching num near %s in %s :"%(theStr,dftlogfpath)
        for logline in dftlogfile :
            linematch = re.search(theStr, logline)
            if linematch :
                #print logmatch.group(0) 
                print(logline.strip())
                #num_match  = re.search("[-+]?[0-9]*\.?[0-9]+",logline)
                #num_match  = re.findall("[-+]?[0-9]*\.?[0-9]+",logline)
                num_match  = re.findall(r'[-+]?\d+\.?\d*[eE]?[-+]?\d+',logline)
                #total_E = string.atof( energy_match.group(0) ) 
                #print total_E * 2 
                if num_match:
                    print(num_match)
                    #return float( num_match.group(0) ) #* 13.60569253 #13.605698
                    return float(num_match[NumIndex])
    return 0.0
#Search_Num_nearStr("./Cu/outdata/pwscf.xml",theStr=r'<etot>')
#print "%.12f"%Search_Num_nearStr("26_Fe_100/OUT.Fe-10-1.7/running_scf.log")

def parse_arguments():
    #global args
    import argparse
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional Description')
    
    # Required positional argument
    parser.add_argument('InputFile', type=str, nargs='?',
                    help='Absolute or relative path including the name of the input file')

    parser.add_argument('--HostList', type=str,
                    help='Host list separated by commas ')
    args = parser.parse_args()
    if args.InputFile == None:
        args.InputFile = "SIAB_INPUT"
    if args.HostList == None:
        args.HostList = "localhost"
    return args

def orbConf_to_list(str, Llabel, maxL=4):
    #input_list = list( str )
    input_list = list( str.ljust(10) )
    #print("input_list:%s"%input_list)
    results_list = []
    Lvalue=0
    index_list = list( range(0, int(len(input_list)/2) ) ) #list( range(0, len(Llabel) ) ) 
    #print("index_list: %s"%index_list)
    for ii in index_list: # range(len(orbConfList)/2 ):
        label_index = 2*ii + 1
        #print(" === label_index:%s, L:%s, input:%s, Llabel:%s."%(label_index, Lvalue, input_list[label_index], Llabel[Lvalue]) )
        if Lvalue == maxL+1: 
            break
        while input_list[label_index] != Llabel[Lvalue]:
            results_list.append(0)
            #print(" --- results_list:%s label_index:%s"%(results_list,label_index) )
            Lvalue+=1
            #print(" --- get for L=%s"%Lvalue )
            if Lvalue == maxL+1:
                break
        if Lvalue == maxL+1: 
            break
        else:
            results_list.append( int(input_list[2*ii]) )
            #print(" --- results_list:%s label_index:%s"%(results_list,label_index) )
            Lvalue+=1
            #print(" --- get for L=%s"%Lvalue )
    #print( " Orbitals List = %s"%( results_list ), end='\n' ) 
    return results_list
orbConf_to_list('1s1p' , ['s','p','d','f','g'], 2)

def dimer(element, mass, fpseudo, lattice_constant, bond_length, nspin):
    """generate dimer structure"""
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f%10.8f%10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f%10.8f%10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f%10.8f%10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "2       //number of atoms\n"
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    return result

def trimer(element, mass, fpseudo, lattice_constant, bond_length, nspin):
    """generate trimer structure"""
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    dis1 = bond_length * 0.86603
    dis2 = bond_length * 0.5
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f%10.8f%10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f%10.8f%10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f%10.8f%10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "3       //number of atoms\n"
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, dis1, dis2)
    return result

def tetramer(element, mass, fpseudo, lattice_constant, bond_length, nspin):
    """generate tetramer structure"""
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    dis1 = bond_length * 0.86603
    dis2 = bond_length * 0.5
    dis3 = bond_length * 0.81649
    dis4 = bond_length * 0.28867
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f%10.8f%10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f%10.8f%10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f%10.8f%10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "4       //number of atoms\n"
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(0.0, dis1, dis2)
    result += "%10.8f%10.8f%10.8f 0 0 0\n"%(dis3, dis4, dis2)
    return result

def STRU(shape: str = "", element: str = "", mass: float = 0.0, fpseudo: str = 0.0, 
         lattice_constant: float = 0.0, bond_length: float = 0.0, nspin: int = 1):
    """generate structure"""
    if shape == "":
        raise ValueError("shape is not specified")
    if element == "":
        raise ValueError("element is not specified")
    if fpseudo == "":
        raise ValueError("fpseudo is not specified")
    if lattice_constant == 0.0:
        raise ValueError("lattice_constant is not specified")
    if bond_length == 0.0:
        raise ValueError("bond_length is not specified")
    if shape == "dimer":
        return dimer(element, mass, fpseudo, lattice_constant, bond_length, nspin), 2
    elif shape == "trimer":
        return trimer(element, mass, fpseudo, lattice_constant, bond_length, nspin), 3
    elif shape == "tetramer":
        return tetramer(element, mass, fpseudo, lattice_constant, bond_length, nspin), 4
    else:
        raise NotImplementedError("Unknown shape %s"%shape)

def KPOINTS():
    """For ABACUS-orbitals numerical orbitals generation workflow specifically"""
    return "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"

def INPUT(user_settings: dict = {}) -> str:
    """generate INPUT file for orbital generation task"""
    inbuilt_template = {
        "suffix": "ABACUS", "stru_file": "STRU", "kpoint_file": "KPT", # wannier_card is deprecated
        "pseudo_dir": "./",
        "calculation": "scf", # calculation, definitely to be scf for orbital generation
        "basis_type": "pw", "ecutwfc": "100",
        "ks_solver": "dav", "nbands": "auto", "scf_thr": "1.0e-7", "scf_nmax": "9000", # scf control
        "ntype": "1", "nspin": "1", # system info
        "lmaxmax": "4", "bessel_nao_rcut": "10", # orbital generation control
        "smearing_method": "gauss", "smearing_sigma": "0.015", # for improving convergence
        "mixing_type": "broyden", "mixing_beta": "0.8", "mixing_ndim": "8", "mixing_gg0": "1", # mixing control
        "printe": "1" # print energy
    }
    result = "INPUT_PARAMETERS"
    for key in user_settings.keys():
        if key in inbuilt_template.keys():
            inbuilt_template[key] = user_settings[key]
        else:
            print("Warning: unknown key %s"%key)
    for key, value in inbuilt_template.items():
        result += "\n%-20s %s"%(key, value)
    return result

def write_string_tofile(input, filename):
    ifile = open(filename, 'w')
    ifile.write(input)
    ifile.flush()
    ifile.close()

# I guess a good habit of naming variable is important for big project
def _set_planewave_calculation_directory_(ielement: int, iecut: int, fSTRUs: list):
    global pwDataPath_STRU # temporarily do nothing on the name and use of global variable
    for fSTRU in fSTRUs:
        if(type_STRU[fSTRU] == "customer"):
            pwDataDir_STRU[fSTRU] = [os.path.basename(pwDataPath_STRU[fSTRU][iBL]) for iBL in range(nBL_STRU[fSTRU])]
            # yes the total list of STRU file depends on how many bond lengths should be tried on for one kind of reference structure.
            print(" Use customized PW WaveFunction Dir: ")
            for iBL in range(nBL_STRU[fSTRU]):
                print("%60s" % pwDataPath_STRU[fSTRU][iBL])
        else:
            pwDataDir_STRU[fSTRU] = [None]*nBL_STRU[fSTRU]
            print("Set default PW WaveFunction Dir for STRU:%s " % fSTRU)
            for iBL in range(nBL_STRU[fSTRU]):
                pwDataDir_STRU[fSTRU][iBL] = "-".join([element[ielement], fSTRU, BL_STRU[fSTRU][iBL]])
                pwDataPath_STRU[fSTRU][iBL] = "../OUT.%s" % pwDataDir_STRU[fSTRU][iBL]
                print("%60s" % pwDataPath_STRU[fSTRU][iBL])

def pw_calculation(iElement, iEcut, STRUList):
    global EXE_env
    global EXE_mpi
    global EXE_pw
    global pwDataDir_STRU
    global Pseudo_dir
    global nspin_STRU, maxL_STRU, nbands_STRU, Ecut, sigma
    # print("(EXE_mpi:%s)"%EXE_mpi)
    #iElement = 0
    #iEcut=0
    #iRcut=0
    for STRUname in STRUList:
      print( "\n %s "%( "-"*92) )
      print( "", (" Get PW WaveFunction for %s with %s Bond Lengths "%(STRUname, nBL_STRU[STRUname])).center(92,"-") )
      print( " %s "%( "-"*92) )

      if (type_STRU[STRUname] == "customer"):
        print( "\n Skip PW Calculation, use customized PW WaveFunction " )

      elif (type_STRU[STRUname] != "customer"):
        for iBL in range( nBL_STRU[STRUname] ):
            print( "\n", (" Do PW Calculation for Ecut(%s) & Bond-Length(%s)"%(
                    Ecut[iEcut], BL_STRU[STRUname][iBL])).center(92,"-") )

            (input_STRU, nAtoms) = STRU(shape=type_STRU[STRUname], 
                                        element=element[iElement], 
                                        mass=mass, 
                                        fpseudo=Pseudo_name[iElement], 
                                        lattice_constant=lat0, 
                                        bond_length=BL_STRU[STRUname][iBL], 
                                        nspin=nspin_STRU[STRUname])
            print( " %s = %-20s"%("nAtoms", nAtoms), end='\n')
            write_string_tofile(input_STRU, "%s.stru"%pwDataDir_STRU[STRUname][iBL] )

            input_KPOINTS = KPOINTS()
            write_string_tofile(input_KPOINTS, "KPOINTS")

            bessel_nao_rcut = " ".join(Rcut) # now ABACUS supports multiple Rcut
            input_INPUT = INPUT(user_settings={
                "suffix": pwDataDir_STRU[STRUname][iBL],
                "stru_file": pwDataDir_STRU[STRUname][iBL]+".stru",
                "pseudo_dir": Pseudo_dir,
                "ecutwfc": Ecut[iEcut],
                "basis_type": "pw",
                "ks_solver": "dav",
                "nspin": nspin_STRU[STRUname],
                "nbands": nbands_STRU[STRUname],
                "lmaxmax": maxL_STRU[STRUname],
                "bessel_nao_rcut": bessel_nao_rcut,
                "smearing_sigma": sigma
            })
            write_string_tofile(input_INPUT, "INPUT")

            PW_WF_Dir = "OUT.%s"%pwDataDir_STRU[STRUname][iBL]
            #PW_WF_file1 = "OUT.%s/orb_matrix.1.dat"%pwDataDir_STRU[STRUname][iBL]
            sys_run_str = '''
%s
echo " pwd: "`pwd`;
export OMP_NUM_THREADS=1
echo " OMP_NUM_THREADS:" $OMP_NUM_THREADS
PW_WF_Dir=%s
EXE_mpi="%s"
EXE_pw=%s
echo " run:  ${EXE_mpi} ${EXE_pw} "
echo " which ${EXE_pw}: "`which ${EXE_pw}`
#which mpirun mpiexec.hydra;
if [ ! -f "${PW_WF_Dir}/orb_matrix*" ]; then
    echo " Missing ${PW_WF_Dir}/orb_matrix*.dat, Calculating PW WF ... "
    stdbuf -oL ${EXE_mpi} ${EXE_pw};
else
    echo " Found ${PW_WF_Dir}/orb_matrix*.dat, Skip PW Calculation "
fi '''%(EXE_env, PW_WF_Dir, EXE_mpi, EXE_pw)
            # print("\n runcmd: %s \n"%sys_run_str )
            sys.stdout.flush() 

            # os.environ["PYTHONUNBUFFERED"] = "1"
            #process = subprocess.Popen(["your_cmd"]...)
            #process.wait()
            #with open('env_ff.txt', 'w') as ff:
            #    out = subprocess.run('env', shell=True, stdout=ff, text=True)
            #subprocess.run( [sys_run_str, "--login"], shell=True, text=True, stdin=subprocess.DEVNULL, timeout=7200) 

            subprocess.run( [sys_run_str, "--login"], shell=True, timeout=72000) 
            sys.stdout.flush() 

            #osvalue = os.system(sys_run_str) 
            #print(" osvalue = %s"%osvalue )
            #if osvalue != 0 : print(" fail to call "+sys_run_str) #continue #quit() 
            #sys.stdout.flush() 
            #break 


##################################  Prepare SIAB INPUT ##################################
def prepare_SIAB_INPUT(iEcut, iRcut, iLevel):
    iLevelm1 = iLevel-1
 
    STRUname = refSTRU_Level[iLevelm1]
    print(" Prepare INPUT for Level%s orbitals with Ref %s "%(iLevel, STRUname))

    INPUT_json = {"file_list":{}, "info":{}, "weight":{}, "C_init_info":{}, "V_info": {} }

    INPUT_json["file_list"] = {"origin":[], "linear":[] }
    INPUT_json["file_list"]["origin"] = [ pwDataPath_STRU[STRUname][iBL]+"/orb_matrix.0.dat" for iBL in range(nBL_STRU[STRUname]) ]
    INPUT_json["file_list"]["linear"] = [ [ pwDataPath_STRU[STRUname][iBL]+"/orb_matrix.1.dat" for iBL in range(nBL_STRU[STRUname]) ] ]

    INPUT_json["info"] = {"Nt_all": element, 
			"Nu":   { element[iElement]:orbConf_to_list(orbConf_Level[iLevelm1][iElement], Llabel, maxL_STRU[STRUname] ) 
			            for iElement in range(len(element) )  },
			"Rcut": { element[iElement]:Rcut[iRcut] for iElement in range(len(element)) },
			"dr":   { element[iElement]:0.01 for iElement in range(len(element)) },
			"Ecut": { element[iElement]:int(Ecut[iEcut]) for iElement in range(len(element)) }, 
            "lr": 0.03, 
            "cal_T": False,  "cal_smooth": True, "max_steps": max_steps } 
    
    if ( refBands_Level[iLevelm1] == "auto" ) :
        refBands_Level[iLevelm1] = [ pwDataPath_STRU[STRUname][iBL]+"/istate.info" for iBL in range(nBL_STRU[STRUname]) ]
        # refBandsFile_Level[iLevelm1] = [ pwDataPath_STRU[STRUname][iBL]+"/istate.info" for iBL in range(nBL_STRU[STRUname]) ]
        # INPUT_json["weight"] = { "stru": [1] * nBL_STRU[STRUname], 
        #                      "bands_file": refBandsFile_Level[iLevelm1] }
    #else:
    #    INPUT_json["weight"] = { "stru": [1] * nBL_STRU[STRUname], 
    #                         "bands_range": refBandsRange_Level[iLevelm1] }
    #print("____ refBands_Level:%s"%refBands_Level)

    if ( type(refBands_Level[iLevelm1][0]) == str ):
        INPUT_json["weight"] = { "stru": [1] * nBL_STRU[STRUname], 
                             "bands_file": refBands_Level[iLevelm1] }
    else:
        INPUT_json["weight"] = { "stru": [1] * nBL_STRU[STRUname], 
                             "bands_range": refBands_Level[iLevelm1] }

    INPUT_json["C_init_info"] = {}
    if ( fixPre_Level[iLevelm1] == "None" or fixPre_Level[iLevelm1] == "none" ):
        INPUT_json["C_init_info"]["init_from_file"] = False
    else:
        INPUT_json["C_init_info"]["init_from_file"] = True

        INPUT_json["C_init_info"]["C_init_file"] = "Level%s.ORBITAL_RESULTS.txt"%iLevelm1
        if(iLevelm1 == 0):
            print( " Prepare input-orbital (./Level0.ORBITAL_RESULTS.txt) for Level1 optimization ")
            if   (not os.path.isfile("./Level0.ORBITAL_RESULTS.txt")) and os.path.isfile("../../ORBITAL_RESULTS.txt") :
                subprocess.run( [ "cp -avp ../../ORBITAL_RESULTS.txt ./Level0.ORBITAL_RESULTS.txt", "--login"],
                                    shell=True, stdin=subprocess.DEVNULL, timeout=60 ) 
            elif (not os.path.isfile("./Level0.ORBITAL_RESULTS.txt")) and os.path.isfile("../../Level0.ORBITAL_RESULTS.txt") :
                subprocess.run( [ "cp -avp ../../Level0.ORBITAL_RESULTS.txt ./Level0.ORBITAL_RESULTS.txt", "--login"],
                                    shell=True, stdin=subprocess.DEVNULL, timeout=60 ) 

            # INPUT_json["C_init_info"]["C_init_file"] = "ORBITAL_RESULTS.txt"

        if ( fixPre_Level[iLevelm1] == "fix" or fixPre_Level[iLevelm1] == "Fix"  ):
            INPUT_json["C_init_info"]["opt_C_read"] = False
        else:
            INPUT_json["C_init_info"]["opt_C_read"] = True

    INPUT_json["V_info"] = {
        "init_from_file":True,
        "same_band":True
        }

    if ( 'opt_C_read' in INPUT_json["C_init_info"] ):
        if ( INPUT_json["C_init_info"]["opt_C_read"] == True  and  INPUT_json["C_init_info"]["init_from_file"] == True ):
            INPUT_json["info"]["lr"] = 0.001

    INPUT_json_str = json.dumps(INPUT_json, indent=2)
    #print(INPUT_json_str)

    ifile_input = open("INPUT", 'w')
    ifile_input.write(INPUT_json_str)
    ifile_input.flush()
    ifile_input.close()



def define_global_var(InputFile):
    
    ###################################  Setting Constants ###################################
    Hartree_to_eV=27.21138505
    periodtable = {   'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
                      'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
                   'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                   'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
                   'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
                   'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
                   'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                   'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
                   'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
                   'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
                    'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
                    'Er': 68, 'Tm': 69, 'Yb': 70, 
                    'Lu': 71, 
                   'Hf': 72, 'Ta': 73,
                   'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
                   'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 
                        ## 'Po': 84, #'At': 85,
                        ## 'Rn': 86, #'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
                        ## 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
                        ## 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
                        ## 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
                        ## 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113,
                        ## 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118
                   } 
    periodtable_r = {v: k for k, v in periodtable.items()}
    #print len(periodtable_r.keys() )
    #print periodtable_r[1]
    #print periodtable['Hg']
    Llabel=['s','p','d','f','g']
    lat0=20.0
    mass=1.0
    
    
    ###################################  Parse Arguments ###################################
    input_string=get_fileString(InputFile)
    #print(input_string)
    
    
    ###################################  Parse InputFile ###################################
    EXE_env     = get_string_linehead( "EXE_env", input_string )
    EXE_mpi     = get_string_linehead( "EXE_mpi", input_string )
    EXE_pw      = get_string_linehead( "EXE_pw", input_string )
    EXE_opt     = get_string_linehead( "EXE_opt", input_string )
    element     = get_array_linehead( "element", input_string )
    Ecut        = strs_to_ints(get_array_linehead( "Ecut", input_string ) )
    Rcut        = strs_to_ints(get_array_linehead( "Rcut", input_string ) )
    sigma       = float(get_string_linehead( "sigma", input_string ) )
    Pseudo_dir  = get_string_linehead( "Pseudo_dir", input_string )
    Pseudo_name = get_array_linehead( "Pseudo_name", input_string )
    max_steps   = int(get_string_linehead( "max_steps", input_string ) )
    
    element_num = [ periodtable[ii] for ii in element ]
    
    if EXE_opt == "":
        EXE_opt = path_thisfile+"/opt_orb_pytorch_dpsi/main.py (default)" 
        opt_mainFunc_path = path_thisfile+"/opt_orb_pytorch_dpsi" 
    else:
        opt_mainFunc_path=os.path.dirname(EXE_opt)
    sys.path.append( opt_mainFunc_path ) 

    print(" %20s = %s "%("EXE_env", EXE_env) )
    print(" %20s = %s "%("EXE_mpi", EXE_mpi) )
    print(" %20s = %s "%("EXE_pw",  EXE_pw) )
    print(" %20s = %s "%("EXE_opt", EXE_opt) )
    print(" %20s = %s "%("element", element) )
    print(" %20s = %s "%("element_num", element_num) )
    print(" %20s = %s "%("Ecut", Ecut) )
    print(" %20s = %s "%("Rcut", Rcut) )
    print(" %20s = %s "%("sigma", sigma) )
    print(" %20s = %s "%("Pseudo_dir", Pseudo_dir) )
    print(" %20s = %s "%("Pseudo_name", Pseudo_name) )
    print(" %20s = %s "%("max_steps", max_steps) )
    
    input={}
    
    nLevel = get_nRows_linehead( "Level", input_string )
    print("\n %20s = %s "%("nLevel", nLevel) )
    for iLevel in range(1,nLevel+1):
        input["Level%s"%iLevel] = get_array_linehead( "Level%s"%iLevel, input_string )
        #print(" %20s : %s"%( "Level%s"%iLevel, input["Level%s"%iLevel] ) )
    
    refSTRU_Level=[]
    fixPre_Level=[]
    maxL_Level=[]
    orbConf_Level=[]
    restartLevel=[]
    refBands_Level=[]
    
    for iLevel in range(1,nLevel+1):
        iLevelm1 = iLevel - 1
    
        #print( "\n Level%s: "%iLevel, end='\n ')
        print( " %-6s:"%("Level%s"%iLevel), end='\n')
    
        refSTRU_Level.append( input["Level%s"%iLevel][0] )
        print( " %20s = %s"%("Reference Struture", refSTRU_Level[iLevelm1] ), end='\n')
    
        if ( input["Level%s"%iLevel][1] == "auto" ):
            refBands = "auto"
        else:
            refBands = eval(input["Level%s"%iLevel][1])
        #refBands = int(float(refBands))
        refBands_Level.append( refBands )
        str_to_print = "%s %s"%(refBands_Level[iLevelm1], type(refBands) )
        print( " %20s = %s"%("Reference Bands", str_to_print), end='\n')
    
        fixPre_Level.append( input["Level%s"%iLevel][2] )
        print( " %20s : %s"%("Fix input orbitals?", fixPre_Level[iLevelm1]), end='\n')
    
        if iLevel == 1 :
            restartLevel.append(False)
        else:
            restartLevel.append(True)
        print( " %20s : %s"%("Restart Level?", restartLevel[iLevelm1]), end='\n') 
    
        orbConf_Level.append( input["Level%s"%iLevel][3:] )
        # print( " %20s = %s"%("Orbital Conf", orbConf_Level[iLevelm1]), end='\n') 
        for iElement in range(len(element) ):
            print( " %20s : %s / %s"%( "Orbitals for %2s"%element[iElement] , \
                        orbConf_Level[iLevelm1][iElement], \
                        orbConf_to_list(orbConf_Level[iLevelm1][iElement], Llabel) \
                        ), end='\n' ) 
    
        print( " ", end='\n' ) 
    # orbConfList_Level = copy.deepcopy(orbConf_Level)
    # for iLevel in range(1,nLevel+1):
    #     iLevelm1 = iLevel - 1 
    #     for iElement in range(len(element) ):
    #         orbConfList_Level[iLevelm1][iElement] = orbConf_to_list(orbConf_Level[iLevelm1][iElement].ljust(10), Llabel)
    #         print( " %20s : %s"%( "Orbitals for %2s"%element[iElement] , orbConfList_Level[iLevelm1][iElement] ), end='\n' ) 
    
    
    STRUList = list(dict.fromkeys(refSTRU_Level))
    nSTRU = len(STRUList)
    #nSTRU = get_nRows_linehead( "STRU", input_string )
    print("\n Parse %s types of structures: %s"%(nSTRU, STRUList) )
    
    type_STRU={}
    nbands_STRU={}
    maxL_STRU={}
    nspin_STRU={}
    BL_STRU={}
    nBL_STRU={}
    pwDataPath_STRU={}
    
    for STRUname in STRUList:
        input[STRUname] = get_array_linehead( STRUname, input_string )
        #print(" %20s : %s"%( STRUname, input[STRUname] ) )
        print( " %s:"%(STRUname), end='\n')
    
        type_STRU[STRUname] = input[STRUname][0]
        print( " %20s = %s"%("STRU Type", type_STRU[STRUname]), end='\n')
    
        if (type_STRU[STRUname] != "customer"):
            nbands_STRU[STRUname] = int(float(input[STRUname][1]))
            print( " %20s = %s"%("nbands", nbands_STRU[STRUname]), end='\n')
    
            maxL_STRU[STRUname] = int(input[STRUname][2])
            print( " %20s = %s"%("maxL", maxL_STRU[STRUname] ), end='\n')
    
            nspin_STRU[STRUname] = int(input[STRUname][3])
            print( " %20s = %s"%("nspin", nspin_STRU[STRUname]), end='\n')
    
            BL_STRU[STRUname] = strs_to_floats(input[STRUname][4:]) 
            print( " %20s = %s"%("Bond Length List", BL_STRU[STRUname]), end='\n')
    
            nBL_STRU[STRUname] = len(BL_STRU[STRUname])
            #print(  " %20s = %s"%("BL List Size", nBL_STRU[STRUname]), end='\n')
    
            pwDataPath_STRU[STRUname] = ["None" for ii in range(nBL_STRU[STRUname]) ]
        else:
            pwDataPath_STRU[STRUname] = input[STRUname][1:]
            print(  " %20s = %s"%("WF Data Path", pwDataPath_STRU[STRUname]), end='\n')
    
            nBL_STRU[STRUname] = len(pwDataPath_STRU[STRUname])
        
        print(  " %20s = %s"%("BL List Size", nBL_STRU[STRUname]), end='\n')
    
    nSave = get_nRows_linehead( "Save", input_string )
    if (nSave > 0):
        print("\n %20s : %s "%("Save Orbital?", True if nSave>0 else False) )
        print(  " %20s = %s "%("nSave", nSave) )
        for ii in range(1,nSave+1):
            input["Save%s"%ii] = get_array_linehead( "Save%s"%ii, input_string )
            print(" %12s %s as %s"%( "Save", input["Save%s"%ii][0] , input["Save%s"%ii][1]) )
    print( " ", end='\n' ) 
    
    
    ##################################  Derived parameter  ##################################
    nRcut = len(Rcut)
    
    #refBandsRange_Level=[]
    #refBandsFile_Level=[]
    for iLevel in range(1,nLevel+1):
        iLevelm1 = iLevel - 1
        STRUname = refSTRU_Level[iLevelm1]
        #print( " type(refBands_Level[iLevelm1] : %s"%type(refBands_Level[iLevelm1] ) )
    
        if ( type(refBands_Level[iLevelm1]) == list ):
            refBands_Level[iLevelm1] = int(float( refBands_Level[iLevelm1]) )
            # refBandsRange_Level.append( int(float( refBands_Level[iLevelm1]) ) )
        #elif ( refBands_Level[iLevelm1] == "auto" ) :
        #    refBands_Level[iLevelm1] = [ "istate.info" ] * nBL_STRU[STRUname]
        #    # refBandsFile_Level.append( [ "istate.info" ] * nBL_STRU[STRUname] )
        elif ( type(refBands_Level[iLevelm1]) == int ):
            refBands_Level[iLevelm1] = [ refBands_Level[iLevelm1] for iBL in range(nBL_STRU[STRUname]) ]
        #else:
        #    refBands_Level[iLevelm1] = [ "istate.info"            for iBL in range(nBL_STRU[STRUname]) ]
        #    # refBandsRange_Level.append( [ refBands_Level[iLevelm1] ] * nBL_STRU[STRUname] )
    # print( " refBandsRange_Level: %s \n refBandsFile_Level: %s"%(refBandsRange_Level, refBandsFile_Level) )
    print( " refBands_Level: %s "%(refBands_Level) )


    ElementDir = os.getcwd()
    #print(" Current working directory %s "%ElementDir )
    globals().update(locals())
    
    
def SaveOrb(ElementDir, SIAB_wdir, iRcut):
        ##################################  Save Orbitals  #################################
        print( "\n %s "%( "-"*92) )
        print( "", (" Save Orbitals ").center(92,"-") )
        print( " %s "%( "-"*92) )
    
        print("\n Current working directory %s "%os.getcwd() )
        # Now change the directory
        os.chdir(ElementDir)
        # Check current working directory.
        print(" Directory changed to %s "%os.getcwd() )
    
        if (nSave > 0):
            for ii in range(1,nSave+1):
                Leveln = input["Save%s"%ii][0]
                Leveln = int(Leveln[5:])
                resultPathPrefix = SIAB_wdir+"/"+str(Rcut[iRcut])+"/Level"+str(Leveln) 
                orbName = element[iElement]+"_gga_"+str(Rcut[iRcut])+"au_"+str(Ecut[iEcut])+"Ry_"+orbConf_Level[Leveln-1][iElement]
                print( "\n resultPathPrefix: %s"%resultPathPrefix )
                print( " orbName: %s"%orbName )
    
                # todo: check the orbConf_Level in orbital files
                # Number of Sorbital-->       3
                # Number of Porbital-->       3
                # Number of Dorbital-->       2
    
                orbType = input["Save%s"%ii][1]
                orbSaveDir="Orbital_%s_%s"%(element[iElement], orbType)
                try:
                    os.mkdir(orbSaveDir)
                except OSError as error:
                    print(" Already has directory: %s"%( orbType ) ) 
                print(" Save Level%s results to dir: %s"%(str(Leveln), orbSaveDir) )
    
                orbSaveRPath   = orbSaveDir+"/info/"+str(Rcut[iRcut])
                try:
                    os.makedirs(orbSaveRPath)
                except OSError as error:
                    print(" Already has directory: %s"%( orbSaveRPath ) ) 
    
                sys_run_str = '''
                InputFile=%s
                resultPathPrefix=%s
                orbSaveRPath=%s
                element=%s
                orbSaveDir=%s
                orbName=%s
                cp -avp ${resultPathPrefix}.ORBITAL_${element}U.dat     ${orbSaveDir}/${orbName}.orb
                cp -avp ${InputFile}                                    ${orbSaveRPath}/SIAB_INPUT
                cp -avp ${resultPathPrefix}.ORBITAL_${element}U.dat     ${orbSaveRPath}/ORBITAL_${element}U.dat
                cp -avp ${resultPathPrefix}.INPUT                       ${orbSaveRPath}/INPUT
                cp -avp ${resultPathPrefix}.ORBITAL_PLOTU.dat           ${orbSaveRPath}/ORBITAL_PLOTU.dat
                cp -avp ${resultPathPrefix}.ORBITAL_RESULTS.txt         ${orbSaveRPath}/ORBITAL_RESULTS.txt
                cp -avp ${resultPathPrefix}.Spillage.dat                ${orbSaveRPath}/Spillage.dat
                '''%( InputFile, resultPathPrefix, orbSaveRPath, element_num[iElement], orbSaveDir, orbName ) 
                subprocess.run( [ sys_run_str, "--login"], shell=True, stdin=subprocess.DEVNULL, timeout=60) 
    
            print("\n Saved %s Orbitals"%(nSave) )
        print( " ", end='\n' ) 


if __name__=="__main__":
    #######################################  print logo  #####################################
    Version = "0.9"
    print(" Starting SIAB Version %s @ %s\n"%(Version, time.ctime(time.time()))  )
    print( " %s "%('*'*92) )
    
    print( " *", ("  ___   ___     _     ___  ").center(88), "*" ) 
    print( " *", (" / __| [_ _]   /_\   | _ ) ").center(88), "*" ) 
    print( " *", (" \__ \  | |   / _ \  | _ \ ").center(88), "*" ) 
    print( " *", (" |___/ [___] /_/ \_\ |___/ ").center(88), "*" ) 
    print( " *", (" ").center(88), "*" ) 
    print( " *", ("Systematically Improvable Atomic-orbital Basis (SIAB) generator").center(88), "*" ) 
    print( " *", ("for Linear Combination of Atomic Orbitals (LCAO)").center(88), "*" ) 
    print( " *", (" ").center(88), "*" ) 

    print( " %s "%('*'*92) )
    #quit()
    
    args=parse_arguments()
    print("\n %20s = %s "%("InputFile",args.InputFile))
    #print(" HostList: %s "%args.HostList)
    define_global_var(args.InputFile)
    
    print(" import mainFunc @ path: ", opt_mainFunc_path )
    #print("[pyTorch Version: "+torch.__version__+"]" , flush=True )
    import opt_orb_pytorch_dpsi.main as mainFunc

    
    ##################################  Do    Calculation ##################################
    iElement=0
    iEcut=0
    os.chdir(ElementDir)
    print("\n Current working directory %s "%ElementDir )
    # Now change the directory
    SIAB_wdir = "%s_%s_%sRy"%(element_num[iElement], element[iElement], Ecut[iEcut])
    SIAB_fullpath= ElementDir+"/"+SIAB_wdir
    os.makedirs(SIAB_fullpath, exist_ok=True) # this will create a dir named as 68_Er_100Ry or something, but why it does not work actually???

    os.chdir(SIAB_fullpath) # check-in the workdir
    print(" Directory changed to %s "%os.getcwd() )

    ################################  Do PW Calculation ################################
    # get pwDataDir_STRU[STRUname][iBL] pwDataPath_STRU[STRUname][iBL]
    pwDataDir_STRU = {}
    _set_planewave_calculation_directory_(iElement, iEcut, STRUList)

    #subprocess.run
    pw_calculation(iElement, iEcut, STRUList) # refactored
    
    #quit()

    print("\n Current working directory %s "%os.getcwd() )
    # SIAB_rcutdir = SIAB_fullpath+"/%s"%(Rcut[iRcut])
    # try:
    #     os.mkdir(SIAB_rcutdir)
    # except OSError as error:
    #     print(" Already has directory: %s"%SIAB_rcutdir )    
    # os.chdir(SIAB_rcutdir)
    # # Check current working directory.
    # print(" Directory changed to rcut dir: %s "%os.getcwd() )

    ################################  Do SIAB Calculation ###############################
    for iRcut in range(nRcut):
        SIAB_rcutdir = SIAB_fullpath+"/%s"%(Rcut[iRcut])
        os.makedirs(SIAB_rcutdir, exist_ok=True)
        os.chdir(SIAB_rcutdir)
        # Check current working directory.
        print(" Directory changed to rcut dir: %s "%os.getcwd() )
        for iLevel in range(1,nLevel+1):
            Leveln = "Level"+str(iLevel)
            iLevelm1 = iLevel - 1 

            print( "\n %s "%( "-"*92) )
            print( "", (" Generate the Level %s Orbitals"%(iLevel)).center(92,"-") )
            print( " %s "%( "-"*92) )
            print("\n Current working directory %s "%os.getcwd() )

            genOrbFile = "%s.ORBITAL_%sU.dat"%(Leveln,element_num[iElement])
            if os.path.isfile(genOrbFile) :
                print(" Found file:%s, skip this level"%(genOrbFile) ) 
                continue 
            else:
                prepare_SIAB_INPUT(iEcut, iRcut, iLevel)

            sys_run_str = '''
            #conda init bash
            
            %s
            nproc_pw=`%s hostname| wc -l`
            export OMP_NUM_THREADS=$nproc_pw
            echo " OMP_NUM_THREADS:" $OMP_NUM_THREADS
            
            #module purge
            #module load anaconda3_nompi
            #module load gcc/9.2.0
            #module list 2>&1;
            #pwd;
            #source activate pytorch110 
            #conda activate pytorch110 
            #conda info --envs
            echo python: `which python3`
            
            #python3 -c "import torch; print( '[pyTorch Version: '+torch.__version__+']' , flush=True )" 2>&1
            python3 %s
            
            #conda deactivate
            '''%(EXE_env, EXE_mpi, EXE_opt)
            #print(" runcmd: \n%s \n"%sys_run_str )
            #subprocess.run( [ sys_run_str, "--login"], shell=True, text=True, stdin=subprocess.DEVNULL, timeout=18000) 
        
            ##execfile(EXE_opt)

            # continue 
            # print(" import mainFunc @ path: ", opt_mainFunc_path )
            # #print("[pyTorch Version: "+torch.__version__+"]" , flush=True )
            # import main as mainFunc
            mainFunc.main() #!!!
            

            sys_run_str = '''
            mv INPUT                %s.INPUT
            mv ORBITAL_%sU.dat      %s.ORBITAL_%sU.dat
            mv ORBITAL_PLOTU.dat    %s.ORBITAL_PLOTU.dat
            mv ORBITAL_RESULTS.txt  %s.ORBITAL_RESULTS.txt
            mv Spillage.dat         %s.Spillage.dat
            '''%( Leveln, element_num[iElement],  Leveln,element_num[iElement],  Leveln,   Leveln,  Leveln ) 
            subprocess.run( [ sys_run_str, "--login"], shell=True, stdin=subprocess.DEVNULL, timeout=60)  #!!!

            sys.stdout.flush() 
            #quit()

        SaveOrb(ElementDir, SIAB_wdir, iRcut)

print(" Finished SIAB @ %s\n"%time.ctime(time.time() ))
