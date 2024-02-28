import SIAB.spillage.pytorch_swat.util as util
import torch
import itertools
import numpy as np
import re
import copy

def read_file_head(info,file_list):
    """ QI[ist][it][il][ib*ia*im,ie]    <\psi|jY> """
    """ SI[ist][it1][it2][il1][il2][ie1,ia1,im1,ia2,im2,ie2]    <jY|jY> """
    """ VI[ist][ib]        <\psi|\psi> """
    info_true = copy.deepcopy(info)
    info_true.Nst = len(file_list)
    info_true.Nt = util.ND_list(info_true.Nst,element="list()")
    info_true.Na = util.ND_list(info_true.Nst,element="dict()")
    info_true.Nb = util.ND_list(info_true.Nst)
    info_true.Nk = util.ND_list(info_true.Nst)
    info_true.Ne = dict()

    for ist_true,file_name in enumerate(file_list):
        print("Read file:", file_name)
        with open(file_name,"r") as file:

            util.ignore_line(file,4)
            Nt_tmp = int(file.readline().split()[0])
            for it in range(Nt_tmp):
                t_tmp = file.readline().split()[0]
                assert t_tmp in info.Nt_all
                info_true.Nt[ist_true].append( t_tmp )
                info_true.Na[ist_true][t_tmp] = int(file.readline().split()[0])
                util.ignore_line( file, info_true.Na[ist_true][t_tmp] )
            util.ignore_line(file,6)
            Nl_ist = int(file.readline().split()[0])+1
            for it,Nl_C in info.Nl.items():
                print(f"atom symbol: {it}\nnumber of l for present structure: {Nl_ist}\nnumber of l for present coefficients: {Nl_C}")
                assert Nl_ist >= Nl_C
                info_true.Nl[it] = Nl_ist
            info_true.Nk[ist_true] = int(file.readline().split()[0])
            info_true.Nb[ist_true] = int(file.readline().split()[0])
            util.ignore_line(file,1)
            #Ne_tmp = list(map(int,file.readline().split()[:Nt_tmp]))
            #for it,Ne in zip(info_true.Nt[ist_true],Ne_tmp):
            #    assert info_true.Ne.setdefault(it,Ne)==Ne
            Ne_tmp = int(file.readline().split()[0])
            for it in info_true.Nt[ist_true]:
                info_true.Ne[it] = Ne_tmp

    info_all = copy.deepcopy(info)
    info_all.Nst = sum(info_true.Nk,0)
    repeat_Nk = lambda x: list( itertools.chain.from_iterable( map( lambda x:itertools.repeat(*x), zip(x,info_true.Nk) ) ) )
    info_all.Nt = repeat_Nk(info_true.Nt)
    info_all.Na = repeat_Nk(info_true.Na)
    info_all.Nb = repeat_Nk(info_true.Nb)
    info_all.Ne = info_true.Ne

    return info_all


def read_QSV(stru_objs, element, fmatrices: list, V_info):
    """read <OVERLAP_Q>, <OVERLAP_Sq> and <OVERLAP_V> matrices output from ABACUS calculation
    
    Args:
        stru_objs: list of Structure objects, for each, has Nb, Nl, Na, Nk, Ne, Nt... attributes
        element: str, element name
        fnames: list of str, file names of ABACUS output
        V_info: dict, information of V matrix
    
    Returns:
        three overlap matrices, Q, S and V
    """
    print("Reading <OVERLAP_Q>, <OVERLAP_Sq> and <OVERLAP_V> from ABACUS output...", flush=True)
    ovlp_Q, ovlp_Sq, ovlp_V = [], [], []
    istru = 0 # this is actually a flattened 2D index, row index is the structure index, column index is the k-point index
    for ifm, fmatrix in enumerate(fmatrices):
        with open(fmatrix, "r") as file:
            nkpts = int(re.compile(r"(\d+)\s+nks").search(file.read()).group(1))
        with open(fmatrix, "r") as file:
            data = re.compile(r"<OVERLAP_Q>(.+)</OVERLAP_Q>", re.S).search(file.read())
            if not data:
                raise ValueError(f"BROKEN FILE: No <OVERLAP_Q> found in file: {fmatrix}. Seems ABACUS calculation is not finished properly.")
            data = map(float, data.group(1).split())
            print("Read <OVERLAP_Q> from ABACUS output.\n%20s%20s"%("Structure index", "K-point index"), flush=True)
            for ikpt in range(nkpts):
                print("%20d%20d"%(ifm, ikpt), flush=True)
                qi = read_QI(stru_objs[istru+ikpt], element, data)
                ovlp_Q.append(qi)
        with open(fmatrix, "r") as file:
            data = re.compile(r"<OVERLAP_Sq>(.+)</OVERLAP_Sq>", re.S).search(file.read())
            if not data:
                raise ValueError(f"BROKEN FILE: No <OVERLAP_Sq> found in file: {fmatrix}. Seems ABACUS calculation is not finished properly.")
            data = map(float, data.group(1).split())
            print("Read <OVERLAP_Sq> from ABACUS output.\n%20s%20s"%("Structure index", "K-point index"), flush=True)
            for ikpt in range(nkpts):
                print("%20d%20d"%(ifm, ikpt), flush=True)
                si = read_SI(stru_objs[istru+ikpt], element, data)
                ovlp_Sq.append(si)
        if V_info["init_from_file"]:
            with open(fmatrix, "r") as file:
                data = re.compile(r"<OVERLAP_V>(.+)</OVERLAP_V>", re.S).search(file.read())
                if not data:
                    raise ValueError(f"BROKEN FILE: No <OVERLAP_V> found in file: {fmatrix}. Seems ABACUS calculation is not finished properly.")
                data = map(float, data.group(1).split())
        else:
            data = ()
        print("Read <OVERLAP_V> from ABACUS output.\n%20s%20s"%("Structure index", "K-point index"), flush=True)
        for ikpt in range(nkpts):
            print("%20d%20d"%(ifm, ikpt), flush=True)
            vi = read_VI(stru_objs[istru+ikpt], V_info, ifm, data)
            ovlp_V.append(vi)
        istru += nkpts
    return ovlp_Q, ovlp_Sq, ovlp_V


def read_QI(info_stru, info_element, data):
    """ QI[it][il][ib*ia*im,ie]    <\psi|jY> """
    QI = dict()
    for it in info_stru.Na.keys():
        QI[it] = util.ND_list(info_element[it].Nl)
        for il in range(info_element[it].Nl):
            QI[it][il] = torch.zeros((info_stru.Nb, info_stru.Na[it], util.Nm(il), info_element[it].Ne), dtype=torch.complex128)
    for ib in range(info_stru.Nb):
        for it in info_stru.Na.keys():
            for ia in range(info_stru.Na[it]):
                for il in range(info_element[it].Nl):
                    for im in range(util.Nm(il)):
                        for ie in range(info_element[it].Ne):
                            QI[it][il][ib,ia,im,ie] = complex(next(data), next(data))
    for it in info_stru.Na.keys():
        for il in range(info_element[it].Nl):
            QI[it][il] = QI[it][il][:info_stru.Nb_true,:,:,:].view(-1,info_element[it].Ne).conj()
    return QI


def read_SI(info_stru, info_element, data):
    """ SI[it1,it2][il1][il2][ie1,ia1,im1,ia2,im2,ie2]    <jY|jY> """
    SI = dict()
    for it1,it2 in itertools.product( info_stru.Na.keys(), info_stru.Na.keys() ):
        SI[it1,it2] = util.ND_list(info_element[it1].Nl, info_element[it2].Nl)
        for il1,il2 in itertools.product( range(info_element[it1].Nl), range(info_element[it2].Nl) ):
            SI[it1,it2][il1][il2] = torch.zeros((info_stru.Na[it1], util.Nm(il1), info_element[it1].Ne, info_stru.Na[it2], util.Nm(il2), info_element[it2].Ne), dtype=torch.complex128)
    for it1 in info_stru.Na.keys():
        for ia1 in range(info_stru.Na[it1]):
            for il1 in range(info_element[it1].Nl):
                for im1 in range(util.Nm(il1)):
                    for it2 in info_stru.Na.keys():
                        for ia2 in range(info_stru.Na[it2]):
                            for il2 in range(info_element[it2].Nl):
                                for im2 in range(util.Nm(il2)):
                                    for ie1 in range(info_element[it1].Ne):
                                        for ie2 in range(info_element[it2].Ne):
                                            SI[it1,it2][il1][il2][ia1,im1,ie1,ia2,im2,ie2] = complex(next(data), next(data))
#    for it1,it2 in itertools.product( info.Nt[ist], info.Nt[ist] ):
#        for il1,il2 in itertools.product( range(info.Nl[it1]), range(info.Nl[it2]) ):    
#            SI[it1,it2][il1][il2] = torch_complex.ComplexTensor(
#                torch.from_numpy(SI[it1,it2][il1][il2].real),
#                torch.from_numpy(SI[it1,it2][il1][il2].imag))
    return SI



def read_VI(info_stru,V_info,ist,data):
    if V_info["same_band"]:
        """ VI[ib]    <psi|psi> """
        if V_info["init_from_file"]:
            VI = np.empty(info_stru.Nb,dtype=np.float64)
            for ib in range(info_stru.Nb):
                VI.data[ib] = next(data)
            VI = VI[:info_stru.Nb_true]
        else:
            VI = np.ones(info_stru.Nb_true, dtype=np.float64)
    else:
        """ VI[ib1,ib2]    <psi|psi> """
        if V_info["init_from_file"]:
            VI = np.empty((info_stru.Nb,info_stru.Nb),dtype=np.float64)
            for ib1,ib2 in itertools.product( range(info_stru.Nb), range(info_stru.Nb) ):
                VI[ib1,ib2] = next(data)
            VI = VI[info_stru.Nb_true, info_stru.Nb_true]
        else:
            VI = np.eye(info_stru.Nb_true, info_stru.Nb_true, dtype=np.float64)
    return torch.from_numpy(VI)
