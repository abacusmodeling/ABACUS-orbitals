#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# present packages
import SIAB.spillage.pytorch_swat.IO.read_QSV as sspsirqsv
import SIAB.spillage.pytorch_swat.IO.func_C as sspsifc
import SIAB.spillage.pytorch_swat.IO.read_json as sspsirj
import SIAB.spillage.pytorch_swat.IO.print_orbital as sspsipo
import SIAB.spillage.pytorch_swat.IO.change_info as sspsicinfo
import SIAB.spillage.pytorch_swat.IO.cal_weight as sspsicw
from SIAB.spillage.pytorch_swat.opt_orbital import Opt_Orbital
from SIAB.spillage.pytorch_swat.opt_orbital_wavefunc import Opt_Orbital_Wavefunc
import SIAB.spillage.pytorch_swat.orbital as sspso
import SIAB.spillage.pytorch_swat.util as sspsu
# released/official packages
import torch
import torch_optimizer 
#import pprint
import SIAB.spillage.pytorch_swat.IO.stdout as sspsistdout
import numpy as np
import time
# to avoid data racing
import uuid

def main(params: dict = None):
    """not-highly abstracted main function, as workflow function of spillage optimiaztion task"""
    coef_deriv0, coef_deriv1 = params.get("spill_coefs", [2, 1])
    spill_thr = params.get("spill_thr", 1e-8)
    print("""
--------------------------------------------------
Module Spillage - find the most similar space to the target spanned planewave wavefunction:
SIAB.pytorch_swat starts, numerical atomic orbitals are optimized.
--------------------------------------------------
""", flush=True)
    ###################################
    #   RANDOM SEED INITIALIZATION    #
    ###################################
    seed = int(1000*time.time())%(2**32)
    np.random.seed(seed)
    print("SEED INITIALIZATION: due to optimization method is local, random seed is somehow preferred. Present seed:", seed)
    ###################################
    #              TIMER              #
    ###################################
    time_start = time.time()
    ###################################
    #       INPUT FILE READING        #
    ###################################
    if params is None:
        file_list, info_true, weight_info, C_init_info, V_info = sspsirj.read_json("INPUT")
    else:
        file_list = params["file_list"]
        weight_info = params["weight"]
        C_init_info = params["C_init_info"]
        V_info = params["V_info"]
        info_true = sspsirj.Info()
        for key, value in params["info"].items():
            info_true.__dict__[key] = value
        info_true.Nl = {it:len(Nu) for it, Nu in info_true.Nu.items()}
        print("WORKFLOW: use on-the-fly information pass from front-end to back-end.", flush=True)
    ###################################
    # DATA STRUCTURE OF WEIGHT MATRIX #
    ###################################
    # weight is the one has dimension [nSTRU*nkpt][nband], is calculated by multiplication of two weight data matrix,
    # the first is STRU_weight, has dimension [nSTRU][nband], because the band itself doesnot have weight, the weight is from STRU
    # the second is kpt_weight, has dimension [nSTRU][nkpt], for each STRU, there will be a kpoint list and kpt itself has weight
    # every element of weight is calculated by multiplying the corresponding element of STRU_weight and kpt_weight,
    # get physical meaning of the weight per STRU per (k, b)-pair.
    # More exactly, the weight should have the structure like:
    #                band1    band2    band3    ...
    # STRU1-kpt1    w11        w12        w13        ...
    # STRU1-kpt2    ...
    # ...            
    # STRU1-kptN    ...
    # STRU2-kpt1    ...
    # STRU2-kpt2    ...
    # ...
    # STRU2-kptN    ...
    # ...
    ###################################
    #    READ WEIGHT INFO OF STATES	  #
	###################################
    weight = sspsicw.cal_weight(weight_info, V_info["same_band"], file_list["origin"])
	###################################
	#    READ INFO FROM INPUT FILE    #
	###################################
    info_kst = sspsirqsv.read_file_head(info_true, file_list["origin"])
	###################################
	#    CHANGE INFO FROM INPUT FILE  #
	###################################
    info_stru, info_element, info_opt = sspsicinfo.change_info(info_kst, weight)
    ###################################
    #    READ MAX INFO FROM FILES     #
    ###################################
    info_max = sspsicinfo.get_info_max(info_stru, info_element)
	###################################
    #   PRINT INFO (NOT LOOKS GOOD)   #
	###################################
    print("-"*80, flush=True)
    print("INFORMATION CHECK - Please check every detail of the information below:", flush=True)
    print("-"*80, flush=True)
    print(sspsistdout.bundle_print(info_kst=info_kst, info_stru=info_stru, info_element=info_element,
                                   info_opt=info_opt, info_max=info_max, spill_coefs=[coef_deriv0, coef_deriv1]),
                                   flush=True)
    print("-"*80, flush=True)
    ###################################
    # PHYSICAL MEANING OF Q, S, V     #
    ###################################
    # Q is the overlap matrix between psi and jY, or that of the gradient of psi and jY
    # Q = <psi|jY> or <grad(psi)|grad(jY)>
    # S is the overlap matrix between jY and jY, or that of the gradient of jY and jY
    # S = <jY|jY> or <grad(jY)|grad(jY)>
    # V is the overlap matrix between psi and psi, or that of the gradient of psi and psi
    # V = <psi|psi> or <grad(psi)|grad(psi)>
	###################################
    #    READ Q, S, V FROM FILES      #
    ###################################
    QI, SI, VI_origin = sspsirqsv.read_QSV(info_stru, info_element, file_list["origin"], V_info)
    if "linear" in file_list.keys():
        QI_linear, SI_linear, VI_linear = list(zip(*(sspsirqsv.read_QSV(info_stru, 
                                                                        info_element, 
                                                                        file, 
                                                                        V_info) for file in file_list["linear"])))
	###################################
    # INITIALIZE COEFFICIENTS OF ORB  #
	###################################
    # C is intialized here! carefully treat it!
    if C_init_info["init_from_file"]:
        C, C_read_index = sspsifc.read_C_init(C_init_info["C_init_file"], info_element)
    else:
        C = sspsifc.random_C_init(info_element)
    E = sspso.set_E(info_element)
    sspso.normalize(orb=sspso.generate_orbital(info_element, C, E),
                    dr={it:info_element[it].dr for it in info_element},
                    C=C, 
                    flag_norm_C=True)
	###################################
	#    OPTIMIZATION OF ORBITALS     #
	###################################
    #orb_optimizer = torch.optim.Adam(sum(([c.real, c.imag] for c in sum(C,[])), []), lr=info_opt.lr, eps=1e-8)
    #orb_optimizer = torch.optim.Adam(sum(C.values(),[]), lr=info_opt.lr, eps=1e-20, weight_decay=info_opt.weight_decay)
    #orb_optimizer = radam.RAdam(sum(C.values(),[]), lr=info_opt.lr, eps=1e-20)

    # define some files to store temporary data, with uuid3 namespace_dns as the unique identifier, uuid4 to generate
    # random component
    fspill = f"{uuid.uuid3(uuid.NAMESPACE_DNS, f'Spillage-{uuid.uuid4()}').hex}.dat"
    fcoef = f"{uuid.uuid3(uuid.NAMESPACE_DNS, f'ORBITAL_RESULTS-{uuid.uuid4()}').hex}.txt"
    fplotu = f"{uuid.uuid3(uuid.NAMESPACE_DNS, f'ORBITAL_PLOTU-{uuid.uuid4()}').hex}.dat"
    fu = f"{uuid.uuid3(uuid.NAMESPACE_DNS, f'ORBITAL_U-{uuid.uuid4()}').hex}.dat"

    print(f"""
Optimization of the orbital starts.
torch_optimizer.SWATS (Improving Generalization Performance by Switching from Adam to SGD) optimizer is used.
Parameters are listed below
Learning rate: {info_opt.lr}
Epsilon: {1e-20}
Max steps: {info_opt.max_steps}
""", flush=True)

    orb_optimizer = torch_optimizer.SWATS(sum(C.values(),[]), lr=info_opt.lr, eps=1e-20)
    with open(fspill, "w") as S_file:
        
        print("Optimization on Spillage function starts, check \"Spillage.dat\" for detailed trajectory.", flush=True)
        # use f-string to format the output
        ###################################
        #     ITERATION TABLE HEADER      #
        ###################################
        if info_opt.cal_T:
            print("-"*80, flush=True)
            print(f"{'Step':>10}{'Spillage':>20}{'T-term':>20}{'Loss function':>20}{'Time':>10}", flush=True)
            print("-"*80, flush=True)
        else:
            print("-"*60, flush=True)
            print(f"{'Step':>10}{'Spillage':>20}{'deltaSpill':>20}{'Time':>10}", flush=True)
            print("-"*60, flush=True)
        ###################################
        # INITIALIZE OVER-LOOP VARIABLES  #
        ###################################
        # initialize the loss_old to be infinity, so that the optimization will start
        loss_old = np.inf
        # arbitrarily set the max step to 30000 and if input defines it, use the input value
        maxSteps = 30000
        if isinstance(info_opt.max_steps, int):
            if info_opt.max_steps > 0:
                maxSteps = info_opt.max_steps
        # spillage value of the previous step, initialize as 0.
        _spill_0 = 0
        
		###################################
		#    OPTIMIZATION LOOP STARTS     #
		###################################
        # optimization loop starts here
        for istep in range(maxSteps):
            # record the time of each step
            time_stepstart = time.time()
            # --------------------------------
            # START: hack function here to change definition of Spillage
            Spillage = 0
            # for each structure...
            for ist in range(len(info_stru)):
                # initialize a new Opt_Orbital_Wavefunc object for STRU
                opt_orb_wave = Opt_Orbital_Wavefunc(info_stru[ist], info_element, V_info)
                # 
                V_origin = opt_orb_wave.cal_V_origin(C, QI[ist], SI[ist])
                # linear corresponds to the mode of calculating the derivative of the wavefunction
                if "linear" in file_list.keys():
                    V_linear = [ opt_orb_wave.cal_V_linear(C, QI_linear[i][ist], SI_linear[i][ist])
                        for i in range(len(file_list["linear"]))]
                # why defines the spillage function here and not outside the loop?
                def cal_Spillage(V_delta):
                    Spillage = (V_delta * weight[ist][:info_stru[ist].Nb_true]).sum()
                    return Spillage
				# why defines the delta function here and not outside the loop?
                def cal_delta(VI, V):
                    return ((VI[ist]-V)/sspsu.update0(VI[ist])).abs()        # abs or **2?
                # central expression appears to be here, one can modifiy the mixing coefficients between the two terms
                # psi and dpsi here.
                # CALCULATE THE SPILLAGE
                s0 = cal_Spillage(cal_delta(VI_origin, V_origin))
                Spillage += coef_deriv0*s0
				# append the linear term to the spillage if the linear term is defined
                if "linear" in file_list.keys():
                    for i in range(len(file_list["linear"])):
                        s1 = cal_Spillage(cal_delta(VI_linear[i], V_linear[i]))
                        Spillage += coef_deriv1*s1
            # END: hack function here to change definition of Spillage
            # --------------------------------
            # kinetic energy term contribution
            if info_opt.cal_T:
                T = Opt_Orbital.cal_T(C, E)
                if not "TSrate" in vars(): TSrate = torch.abs(0.002*Spillage/T).data[0]
                Loss = Spillage + TSrate*T
            else:
                Loss = Spillage
			# output the information of the current step
            duration = time.time() - time_stepstart
            # calculate the change of spillage
            _dspill = Spillage.item() - _spill_0
            _spill_0 += _dspill
            if info_opt.cal_T:
                print(f"{istep:>10}{Spillage.item():>20.10e}{T.item():>20.10f}{Loss.item():>20.10f}{duration:>10.4f}", file=S_file, flush=True)
                if not istep % 100:
                    print(f"{istep:>10}{Spillage.item():>20.10e}{T.item():>20.10f}{Loss.item():>20.10f}{duration:>10.4f}", flush=True)
            else:
                print(f"{istep:>10}{Spillage.item():>20.10e}{_dspill:>20.10e}{duration:>10.4f}", file=S_file, flush=True)
                if not istep % 100:
                    print(f"{istep:>10}{Spillage.item():>20.10e}{_dspill:>20.10e}{duration:>10.4f}", flush=True)
                    
            flag_finish = 0
            if Loss.item() < loss_old:
                loss_old = Loss.item()
                C_old = sspsifc.copy_C(C, info_element)
                flag_finish = 0
            else:
                flag_finish += 1
                if flag_finish > 50:
                    break

            orb_optimizer.zero_grad()
            Loss.backward()        
            if C_init_info["init_from_file"] and not C_init_info["opt_C_read"]:
                for it, il, iu in C_read_index:
                    C[it][il].grad[:, iu] = 0
            orb_optimizer.step()
            #orbital.normalize(
            #    orbital.generate_orbital(info_element, C, E),
            #    {it:info_element[it].dr for it in info_element},
            #    C, flag_norm_C=True)
            
            # add the convergence condition here, while for old version the optimization
            # will proceed until the max step is reached. However observed that the optimization
            # will converge in a few steps.
            if _dspill <= spill_thr:
                print(f"...\nSpillage meets convergence threshold {spill_thr} ({_dspill}) at step {istep}.", flush=True)
                break
            if istep == maxSteps-1:
                print(f"...\nWARNING: Spillage optimization reaches the maximum steps {maxSteps} without convergence ({spill_thr}).", flush=True)

    orb = sspso.generate_orbital(info_element, C_old, E)
    # this is a ad hoc way to smooth the orbital. A more clean way would be implemented
    # in future version.
    if info_opt.cal_smooth:
        sspso.smooth_orbital(orb=orb,
                             Rcut={it:info_element[it].Rcut for it in info_element}, 
                             dr={it:info_element[it].dr for it in info_element},
                             smearing_sigma=0.1)
    # Schmidt orthogonalization on smoothed orbitals
    sspso.orth(orb=orb, dr={it:info_element[it].dr for it in info_element})
    # write file ORBITAL_{}U.dat -> uuid named
    sspsipo.print_orbital(fu=fu, orb=orb, info_element=info_element)
    # write file ORBITAL_PLOTU.dat -> uuid named
    sspsipo.plot_orbital(fplotu=fplotu,
                         orb=orb,
                         Rcut={it:info_element[it].Rcut for it in info_element},
                         dr={it:info_element[it].dr for it in info_element})
    # write file ORBITAL_RESULTS.txt -> uuid named
    sspsifc.write_C(fcoef, C_old, Spillage)

    print("""
---------------------------------
Optimization of the orbital ends.

Several files generated:
Spillage.dat: detailed trajectory of the optimization
ORBITAL_RESULTS.txt: optimized orbital coefficients
ORBITAL_*U.dat: numerical atomic orbital before renaming
ORBITAL_PLOTU.dat: for plot, the first column is the r, latter colomns are the orbitals

TOTAL TIME (PyTorch):     %s"""%(time.time()-time_start), flush=True)
    # finally return the file names to keep track on the files
    return fspill, fcoef, fplotu, fu

if __name__=="__main__":
    import sys
    np.set_printoptions(threshold=sys.maxsize, linewidth=10000)
    print(sys.version, flush=True) 
    main()
