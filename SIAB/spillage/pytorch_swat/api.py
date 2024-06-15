import SIAB.interface.old_version as siov
import SIAB.spillage.pytorch_swat.main as sspsm
import SIAB.spillage.orbscreen as sso
import multiprocessing
import torch

def run(params: dict = None, cache_dir: str = "./", ilevel: int = 0, nlevel: int = 3):
    """Run the spillage calculation
    
    Args:
        params (dict): parameters for the spillage calculation
        cache_dir (str): the directory to store the cache files
        ilevel (int): the current level of the calculation
        nlevel (int): the total number of levels of the calculation
    
    Returns:
        tuple: a tuple containing the path of the orbital file and the screen values
    """
    
    """convert-back the information organized in the way that is acceptable
    for the original version of SIAB to the following format:
    ```python
    return {
        "element": element,
        "ecutwfc": ecutwfc,
        "rcut": rcut,
        "zeta_notation": zeta_notation,
    }
    ```
    """
    # generate the checkpoint from old version flavor input
    chkpt = siov.unpack(orb_gen=params)
    # generate the folder name from checkpoint which can identify current process of orbital generation
    folder = siov.folder(unpacked_orb=chkpt)
    # certainly if is duplicated task, directly skip
    skip, forb = is_duplicate(folder)
    if skip:
        return forb, sso.screen(fnao=forb, item="T")
    # else there are two ways to run the spillage optimizer, 1 is read from external file, the other
    # is directly from the input parameters
    fspill, fcoef, fplotu, fu = sspsm.main() if params is None else sspsm.main(params)
    fs = ["Spillage.dat", "ORBITAL_RESULTS.txt", "ORBITAL_PLOTU.dat", "ORBITAL.dat"]
    fs = dict(zip(fs, [fspill, fcoef, fplotu, fu]))
    # after the step above, will generate several files like ORBITAL_RESULTS, ORBITAL_PLOTU, ORBITAL_U.dat
    # in cwd. Before the next run, move them to target folders.

    refresh = True if ilevel == nlevel-1 else False
    # refresh is for controlling whether leave the ORBITAL_RESULTS.txt in the original folder
    # if so, orbitals will be optimized based on the previous results, say hierarchical optimization
    # but for the last level, we should remove the ORBITAL_RESULTS.txt.
    _, _, _, forb = checkpoint(src="./", dst=folder, progress=chkpt, cache_dir=cache_dir, out_files=fs,
                               refresh=refresh)

    # analysis the newly generated orbitals' kinetic energies of all components
    screen_vals = sso.screen(fnao=forb, item="T")

    return forb, screen_vals

import SIAB.interface.old_version as siov
import sys
def iter(siab_settings, calculation_settings):
    """iterate on siab_settings, can support parallelization according to user settings"""
    nlevel=len(siab_settings["orbitals"]) # this dimension must be executed in serial
    
    # parallelization setting
    # serial if nthreads_rcut is not set or less than 0.
    nthreads_rcut = siab_settings.get("nthreads_rcut", -1) # siab_setting must have this key because
    # has been set in the read_input.py function siab_settings() function
    be_serial = True if nthreads_rcut <= 0 else False
    
    # according to user setting, calculate how many rcuts can be parallelized at the same time
    nthreads_max = torch.get_num_threads()
    # if nthreads_rcut is -1, then use all available threads for one rcut, which results in serial calculation
    nthreads_rcut = nthreads_max if nthreads_rcut <= 0 else nthreads_rcut
    rcuts = calculation_settings[0]["bessel_nao_rcut"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    nrcuts = len(rcuts)
    nrcuts_toparallel = nthreads_max // nthreads_rcut # the number of rcuts that can be parallelized
    # however, for bad settings, will result in nrcut_toparallel < 1, in this case, be_serial = True
    if nrcuts_toparallel <= 1 and nthreads_rcut > 0 and not be_serial:
        print(f"""
Parallelization - WARNING
The parameter `nthreads_rcut` is set to be larger than all available threads
of present machine, therefore the calculation will switch to run in serial.
nthreads_rcut:     {siab_settings.get("nthreads_rcut", -1)} (number of threads for each rcut)
nthreads_max:      {nthreads_max} (total number of threads available)
nruts:             {nrcuts} (number of rcuts)
nrcuts_toparallel: {nrcuts_toparallel} (number of rcuts that can be parallelized at the same time)
""", flush=True)

    be_serial = True if nrcuts_toparallel < 1 else be_serial

    # run!
    if be_serial:
        for old_input, cache_dir, ilevel in siov.convert(calculation_setting=calculation_settings[0],
                                                         siab_settings=siab_settings):
            orb_out = run(params=old_input, cache_dir=cache_dir, ilevel=ilevel, nlevel=nlevel)
            postprocess(orb_out)
    else:
        orbgen_plans = []
        for old_input, cache_dir, ilevel in siov.convert(calculation_setting=calculation_settings[0],
                                                         siab_settings=siab_settings):
            if ilevel == 0: orbgen_plans.append([])
            orbgen_plans[-1].append((old_input, cache_dir, ilevel))
        # then in orbgen_plans, for one rcut i and level j, can be accessed by orbgen_plans[i][j]
        # then we can parallelize the calculation by rcut, therefore first loop over levels, then
        # loop over rcut. The loop over rcut can be parallelized.
        # nprocs_rcut is the number of processes for each rcut
        # be aware that nprocs_rcut < 1 is not allowed, if there are really only one
        # logical processor, then nprocs_rcut = 1
        print(f"""
Parallelization - RUNTIME
Number of threads for each rcut: {nthreads_rcut}
Number of rcuts that can be parallelized: {nrcuts_toparallel}
Total number of threads available: {nthreads_max}
----------------------------------
NOTE: for parallelized run, the stdout and stderr will be redirected to log.[iproc].txt and err.[iproc].txt respectively.
""", flush=True)
        # start!
        for ilevel in range(nlevel):
            # because for each rcut, the latter level will depend on the former level
            # therefore, parallelize rcut and barrier, then serialize the levels
            procs = []
            # each time get nrcuts_toparallel rcuts from nruts to run
            for ircut_start in range(0, nrcuts, nrcuts_toparallel):
                # refresh the procs list
                procs = []
                # for each rcut, run the calculation
                for ircut in range(ircut_start, min(ircut_start+nrcuts_toparallel, nrcuts)):
                    torch.set_num_threads(nthreads_rcut)
                    inp, cdir, ilv = orbgen_plans[ircut][ilevel]
                    # create a process and also get its return value
                    proc = multiprocessing.Process(target=run, args=(inp, cdir, ilv, nlevel))
                    # redirect stdout to log.[iproc].txt and stderr to err.[iproc].txt for each process
                    sys.stdout = open("log.%d.txt"%ircut, "a+")
                    sys.stderr = open("err.%d.txt"%ircut, "a+")
                    procs.append(proc)
                    proc.start()
                # wait for all processes to finish, then recover stdout and stderr
                for proc in procs:
                    proc.join() # barrier
                torch.set_num_threads(nthreads_max)
                # after all processes finish, recover stdout and stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
            print(f"Finish level {ilevel} orbital generation (in total {nlevel}).", flush=True)

        print("All processes finish, see stdout and stderr in log.[iproc].txt and err.[iproc].txt respectively.", flush=True)
    
    return

import os
import re
def is_duplicate(folder: str):
    """check if the siab calculation is skipped"""
    if not os.path.isdir(folder):
        return False, None
    # becuase ORBITAL_{}U.dat stores the same information as *.orb
    # therefore we only need to check the existence of Spillage.dat, ORBITAL_RESULTS.txt, ORBITAL_PLOTU.dat
    # and *.orb
    # deprecated
    # orbital_u = r"^(ORBITAL_)([0-9]+)(U\.dat)$"
    orbital = r"^([A-Z][a-z]?)(_gga_)(\d+\.?\d?)(Ry_)(\d+\.?\d?)(au_)((\d\w)+)(\.orb)$"
    files = os.listdir(folder)
    print("Checking files in %s..."%folder)
    if "Spillage.dat" in files:
        print("    Spillage.dat exists")
        if "ORBITAL_RESULTS.txt" in files:
            print("    ORBITAL_RESULTS.txt exists")
            if "ORBITAL_PLOTU.dat" in files:
                print("    ORBITAL_PLOTU.dat exists")
                for f in files:
                    if re.match(orbital, f):
                        print("    %s exists => duplicate check pass, will skip."%f)
                        return True, f"{folder}/{f}"
    return False, None

import SIAB.interface.env as sienv
#import SIAB.data.interface as sdi
def checkpoint(src: str,
               dst: str,
               progress: dict,
               cache_dir: str = "./",
               out_files: dict = None,
               env: str = "local",
               refresh: bool = False):
    """After optimization of numerical orbitals' coefficients,
       move generated orbitals to the folder named as:
       [element]_gga_[Ecut]Ry_[Rcut]au_[orbital_config]

       ONCE ONE OPTIMIZATION TASK COMPLETES, CALL THIS FUNCTION.
    Design:
        all information should be included in user_settings
        rather than externally defined additionally.
    
    Args:
        user_settings (dict): user settings
        rcut (float): cutoff radius
        orbital_config (str): orbital configuration, e.g. 1s1p
    
    Returns:
        all files path after mv/cp
    """
    # first check if the folder exists, if not, create it
    element, ecutwfc, rcut, orbital_config = progress["element"], progress["ecutwfc"], progress["rcut"], progress["zeta_notation"]
    print("CHECKPOINT: handling on temporary files:", flush=True)
    for k, v in out_files.items():
        print("            %-20s: %s"%(k, v), flush=True)
    
    if not os.path.isdir(dst):
        sienv.op("mkdir", dst, additional_args=["-p"], env=env)
        print(f"CHECKPOINT: folder {dst} created.")
    if not os.path.isdir(cache_dir):
        sienv.op("mkdir", cache_dir, additional_args=["-p"], env=env)
        print(f"CHECKPOINT: folder {cache_dir} created.")

    # deprecated
    # """backup input file, unlike the original version, we fix it must be named as SIAB_INPUT"""
    # sienv.op("cp", "%s/SIAB_INPUT"%src, "%s/SIAB_INPUT"%dst, env=env)
    # files.append("%s/SIAB_INPUT"%dst)

    """move spillage.dat"""
    fspill = out_files["Spillage.dat"]
    sienv.op("mv", src = f"{src}/{fspill}", dst = f"{dst}/Spillage.dat", env=env)
    fspill = f"{dst}/Spillage.dat"

    """move ORBITAL_PLOTU.dat"""
    fplotu = out_files["ORBITAL_PLOTU.dat"]
    sienv.op("mv", src = f"{src}/{fplotu}", dst = f"{dst}/ORBITAL_PLOTU.dat", env=env)
    fplotu = f"{dst}/ORBITAL_PLOTU.dat"

    """cache and/or move ORBITAL_RESULTS.txt"""
    fcoef = out_files["ORBITAL_RESULTS.txt"]
    if not refresh:
        sienv.op("cp", src = f"{src}/{fcoef}", dst = f"{dst}/ORBITAL_RESULTS.txt", env=env)
        cache(src=src, fcoef=fcoef, cache_dir=cache_dir, env=env)
    else:
        sienv.op("mv", src = f"{src}/{fcoef}", dst = f"{dst}/ORBITAL_RESULTS.txt", env=env)
        sienv.op("rm", f"{cache_dir}", env=env, additional_args=["-rf"])
    fcoef = f"{dst}/ORBITAL_RESULTS.txt"

    """move ORBITAL.dat to *.orb"""
    forb = f"{element}_gga_{ecutwfc}Ry_{rcut}au_{orbital_config}.orb"
    fu = out_files["ORBITAL.dat"]
    sienv.op("mv", src = f"{src}/{fu}", dst = f"{dst}/{forb}", env=env)
    forb = f"{dst}/{forb}"
    print("Orbital file %s generated."%forb, flush=True)
    
    """plot the orbital"""
    from SIAB.spillage.plot import plot_orbfile
    plot_orbfile(forb, save=forb.replace(".orb", ".png"))

    # deprecated
    # """and directly move it to the folder"""
    # index = sdi.PERIODIC_TABLE_TOINDEX[element]
    # sienv.op("mv", "%s/ORBITAL_%sU.dat"%(src, index), "%s/ORBITAL_%sU.dat"%(dst, index), env=env)
    # files.append("%s/ORBITAL_%sU.dat"%(dst, index))

    # finally will contain Spillage.dat, ORBITAL_PLOTU.dat, ORBITAL_RESULTS.txt, and *.orb
    return fspill, fcoef, fplotu, forb

def cache(src: str = "./", fcoef: str = "ORBITAL_RESULTS.txt", cache_dir: str = "./", env: str = "local"):
    """in cache_dir, if there is already a file named as Level[ilevel].ORBITAL_RESULTS.txt,
    then move the ORBITAL_RESULTS.txt to Level[ilevel+1].ORBITAL_RESULTS.txt"""
    ilevel = 0
    while True:
        if os.path.isfile("%s/Level%s.ORBITAL_RESULTS.txt"%(cache_dir, ilevel)):
            ilevel += 1
        else:
            break
    sienv.op("mv", f"{src}/{fcoef}", "%s/Level%s.ORBITAL_RESULTS.txt"%(cache_dir, ilevel), env=env)

def postprocess(orb_out = None):
    
    if orb_out is None:
        return
    
    forb, quality = orb_out
    # instantly print the quality of the orbital generated
    print("Report: quality of the orbital %s is:"%forb, flush=True)
    for l in range(len(quality)):
        print("l = %d: %s"%(l, " ".join(["%10.8e"%q for q in quality[l] if q is not None])), flush=True)

    return None
