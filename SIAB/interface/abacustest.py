"""this is copied from ABACUS-Pseudopot-Nao-Square (APNS) project,
for parallelization of the reference structure PW calculation.
----
INTERFACE TO ABACUSTEST
Author: @kirk0830
Github repo: https://github.com/pxlxingliang/abacus-test (by @pxlxingliang)

Usage:
    call function `write_abacustest_param` to generate abacustest param.json contents

Example:
    write_abacustest_param(jobgroup_name="example_run",
                           bohrium_login={"username": "***@aisi.ac.cn", "password": "****", "project_id": "***"},
                           save_dir="~/abacus_test",
                           prepare={"abacus2qe": True, "folders": ["example1", "example2"]},
                           predft={"ifrun": True, "command": "python3 prepare_something.py", "shared_files": ["file1", "file2"]},
                           rundft=[{"ifrun": True, "command": "abacus --version", "shared_files": ["file1", "file2"]}],
                           postdft={"ifrun": True, "command": "python3 post_something.py", "shared_files": ["file1", "file2"]})
    can specify `export=True` to save the param.json file to the current directory, then will return absolute path of the file
    instead of contents of the file

"""

#ABACUS_IMAGE = "registry.dp.tech/deepmodeling/abacus-intel:latest"
ABACUS_IMAGE = "registry.dp.tech/dptech/abacus:3.6.4" # this is for submit large batch of jobs, but need to always update
QE_IMAGE = "registry.dp.tech/dptech/prod-471/abacus-vasp-qe:20230116"
VASP_IMAGE = "registry.dp.tech/dptech/prod-471/abacus-vasp-qe:20230116"
PYTHON_IMAGE = "python:3.8"
ABACUS_COMMAND = "OMP_NUM_THREADS=1 mpirun -n 16 abacus | tee out.log"

import os
import json
def read_apns_inp(fname: str) -> dict:
    assert os.path.exists(fname), f"File not found: {fname}"
    with open(fname, "r") as f:
        inp = json.load(f)
    return inp.get("abacustest", {})

def manual_submit(username, password, project_id, ncores, memory, folders):
    import time, os
    assert all([os.path.exists(f) for f in folders]), "Some folders do not exist."
    jobgroup = f"apns_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"
    run_dft = [{"ifrun": True, "job_folders": folders, "command": ABACUS_COMMAND, "ncores": ncores, "memory": memory}]
    param = write_abacustest_param(jobgroup_name=jobgroup, 
                                   bohrium_login={"bohrium.account": username, "bohrium.password": password, "project_id": project_id}, 
                                   rundft=run_dft)
    result_folder = submit(param)
    return result_folder


def bohrium_machine(ncores: int, memory: float, device: str, supplier: str):
    """Configure Bohrium machine information
    
    Args:
        ncores (int): number of cores
        memory (float): memory size
        device (str): device type, example: "cpu", "gpu"
        supplier (str): supplier name, example: "ali", "para"

    Returns:
        bohrium machine settings: dict, involved in abacustest configuration
    """
    return {
        "scass_type": "_".join(["c"+str(ncores), "m"+str(memory), device]),
        "job_type": "container",
        "platform": supplier
    }

def prepare_dft(**kwargs):
    """Generate "prepare" section of abacustest configuration

    Returns:
        dict: prepare configuration
    """
    # there are two ways to generate batch of input:
    # 1. with already-existed input files to overwrite
    # 2. generate input files from scratch
    #
    # for mode 1, a list of folders will be the value
    # of list "example_template"
    # for mode 2, three keys "input_template", "stru_template",
    # and "kpt_template" will/should be defined.
    result = {}
    keys, vals = abacus_default()
    for key in keys:
        if key in kwargs.keys() and not key in ["pseudo_dir", "orbital_dir"]:
            result.setdefault("mix_input", {})[key] = kwargs[key] if isinstance(kwargs[key], list) else [kwargs[key]]
    result.update({"example_template": kwargs.get("folders", [])})
    result.update(dict(zip(["input_template", "stru_template", "kpt_template"], 
                           [kwargs.get(key, key.upper()) for key in ["input", "stru", "kpt"]]
                           ))) if result["example_template"] == [] else None
    result = {} if result.get("mix_input", {}) == {} and result.get("example_template", []) == [] else result

    result.update(dict(zip(["mix_kpt", "mix_stru"], [[], []])))
    result["pp_dict"] = kwargs.get("pp_dict", {})
    result["orb_dict"] = kwargs.get("orb_dict", {})
    result["pp_path"] = kwargs.get("pseudo_dir", "")
    result["orb_path"] = kwargs.get("orbital_dir", "")
    result["dpks_descriptor"] = kwargs.get("dpks_descriptor", "")
    result["extra_files"] = kwargs.get("shared_files", [])
    result["abacus2qe"] = kwargs.get("abacus2qe", False)
    result["qe_setting"] = kwargs.get("qe_setting", {})
    result["abacus2vasp"] = kwargs.get("abacus2vasp", False)
    result["vasp_setting"] = kwargs.get("vasp_setting", {})
    result["potcar"] = kwargs.get("potcar", [])
    # seperate key-value pairs, for value who has len() attribute, save to _container, otherwise save to _rest
    _container = {key: value for key, value in result.items() if hasattr(value, "__len__")}
    _rest = {key: value for key, value in result.items() if not key in _container.keys()}
    _container = {key: value for key, value in _container.items() if len(value) > 0}
    _rest = {key: value for key, value in _rest.items() if value}
    return {**_container, **_rest}

def setup_dft(**kwargs):
    """Generate predft/rundft/postdft configuration according to given parameters

    Returns:
        dict: predft/rundft/postdft configuration
    """
    # if kwargs is empty, return an empty dictionary
    if not kwargs:
        return {}
    def find_image(command: str):
        if "abacus" in command:
            return ABACUS_IMAGE
        elif "pw.x" in command:
            return QE_IMAGE
        elif "vasp" in command:
            return VASP_IMAGE
        elif "python" in command:
            return PYTHON_IMAGE
        else:
            return kwargs.get("image", PYTHON_IMAGE)
    # to distinguish between predft/rundft and postdft
    metrics = kwargs.get("metrics", None)
    postdft = bool(metrics)
    # rundft specific
    sub_save_path = kwargs.get("sub_save_path", None)
    rundft = True if sub_save_path else False
    # mutual keys
    switch = kwargs.get("ifrun", False)
    command = kwargs.get("command", "abacus --version")
    shared_files = kwargs.get("shared_files", [])
    image = find_image(command)
    example = kwargs.get("job_folders", [])
    print("Presently the key `outputs` is deprecated due to imcompleted implementation of the feature of abacustest.")
    machine = bohrium_machine(kwargs.get("ncores", 16), kwargs.get("memory", 32), "cpu", "ali") if "ncores" in kwargs.keys() or "memory" in kwargs.keys() else None
    # rundft specific
    group_size = kwargs.get("njobs_node", 1) # rundft specific

    result = {
        "ifrun": switch,
        "command": command,
        "extra_files": shared_files,
        "image": image,
        "example": example
    }
    if machine is not None:
        result["bohrium"] = machine
    # on demand is always on, avoiding to be killed
    result["bohrium"]["on_demand"] = 1

    if rundft:
        result["group_size"] = group_size
        result["sub_save_path"] = sub_save_path
    if postdft:
        result["metrics"] = metrics
    return result

def write_abacustest_param(jobgroup_name: str, bohrium_login: dict, save_dir: str = "", prepare: dict = None,
                           predft: dict = None, rundft: list = None, postdft: dict = None, export: bool = False):
    """Generate abacustest param.json contents

    Args:
        jobgroup_name (str): identifier for the jobgroup, not the lbg_jobgroup_id
        bohrium_login (dict): a dictionary should have `username`, `password`, `project_id` keys
        save_dir (str): path to save the job results
        predft (dict): preprocessing procedure, usually for setting up some parameters in batch
        rundft (list): a list of dictionaries, each dictionary represents a manner of running DFT calculation
        postdft (dict): postprocessing procedure, usually for analyzing the results

    Returns:
        dict: abacustest param.json contents
    """
    import time
    prepare = prepare or {}
    predft = predft or {}
    rundft = rundft or []
    postdft = postdft or {}
    
    save_dir = save_dir if len(save_dir) > 0 else f"abacustest-autosubmit-{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"
    result = {
        "bohrium_group_name": jobgroup_name,
        "config": bohrium_config(**bohrium_login),
        "save_path": save_dir,
        "prepare": prepare_dft(**prepare),
        "pre_dft": setup_dft(**predft),
        "run_dft": [setup_dft(**dft) for dft in rundft],
        "post_dft": setup_dft(**postdft)
    }
    result = {k: v for k, v in result.items() if len(v) > 0}
    if export:
        with open("param.json", "w") as f:
            json.dump(result, f, indent=4)
        return os.path.abspath("/".join([os.getcwd(), "param.json"]))
    return result

def submit(abacustest_param: dict) -> str:
    import time
    fparam = f"param-{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.json"
    with open(fparam, "w") as f:
        json.dump(abacustest_param, f, indent=4)
    
    folder = abacustest_param.get("save_path", "result")
    flog = fparam.rsplit(".", 1)[0] + ".log"
    os.system(f"nohup abacustest submit -p {fparam} > {flog}&")
    print(f"Job submitted, log file is {flog}, results will be downloaded into {folder}")
    return folder

def read_keyvals_frominput(fin, keyword: str = None):
    import re
    kv = r"^([\w_-]+)(\s+)([^#]*)(.*)$"

    result = {}
    with open(fin, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    lines = [line for line in lines if len(line) > 0]
    for line in lines:
        _match = re.match(kv, line)
        if _match is not None:
            result[_match.group(1)] = _match.group(3).strip()
    
    return result if keyword is None else result[keyword]

def abacus_default():
    import os
    fthis = os.path.abspath(__file__)
    fabacus = fthis.replace("/abacustest.py", "/abacus_input_example")
    result = read_keyvals_frominput(fabacus).items()
    keys, vals = zip(*result)
    return keys, vals

if __name__ == "__main__":

    import os, time
    src = "/root/documents/simulation/orbgen/apns-orbgen-project/nelec_delta_test/lcao-v2.0"
    jobgroup = "u-nspin2_lcao-v2.0"
    fgroup = os.path.join(src, jobgroup)
    os.chdir(fgroup)
    folders = os.listdir()
    manual_submit("_", "_", "28682", 32, 64, folders)
    time.sleep(10)
    print(f"ABACUSTEST: Jobgroup {jobgroup} submitted.")
    exit()
    jobgroups = os.listdir(src)
    for jobgroup in jobgroups:
        fgroup = os.path.join(src, jobgroup)
        os.chdir(fgroup)
        folders = os.listdir()
        manual_submit("_", "_", "28682", 32, 64, folders)
        time.sleep(10)
        print(f"ABACUSTEST: Jobgroup {jobgroup} submitted.")
        os.chdir(src)