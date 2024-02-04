import SIAB.io.restart as sisrt
import SIAB.interface.old_version as siov
import SIAB.opt_orb_pytorch_dpsi.main as soopdm
def driver(params: dict = None, ilevel: int = 0, nlevel: int = 3):

    chkpt = siov.unpack(orb_gen=params)
    folder = siov.folder(unpacked_orb=chkpt)
    if sisrt.siab_skip(folder):
        return
    
    if params is None:
        soopdm.main()
    else:
        soopdm.main(params)
    
    refresh = True if ilevel == nlevel-1 else False
    sisrt.checkpoint(src="./", dst=folder, this_point=chkpt, refresh=refresh)
    return