import SIAB.io.restart as sisrt
import SIAB.interface.old_version as siov
import SIAB.spillage.pytorch_swat.main as sspsm
def driver(params: dict = None, ilevel: int = 0, nlevel: int = 3):

    chkpt = siov.unpack(orb_gen=params)
    folder = siov.folder(unpacked_orb=chkpt)
    if sisrt.siab_skip(folder):
        return
    
    if params is None:
        sspsm.main()
    else:
        sspsm.main(params)
    
    refresh = True if ilevel == nlevel-1 else False
    sisrt.checkpoint(src="./", dst=folder, this_point=chkpt, refresh=refresh)
    return