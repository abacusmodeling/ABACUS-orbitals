def initialize(calculation_settings, siab_settings, folders):
    """prepare the input for orbital optimization task.
    Because the folder name might not be determined at the beginning of task if perform
    `auto` on bond_length, the exact folder name will be given uniformly after performing
    abacus calculation, therefore it is compulsory to update the folder name in siab_settings
    and cannot be somewhere else."""
    # because it is orbital that is generated one-by-one, it is reasonable to iterate
    # the orbitals...
    # NOTE: the following loop move necessary information for orbital generation from 
    # calculation_settings to siab_settings, so to make decouple between these
    # two dicts -> but this can be done earlier. The reason why do it here is because
    # the conversion from newly designed data structure, calculation_settings, and
    # siab_settings, to the old version of SIAB, is taken place here.
    for orbital in siab_settings["orbitals"]:
        # MOVE: copy lmax information from calculation_settings to siab_settings
        # for old version SIAB, but lmax can be read from orb_matrix*.dat for better design
        # in that way, not needed to import lmaxmax information from calculation_settings
        orbital["lmax"] = calculation_settings[orbital["folder"]]["lmaxmax"]
        # the key "folder" has a more reasonable alternative: "abacus_setup"

        # MOVE: let siab_settings know exactly what folders are used for orbital generation
        # in folders, folders are stored collectively according to reference structure
        # and the "abacus_setup" is then expanded to the corresponding folder
        # therefore this setup is to expand the "folder" key to the corresponding folder
        # list
        i = orbital["folder"] if siab_settings.get("optimizer", "none") not in ["none", "restart"] else 0
        orbital["folder"] = folders[i]

    return siab_settings