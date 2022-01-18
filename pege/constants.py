import numpy as np

pH_scale = list(np.arange(-6, 20.1, 0.5))

AA_HS = {
    "ASP": ["HD11", "HD21", "HD12", "HD22"],
    "CTR": ["HO11", "HO21", "HO12", "HO22"],
    "CYS": ["HG1", "HG2", "HG3"],
    "GLU": ["HE11", "HE21", "HE12", "HE22"],
    "HIS": ["HD1", "HE2"],
    "LYS": ["HZ1", "HZ2", "HZ3"],
    "NTR": ["H1", "H2", "H3"],
    "TYR": ["HH1", "HH2"],
    "SER": ["HG1", "HG2", "HG3", "HG"],
    "THR": ["HG1", "HG2", "HG3", "HG"],
}

OHE_ATOMS_GRAPH = {
    "H": [(res, atoms) for res, atoms in AA_HS.items()],
    "CA": [(None, ["CA"])],
    "N": [(None, ["N"])],
    "NZ_LYS": [("LYS", "NZ"), ("NTR", "N")],
    "NH_ARG": [("ARG", ["NH1", "NH2"])],
    "NE_ARG": [("ARG", "NE")],
    "NE1_TRP": [("TRP", "NE1")],
    "NE2_HIS": [("HIS", "NE2")],
    "ND1_HIS": [("HIS", "ND1")],
    "N_AMIDE": [("GLN", "NE2"), ("ASN", "ND2")],
    "O": [(None, "O")],
    "O_COOH": [("GLU", ["OE1", "OE2"]), ("ASP", ["OD1", "OD2"]), ("CTR", ["O1", "O2"])],
    "O_AMIDE": [("GLN", "OE1"), ("ASN", "OD1")],
    "OG_SER": [("SER", "OG")],
    "OH_TYR": [("TYR", "OH")],
    "OG1_THR": [("THR", "OG1")],
    "SG_CYS": [(None, "SG")],
    "SD_MET": [("MET", "SD")],
}
