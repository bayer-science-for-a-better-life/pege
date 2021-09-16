import torch

OHE_ATOMS_GRAPH = [
    "H",
    "N",
    "NZ_LYS",
    "NH_ARG",
    "O_AMIDE",
    "OH_TYR",
    "NE2_HIS",
    "SD_MET",
    "SG_CYS",
    "OG_SER",
    "O",
    "NE1_TRP",
    "OG1_THR",
    "O_COOH",
    "NE_ARG",
    "N_AMIDE",
    "ND1_HIS",
]


def read_pdb_line(line: str) -> list:
    aname = line[12:16].strip()
    anumb = int(line[5:11].strip())
    resname = line[17:21].strip()
    chain = line[21]
    resnumb = int(line[22:26])
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    return (aname, anumb, resname, chain, resnumb, x, y, z)


def classify_atom(aname: str, resname: str) -> str:
    if aname in ("N", "O"):
        if resname != "NTR":
            return aname
        else:
            return "NZ_LYS"
    elif resname in ("CTR", "ASP", "GLU"):
        return f"O_COOH"
    elif resname in ("GLN", "ASN"):
        return f"{aname[0]}_AMIDE"
    elif resname == "ARG" and aname in ("NH1", "NH2"):
        return "NH_ARG"
    elif resname == "CY0" and aname == "SG":
        return "SG_CYS"
    else:
        return f"{aname}_{resname}"


def pdb2feats(fname: str):
    def iter_f(fname):
        with open(fname) as f:
            for line in f:
                yield line

    coords, feats, anumbs, details = [], [], [], []
    for line in iter_f(fname):
        if not line.startswith("ATOM "):
            continue

        aname, anumb, resname, chain, resnumb, x, y, z = read_pdb_line(line)
        res = classify_atom(aname, resname)

        if res in OHE_ATOMS_GRAPH:
            res_ohe = OHE_ATOMS_GRAPH.index(res)

            coords.append((x, y, z))
            feats.append(res_ohe)
            anumbs.append(anumb)
            details.append((chain, resnumb, resname, aname))

    coords = torch.tensor(coords).reshape(1, -1, 3)
    feats = torch.tensor(feats).reshape(1, -1)

    return coords, feats, anumbs, details
