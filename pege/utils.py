import torch
import os
from pdbmender import (
    mend_pdb,
    add_tautomers,
    identify_tit_sites,
    rm_cys_bridges,
    identify_cter,
)
from pdbmender.formats import get_chains_from_file
from pege.constants import OHE_ATOMS_GRAPH, AA_HS


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
    if aname.startswith("H") and (resname in AA_HS and aname in AA_HS[resname]):
        return "H"
    elif aname == "CA":
        return "CA"

    if aname in ("N", "O"):
        if resname != "NTR":
            return aname
        else:
            return "NZ_LYS"
    elif resname in ("CTR", "ASP", "GLU"):
        return "O_COOH"
    elif resname in ("GLN", "ASN"):
        return f"{aname[0]}_AMIDE"
    elif resname == "ARG" and aname in ("NH1", "NH2"):
        return "NH_ARG"
    elif resname == "CY0" and aname == "SG":
        return "SG_CYS"
    else:
        return f"{aname}_{resname}"


def iter_f(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("ATOM "):
                yield line


def get_atom_info(fname):
    original_atom_details = {}
    termini_restypes = {}
    for line in iter_f(fname):
        aname, anumb, resname, chain, resnumb, x, y, z = read_pdb_line(line)
        inscode = line[26]
        details = (chain, resnumb, resname, aname, inscode)
        original_atom_details[anumb] = details

    original_atom_numbs = original_atom_details.keys()
    original_atom_infos = original_atom_details.values()

    return original_atom_numbs, original_atom_infos


def fix_structure(fname, ff):
    sysname = fname
    if ".pdb" in fname:
        sysname = fname.split(".pdb")[0]
    elif ".pqr" in fname:
        sysname = fname.split(".pqr")[0]

    pdb_cleaned = f"{sysname}_cleaned.pdb"
    logfile_mend = "LOG_pdb2pqr"
    renumbered = mend_pdb(fname, pdb_cleaned, ff, ff, logfile=logfile_mend)

    chains = get_chains_from_file(fname)
    chains_res = identify_tit_sites(fname, chains, add_ser_thr=True)

    chains_res, cys_bridges = rm_cys_bridges(chains_res, logfile_mend)

    old_ctrs = {
        chain: resnumb
        for chain, residues in chains_res.items()
        for resnumb, resname in residues.items()
        if resname == "CTR"
    }
    new_ctrs = identify_cter(pdb_cleaned, old_ctrs)
    for chain, resnumb in new_ctrs.items():
        chains_res[chain][str(resnumb)] = "CTR"

    output_pdb = f"{sysname}_final.pdb"
    _ = add_tautomers(pdb_cleaned, chains_res, ff, output_pdb)

    os.system(f"rm {pdb_cleaned} LOG* removed.pqr addhtaut_cleaned.pdb")
    return output_pdb, chains_res, renumbered


def get_termini_info(fname, chains_res):
    termini_resnumbs = {chain: [None, None] for chain in chains_res.keys()}
    for chain, residues in chains_res.items():
        for resnumb, resname in residues.items():
            resnumb = int(resnumb)
            if resname == "NTR" and not termini_resnumbs[chain][0]:
                termini_resnumbs[chain][0] = resnumb
            elif resname == "CTR" and not termini_resnumbs[chain][1]:
                termini_resnumbs[chain][1] = resnumb

    termini_resnames = {chain: ["NTR", "CTR"] for chain in chains_res.keys()}
    for line in iter_f(fname):
        aname, anumb, resname, chain, resnumb, x, y, z = read_pdb_line(line)

        if (
            termini_resnumbs[chain][0]
            and resnumb == termini_resnumbs[chain][0]
            and resname != "NTR"
        ):
            termini_resnames[chain][0] = resname
        elif (
            termini_resnumbs[chain][1]
            and resnumb == termini_resnumbs[chain][1]
            and resname != "CTR"
        ):
            termini_resnames[chain][1] = resname

    return termini_resnumbs, termini_resnames


def pdb2feats(fname: str, save=True, ff="GROMOS", fix_pdb=True) -> tuple:
    original_atom_details = get_atom_info(fname)

    if fix_pdb:
        output_pdb, chains_res, renumbered = fix_structure(fname, ff)
    else:
        chains = get_chains_from_file(fname)
        chains_res = identify_tit_sites(fname, chains, add_ser_thr=False)
        output_pdb = fname
        renumbered = {}

    termini_details = get_termini_info(fname, chains_res)

    coords, feats, anumbs, details, aindices = encode_structure(
        output_pdb, termini_details, original_atom_details, renumbered, fix_pdb
    )

    if not save:
        os.system(f"rm -f {output_pdb}")

    return coords, feats, anumbs, details, aindices, chains_res


def map_atoms_2_class(ohe_atoms: dict) -> dict:
    atom_2_class = {}
    for aclass, residues in ohe_atoms.items():
        for (residue, atypes) in residues:
            if isinstance(atypes, str):
                atom_2_class[(residue, atypes)] = aclass
            else:
                for atype in atypes:
                    atom_2_class[(residue, atype)] = aclass
    return atom_2_class


def encode_structure(
    output_pdb,
    termini_details,
    original_atom_details,
    renumbered,
    fix_pdb,
    to_include=OHE_ATOMS_GRAPH,
):
    termini_resnumbs, termini_resnames = termini_details
    original_atom_numbs, original_atom_infos = original_atom_details

    coords, feats, anumbs, details = [], [], [], []
    aindices = {}

    atom_classes = list(to_include.keys())
    atoms_2_class = map_atoms_2_class(to_include)
    atoms_to_include = atoms_2_class.keys()

    for line in iter_f(output_pdb):
        aname, anumb, resname, chain, resnumb, x, y, z = read_pdb_line(line)

        b, icode = line[16], line[26]
        if b not in (" ", "A") or icode != " ":
            continue

        if chain in termini_resnumbs and resnumb in termini_resnumbs[chain]:
            i_ter = termini_resnumbs[chain].index(resnumb)
            ter_type = "NTR" if i_ter == 0 else "CTR"

            if ter_type == "NTR" and aname in ("N", "H1", "H2", "H3"):
                resname = "NTR"
            elif aname in ("O1", "O2", "HO11", "HO21", "HO12", "HO22"):
                resname = "CTR"

        if (resname, aname) in atoms_to_include:
            res = atoms_2_class[(resname, aname)]
        elif (None, aname) in atoms_to_include:
            res = atoms_2_class[(None, aname)]
        else:
            continue

        # res = classify_atom(aname, resname) # deprecated

        if res in atom_classes:
            res_ohe = atom_classes.index(res)

            ainfo = (chain, resnumb, resname, aname, " ")

            coords.append((x, y, z))
            feats.append(res_ohe)
            anumbs.append(anumb)
            details.append(ainfo[:-1])

            if resnumb in renumbered.keys():
                old_resnumb, inscode = renumbered[resnumb]
                ainfo_search = (chain, old_resnumb, resname, aname, inscode)
            elif resname in ("NTR", "CTR"):
                old_aname = aname
                if resname == "NTR":
                    old_resname = termini_resnames[chain][0]
                elif resname == "CTR":
                    converted_atoms = {"O1": "O", "O2": "OXT"}
                    if aname in converted_atoms and fix_pdb:
                        old_aname = converted_atoms[aname]
                    old_resname = termini_resnames[chain][1]
                ainfo_search = (chain, resnumb, old_resname, old_aname, " ")
            else:
                ainfo_search = ainfo

            if ainfo_search in original_atom_infos:
                old_anumb = list(original_atom_numbs)[
                    list(original_atom_infos).index(ainfo_search)
                ]
                aindices[old_anumb] = len(anumbs) - 1
                # print(f'{line.strip()}   {res:10}   {old_anumb}')

            elif res != "H":
                warning_msg = f"WARNING: atom {anumb} in {output_pdb} will be ignored"
                print(warning_msg, line.strip())

    coords = torch.tensor(coords)
    feats = torch.tensor(feats)

    return coords, feats, anumbs, details, aindices
