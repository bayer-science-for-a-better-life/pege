import torch
import os
from pdbmender import (
    mend_pdb,
    prepare_for_addHtaut,
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
    chains_res = identify_tit_sites(pdb_cleaned, chains, add_ser_thr=True)

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


def pdb2feats(fname: str, save=True, ff="GROMOS") -> tuple:
    original_atom_details = get_atom_info(fname)

    output_pdb, chains_res, renumbered = fix_structure(fname, ff)

    termini_details = get_termini_info(fname, chains_res)

    coords, feats, anumbs, details, aindices = encode_structure(
        output_pdb, termini_details, original_atom_details, renumbered
    )

    if not save:
        os.system(f"rm -f {output_pdb}")

    return coords, feats, anumbs, details, aindices


def encode_structure(output_pdb, termini_details, original_atom_details, renumbered):
    termini_resnumbs, termini_resnames = termini_details
    original_atom_numbs, original_atom_infos = original_atom_details

    coords, feats, anumbs, details = [], [], [], []
    aindices = {}

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

        if (
            aname[0] not in "NOS"
            and (resname not in AA_HS or aname not in AA_HS[resname])
        ) and aname != "CA":
            continue

        res = classify_atom(aname, resname)

        if res in OHE_ATOMS_GRAPH:
            res_ohe = OHE_ATOMS_GRAPH.index(res)

            ainfo = (chain, resnumb, resname, aname, " ")

            coords.append((x, y, z))
            feats.append(res_ohe)
            anumbs.append(anumb)
            details.append(ainfo)

            if resnumb in renumbered.keys():
                old_resnumb, inscode = renumbered[resnumb]
                ainfo_search = (chain, old_resnumb, resname, aname, inscode)
            elif resname in ("NTR", "CTR"):
                old_aname = aname
                if resname == "NTR":
                    old_resname = termini_resnames[chain][0]
                elif resname == "CTR":
                    converted_atoms = {"O1": "O", "O2": "OXT"}
                    if aname in converted_atoms:
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

    coords = torch.tensor(coords).reshape(1, -1, 3)
    feats = torch.tensor(feats).reshape(1, -1)

    return coords, feats, anumbs, details, aindices
