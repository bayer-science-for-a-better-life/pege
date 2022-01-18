from torch import Tensor, flip, clamp
import random

# from pege.egnn import model
from pege.egnn.model import model
from pege.utils import pdb2feats
from pege.constants import pH_scale
import pandas as pd
from pdbmender.formats import new_pqr_line, gro2pdb


class Pege:
    """
    Protein Enviroment Graph Embeddings

    Attributes
    ----------
    path : str
        the protein PDB file path
    coords: Tensor[1, n, 3] where n is the number of atoms
        the coordinates of the all the atoms
    feats: Tensor[1, n] where n is the number of atoms
        the features (atom classes) of all the atoms
    anumbs: list
        the atom numbers
    details: list
        the atom details
    embs: Tensor[n, 64] where n is the number of atoms
        the atom embeddings

    Methods
    -------
    get_protein(pooling=avg)
        Returns
    """

    def __init__(self, path: str, save_final_pdb: bool = False, fix_pdb: bool = True):
        """
        Parameters
        ----------
        path : str
            The protein PDB file path
        """
        self.path = path

        if path.endswith(".gro"):
            f_in = path.replace(".gro", ".pdb")
            gro2pdb(path, f_in)
            path = f_in

        (
            self.coords,
            self.feats,
            self.anumbs,
            self.details,
            self.aindices,
            self.chain_res,
        ) = pdb2feats(path, save=save_final_pdb, fix_pdb=fix_pdb)
        self.natoms = self.coords.shape[1]
        self.hindices = (self.feats == 0).nonzero().squeeze()
        self.embs, self.h_probs = model(self.coords, self.feats, self.hindices)

    def as_df(self) -> pd.DataFrame:
        aindices_old = list(self.aindices.keys())
        aindices_new = list(self.aindices.values())
        df_dict = {
            "anumb": self.anumbs,
            "chain": [i[0] for i in self.details],
            "resnumb": [i[1] for i in self.details],
            "resname": [i[2] for i in self.details],
            "aname": [i[3] for i in self.details],
            "embs": self.embs.detach().numpy().tolist(),
            "feats": self.feats,
            "coords": self.coords.tolist(),
            "old_anumb": [
                aindices_old[aindices_new.index(i)] if i in aindices_new else None
                for i in range(len(self.anumbs))
            ],
        }
        return pd.DataFrame(df_dict)

    def as_pdb(self, to_file: bool = None) -> str:
        pdb = []
        for i, line_df in self.as_df().iterrows():
            x, y, z = line_df["coords"]
            chain = line_df["chain"]
            resnumb = line_df["resnumb"]
            resname = line_df["resname"]
            aname = line_df["aname"]
            feat = line_df["feats"]
            anumb = line_df["anumb"]

            new_line = new_pqr_line(
                anumb, aname, resname, resnumb, x, y, z, feat, 0.0, chain=chain
            )
            pdb.append(new_line)

        pdb_content = "".join(pdb)
        if to_file:
            with open(to_file, "w") as f:
                f.write(pdb_content)

        return pdb_content

    def get_protein(self, pooling: str = "avg") -> Tensor:
        """Get the protein embedding.

        If the argument pooling is set to None, all atom embeddings are returned.

        Parameters
        ----------
        pooling : str, optional
            The type of pooling to be performed (default is avg)

        Returns
        -------
        Tensor
            a tensor of (n) dimensions representing the full protein
        """
        protein_emb = self.embs
        if pooling:
            protein_emb = self.apply_pool(protein_emb, pooling)
        return protein_emb

    def get_atoms(
        self, atom_numbers: list, ignore_missing: bool = False, pooling: str = "avg"
    ) -> Tensor:
        """Get the embeddings for a list of atoms.

        If the argument pooling is set to None, all corresponding atom embeddings are returned.

        Parameters
        ----------
        atom_numbers: list
            The serial number of the atoms (in the original structure) for which to return the embeddings

        ignore_missing: bool, optional
            Controls whether an error is raised in case some atoms are not found (default is False)
        pooling : str, optional
            The type of pooling to be performed (default is avg)

        Returns
        -------
        Tensor
            a tensor of (n, 64) dimensions representing the full protein

        Raises
        ------
        Exception
            If some of the atoms present in atom_numbers are not found.
            Can be turned off by using ignore_missing.
        """
        atom_is = []
        for anumb in atom_numbers:
            if anumb not in self.aindices:
                if not ignore_missing:
                    raise Exception(f"Atom {anumb} not present")
            else:
                a_i = self.aindices[anumb]
                atom_is.append(a_i)
        if not atom_is:
            return None
        pocket_embs = self.embs[atom_is]
        if pooling:
            pocket_embs = self.apply_pool(pocket_embs, pooling)
        return pocket_embs

    def get_residues(
        self, residue_numbers: list, ignore_missing: bool = False, pooling: str = "avg"
    ) -> Tensor:
        """Get the embeddings for a list of residues.

        If the argument pooling is set to None, all corresponding atom embeddings are returned.

        Parameters
        ----------
        residue_numbers: list
            The number of the residues (in the original structure) for which to return the embeddings

        ignore_missing: bool, optional
            Controls whether an error is raised in case some atoms are not found (default is False)
        pooling : str, optional
            The type of pooling to be performed (default is avg)

        Returns
        -------
        Tensor
            a tensor of (n, 64) dimensions representing the full protein

        Raises
        ------
        Exception
            If some of the atoms present in atom_numbers are not found.
            Can be turned off by using ignore_missing.
        """
        atom_numbers = list(
            self.as_df().query("resnumb == @residue_numbers")["old_anumb"]
        )
        return self.get_atoms(
            atom_numbers, ignore_missing=ignore_missing, pooling=pooling
        )

    def get_residue_atoms(self, chain, resnumb):
        if isinstance(resnumb, str):
            if resnumb[-1] in "NC":
                resname = {"N": "NTR", "C": "CTR"}[resnumb[-1]]
                resnumb = int(resnumb[:-1])
                termini_condition = f"resname == '{resname}' and resnumb == {resnumb}"
            else:
                resnumb = int(resnumb)
                termini_condition = (
                    f"resname in ('NTR', 'CTR') and resnumb == {resnumb}"
                )
        else:
            termini_condition = (
                f"resname not in ('NTR', 'CTR') and resnumb == {resnumb}"
            )

        atoms = self.as_df().query(
            f"chain == '{chain}' and {termini_condition} and feats == 0"
        )
        return atoms

    def get_residue_titration_curve(self, chain, resnumb):
        atoms = self.get_residue_atoms(chain, resnumb)

        if len(atoms) == 0:
            return None

        hs = list(atoms["anumb"].index)
        hs_i = [list(self.hindices).index(i) for i in hs]

        resname = atoms["resname"].values[0]

        tit_curve = {}
        for i, pH in enumerate(pH_scale):
            taut_probs = self.h_probs[hs_i, i]
            _, prot_avg = self.fix_taut_probs(taut_probs, resname)
            tit_curve[pH] = prot_avg

        return tit_curve

    def get_residue_taut_probs(self, chain, resnumb, pH):
        if pH not in pH_scale:
            raise Exception("pH not valid.")

        atoms = self.get_residue_atoms(chain, resnumb)

        if len(atoms) == 0:
            return None, None, None

        hs = list(atoms["anumb"].index)
        hs_i = [list(self.hindices).index(i) for i in hs]
        resname = atoms["resname"].values[0]

        pH_i = pH_scale.index(pH)
        taut_probs = self.h_probs[hs_i, pH_i]

        # assert order of tautomers

        taut_probs, prot_avg = self.fix_taut_probs(taut_probs, resname)

        tauts = list(range(len(taut_probs)))
        prot_state = random.choices(tauts, taut_probs)[0]

        return (
            prot_state,
            prot_avg,
            taut_probs,
        )

    @staticmethod
    def fix_taut_probs(taut_probs, resname):
        if resname in ("NTR", "LYS", "HIS"):
            tmp = 1 - taut_probs
            taut_probs = flip(tmp, [0])

        if sum(taut_probs) > 1:
            taut_probs = (taut_probs / taut_probs.sum()) - 0.0000001

        taut_probs = taut_probs.detach().numpy().tolist()
        taut_probs.append(1 - sum(taut_probs))

        if resname in ("NTR", "LYS", "HIS"):
            prot_avg = taut_probs[-1]
        else:
            prot_avg = 1 - taut_probs[-1]

        return taut_probs, prot_avg

    @staticmethod
    def apply_pool(t: Tensor, ptype: str) -> Tensor:
        """Applies a type of pooling

        Parameters
        ----------
        t : Tensor
            the tensor to be pooled
        ptype : str
            the pooling type to be performed

        Returns
        -------
        Tensor
            the pooled tensor
        """
        if ptype == "avg":
            t = t.mean(dim=0)
        elif ptype == "sum":
            t = t.sum(dim=0)
        return t


# TODO: check S-S how to deal with bridges
