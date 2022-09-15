import os
import pandas as pd
from torch import jit, stack, Tensor, tensor
import torch_cluster
import torch_sparse
from pdbmender.formats import new_pqr_line, gro2pdb
from pege.utils import pdb2feats


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

    def __init__(
        self,
        path: str,
        save_final_pdb: bool = False,
        fix_pdb: bool = True,
        device="cpu",
    ):
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

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model = jit.load(f"{cur_dir}/pege.pt").to(device)

        self.embs = model(self.feats, self.coords)

    def as_df(self) -> pd.DataFrame:
        aindices_old = list(self.aindices.keys())
        aindices_new = list(self.aindices.values())
        df_dict = {
            "anumb": self.anumbs,
            "chain": [i[0] for i in self.details],
            "resnumb": [i[1] for i in self.details],
            "res_icode": [i[4] for i in self.details],
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
            residcode = line_df["res_icode"]
            resname = line_df["resname"]
            aname = line_df["aname"]
            feat = line_df["feats"]
            anumb = line_df["anumb"]

            if residcode == "":
                residcode = " "

            new_line = new_pqr_line(
                anumb,
                aname,
                resname,
                resnumb,
                x,
                y,
                z,
                feat,
                0.0,
                chain=chain,
                icode=residcode,
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
        self,
        atom_numbers: list,
        ignore_missing: bool = False,
        show_atoms: bool = False,
        pooling: str = "avg",
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

        atom_is = self.as_df().query(f"old_anumb == @atom_numbers").index

        if ignore_missing:
            missing = self.as_df().query(
                f"old_anumb == @atom_numbers and old_numb != old_numb"
            )
            print(missing)
            raise

        if show_atoms:
            print(self.as_df().query(f"old_anumb == @atom_numbers"))

        embs = self.embs[atom_is]
        if pooling:
            embs = self.apply_pool(embs, pooling)
        return embs

    def get_all_res_embs(self, chain: str):
        embs = list(
            self.as_df()
            .query(f'chain == "{chain}" and aname == "CA" ')["embs"]
            .drop_duplicates()
            .values
        )
        all_residues = self.get_resnumbs_w_insertion_code(chain)

        return all_residues, tensor(embs)

    def get_resnumbs_w_insertion_code(self, chain):
        residues = (
            self.as_df()
            .query(f'chain == "{chain}"')[["resnumb", "res_icode"]]
            .drop_duplicates()
            .values
        )

        all_residues = []
        for resnumb, insertion_code in residues:

            if insertion_code:
                resnumb = f"{resnumb}{insertion_code}"

            all_residues.append(resnumb)
        return all_residues

    def get_all_custom_res_embs(
        self,
        chain: str,
        use_anames: list = None,
        show_atoms: bool = False,
        pooling: str = "avg",
    ):
        all_residues = self.get_resnumbs_w_insertion_code(chain)

        embs = self.get_residues(
            chain,
            all_residues,
            use_anames=use_anames,
            show_atoms=show_atoms,
            pooling=pooling,
        )
        return all_residues, embs

    def get_residues(
        self,
        chain: str,
        residue_numbers: list,
        use_anames: list = None,
        show_atoms: bool = True,
        pooling: str = "avg",
    ) -> Tensor:
        """Get the embeddings for a list of residues.

        If the argument pooling is set to None, all corresponding atom embeddings are returned.

        Parameters
        ----------
        residue_numbers: dict
            The number of the residues (in the original structure) for which to return the embeddings

        pooling : str, optional
            The type of pooling to be performed (default is avg)

        Returns
        -------
        Tensor
            a tensor of (n, 64) dimensions representing the full protein

        """
        res_embs = []
        for resnumb in residue_numbers:
            insertion_code = ""
            if isinstance(resnumb, str) and resnumb[-1].isalpha():
                insertion_code = resnumb[-1]
                resnumb = int(resnumb[:-1])

            query = f"chain == '{chain}' and resnumb == {resnumb} and res_icode == '{insertion_code}' and old_anumb == old_anumb"
            if use_anames:
                query += f" and aname in {use_anames}"

            atom_numbers = list(self.as_df().query(query)["old_anumb"])

            embs = self.get_atoms(
                atom_numbers,
                ignore_missing=False,
                show_atoms=show_atoms,
                pooling=pooling,
            )
            res_embs.append(embs)

        return stack(res_embs)

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
