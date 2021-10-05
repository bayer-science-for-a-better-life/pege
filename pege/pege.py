from torch import Tensor
from pege.egnn import model
from pege.utils import pdb2feats
import pandas as pd

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

    def __init__(self, path: str):
        """
        Parameters
        ----------
        path : str
            The protein PDB file path
        """
        self.path = path
        self.coords, self.feats, self.anumbs, self.details, self.aindices = pdb2feats(path, save=False)
        self.embs = model(self.coords, self.feats).squeeze()

    def asdf(self):    
        aindices_old = list(self.aindices.keys())
        aindices_new = list(self.aindices.values())
        df_dict = {
            'anumb': self.anumbs,
            'details': self.details,
            'resnumbs': [i[1] for i in self.details],
            'embs': self.embs.detach().numpy().tolist(),
            'feats': self.feats[0],
            'coords': self.coords[0].tolist(),
            'old_anumb': [aindices_old[aindices_new.index(i)] if i in aindices_new else None for i in range(len(self.anumbs))]
        }
        return pd.DataFrame(df_dict)

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
                if not ignore_missing: raise Exception(f"Atom {anumb} not present")
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
        atom_numbers = list(self.asdf().query("resnumbs == @residue_numbers")['old_anumb'])
        return self.get_atoms(atom_numbers, ignore_missing=ignore_missing, pooling=pooling)

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
