from torch import Tensor
from pege.egnn import model
from pege.utils import pdb2feats


class Pege:
    def __init__(self, path: str):
        """ """
        self.path = path
        self.coords, self.feats, self.anumbs, self.details = pdb2feats(path)
        self.embs = model(self.coords, self.feats).squeeze()

    def get_protein(self, pooling: str = "avg") -> Tensor:
        """ """
        protein_emb = self.embs
        if pooling:
            protein_emb = self.apply_pool(protein_emb, pooling)
        return protein_emb

    def get_atoms(
        self, atom_numbers: list, ignore_missing: bool = False, pooling: str = "avg"
    ) -> Tensor:
        """ """
        atom_is = []
        if ignore_missing:
            for i in atom_numbers:
                if i not in self.anumbs:
                    raise Exception(f"Atom {i} not present")
                a_i = self.anumbs.index(i)
                atom_is.append(a_i)
        pocket_embs = self.embs[atom_is]
        if pooling:
            pocket_embs = self.apply_pool(pocket_embs, pooling)
        return pocket_embs

    @staticmethod
    def apply_pool(t: Tensor, ptype: str) -> Tensor:
        if ptype == "avg":
            t = t.mean(dim=0)
        elif ptype == "sum":
            t = t.sum(dim=0)
        return t
