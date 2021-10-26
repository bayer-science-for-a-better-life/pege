import sys

sys.path.insert(0, "../")

from pege import Pege

fname = "1cet.pdb"
protein = Pege(fname)

print(protein.asdf())
print(protein.get_protein().shape)
exit()

atom_embs = protein.get_atoms([1, 2], ignore_missing=True)
print(atom_embs.sum())

res_embs = protein.get_residues([18], ignore_missing=True)
print(res_embs.sum())
