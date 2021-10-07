import sys
sys.path.insert(0, '../')

from pege import Pege

fname = "2cn0.pqr" #"1cet.pdb"
protein = Pege(fname)

print(protein.asdf())
print(protein.get_protein().shape)

atom_embs = protein.get_atoms([4359], ignore_missing=True) #[2292, 2295]
print(atom_embs.sum())

res_embs = protein.get_residues([14], ignore_missing=True)
print(res_embs.sum())
