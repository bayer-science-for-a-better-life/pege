import sys
sys.path.insert(0, '../')

from pege import Pege

fname = "1cet.pdb"    
protein = Pege(fname)

#print(protein.embs.shape)
#for anumb, feat in zip(protein.anumbs, protein.feats[0]):
#    print(anumb, feat)

print(protein.asdf())
print(protein.get_protein().shape)

atom_embs = protein.get_atoms([2292, 2295], ignore_missing=True)
print(atom_embs.sum())

res_embs = protein.get_residues([328], ignore_missing=True)
print(res_embs.sum())
