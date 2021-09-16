import sys
sys.path.insert(0, '../')

from pege import Pege

fname = "4lzt.pdb"    
protein = Pege(fname)
atom_embs = protein.get_atoms([1000, 1001, 1010, 1011], ignore_missing=True)
print(atom_embs.shape)