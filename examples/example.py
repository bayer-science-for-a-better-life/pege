from pege import Pege

fname = "4lzt.pdb"    
protein = Pege(fname)
pocket = protein.get_atoms([1000, 1001, 1011, 1012])
print(pocket.shape)