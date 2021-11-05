import sys

sys.path.insert(0, "../")

from pege import Pege

fname = "1cet.pdb"
protein = Pege(fname, save_final_pdb=True)

df_prot = protein.as_df()

print(df_prot["feats"].describe())
print(protein.get_protein().shape)

protein.as_pdb(to_file="1cet_encoded.pdb")

atom_embs = protein.get_atoms([1, 2], ignore_missing=True)
print(atom_embs.sum())

res_embs = protein.get_residues([18], ignore_missing=True)
print(res_embs.sum())
