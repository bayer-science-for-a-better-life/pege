import sys

sys.path.insert(0, "../")

from pege import Pege

fname = "1cet.pdb"
protein = Pege(fname, save_final_pdb=True, fix_pdb=True)
# protein.as_pdb(to_file="1cet_encoded.pdb")

df_prot = protein.as_df()


from IPython import embed
embed()
exit()

# print(df_prot.query("resnumb == 73 and res_icode == 'A'"))

protein_emb = protein.get_protein()
print(protein_emb.shape)

all_res_embs = protein.get_all_res_embs(chain="A")
print(len(all_res_embs[0]))
print(all_res_embs[1].shape)

# all_res_embs_custom = protein.get_all_custom_res_embs(chain="A", use_anames=["CA"])
# print(all_res_embs_custom.shape)

res_embs = protein.get_residues(
    chain="A", residue_numbers=[18, 20, "73A"], use_anames=["CA", "CB", "NZ"]
)
print(res_embs.shape)

atom_embs = protein.get_atoms([1, 2], ignore_missing=False, pooling=None)
print(atom_embs.shape)
