import sys

sys.path.insert(0, "../")

from pege import Pege

# fname = "1cet_final.pdb"
# protein = Pege(fname, save_final_pdb=True, fix_pdb=False)
# protein.as_pdb(to_file="1cet_encoded.pdb")

fname = "cph_lyso.gro"
protein = Pege(fname, save_final_pdb=True, fix_pdb=False)
protein.as_pdb(to_file="cph_lyso_enconded.pdb")

df_prot = protein.as_df()

from pprint import pprint

for chain, residues in protein.chain_res.items():
    for residue in residues:
        print(f"\n{residue if not isinstance(residue, str) else residue + '-TERMINAL'}")
        pprint(protein.get_residue_titration_curve(chain, residue))
        print(protein.get_residue_taut_probs("A", residue, -5.5))

exit()

print(df_prot["feats"].describe())
print(protein.get_protein().shape)


atom_embs = protein.get_atoms([1, 2], ignore_missing=True)
print(atom_embs.sum())

res_embs = protein.get_residues([18], ignore_missing=True)
print(res_embs.sum())
