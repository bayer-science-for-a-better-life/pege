# Protein Enviroment Graph Embeddings (PEGE)

Protein embeddings to describe local electrostic enviroments

# Installation & Basic Usage

PEGE is installable from the Pypi repo:
```bash
python3 -m pip install pege
```

In order for the structure preprocessing to work python2 and gawk need to installed.
```bash
apt install python2 gawk
```

Pege can be used to obtain protein embeddings as well as descriptors for specific `atom_numbers` from a `pdb` file:
```python
from pege import Pege

protein = Pege(<pdb>)
protein_emb = protein.get_protein()
atoms_emb = protein.get_atoms([<atom_numbers>])
```

# Documentation
TBA

# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

# Contacts
Please submit a github issue to report bugs and to request new features. Alternatively, you may email the developer [directly](mailto:pdreis@fc.ul.pt).

