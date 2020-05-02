# Information about Compounds from DrugBank

The KG currently has 9708 compounds from DrugBank. DB05697, DB06517, and DB15351 have been deprecated from DrugBank 
in the sense that there are no associated pages. DB02709 is a duplicate of DB05073 and should be merged

For the rest 9704 ids, 8968 are characterized as "small-molecule" typed drugs in DrugBank and 736 are 
characterized as "Biotech" typed drugs in DrugBank. A full list of them are separately stored in `drugbank_smiles.txt` 
and `drugbank_biotech.txt`.

For the 8968 "small-molecule" typed drugs, we manage to extract SMILES 
(The simplified molecular-input line-entry system) for 8646 of them from DrugBank, PubChem, KEGG, ChEMBL, and 
ChemSpider. The SMILES can be found in `drugbank_smiles.txt`. For the rest "small-molecule" typed drugs, SMILES is not 
available either because their structural information is not available or because they are substances of multiple 
ingredients.

We also include the weight of compounds available in DrugBank in `drugbank_weight.txt`. For compounds whose weight 
information is missing, we use -1 for a placeholder. The weight information is available for 7158 compounds 
and missing for 2547 compounds.
