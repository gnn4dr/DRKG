# Drug Repurposing Knowledge Graph (DRKG)
Drug Repurposing Knowledge Graph (DRKG) is a comprehensive biological knowledge graph relating genes, compounds, diseases, biological processes, side effects and symptoms. DRKG includes information from six existing databases including DrugBank, Hetionet, GNBR, String, IntAct and DGIdb, and data collected from recent publications particularly related to Covid19. It includes 97,055 entities belonging to 13 entity-types; and 5,869,294 triplets belonging to 106 edge-types. These 106 edge-types show a type of interaction between one of the 17 entity-type pairs (multiple types of interactions are possible between the same entity-pair), as depicted in the figure below. It also includes a bunch of notebooks about how to explore and analysis the DRKG using statistical methodologies or using machine learning methodologies such as knowledge graph embedding.


<p align="center">
  <img src="connectivity.png" alt="DRKG schema" width="600">
  <br>
  <b>Figure</b>: Interactions in the DRKG. The number next to an edge indicates the number of relation-types for that entity-pair in DRKG.
</p>

## Download DRKG
You can directly download drkg by following commands:
```
wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
```

### DRKG dataset
The whole dataset contains three part:
 - drkg.tsv, a tsv file containg the original drkg in the format of (h, r, t) triplets.
 - embed, a folder containing the pretrained Knowledge Graph Embedding using the entire drkg.tsv as the training set. 
 - entity2src.tsv, a directory mapping entities in drkg to their original sources.

### Pretrained DRKG embedding
The DRKG mebedding is trained using TransE\_l2 model with dimention size of 400, there are four files:

 - DRKG\_TransE\_l2\_entity.npy, NumPy binary data, storing the entity embedding
 - DRKG\_TransE\_l2\_relation.npy, NumPy binary data, storing the relation embedding
 - entities.tsv, mapping from entity\_name to tentity\_id.
 - relations.tsv, mapping from relation\_name to relation\_id
 
To use the pretrained embedding, one can use np.load to load the entity embeddings and relation embeddings separately:

```
import numpy as np
entity_emb = np.load('./embed/DRKG_TransE_l2_entity.npy')
rel_emb = np.load('./embed/DRKG_TransE_l2_relation.npy')
```

## Statistics of DRKG
The type-wise distribution of the entities in DRKG and their original data-source(s) is shown in following table. 

| Entity type         | Drugbank | GNBR  | Hetionet | STRING | IntAct | DGIdb | Bibliography | Total Entities |
|:--------------------|---------:|------:|---------:|-------:|-------:|------:|-------------:|---------------:|
| Anatomy             | \-       | \-    | 400      | \-     | \-     | \-    | \-           | 400            |
| Atc                 | 4,048     | \-    | \-       | \-     | \-     | \-    | \-           | 4,048           |
| Biological Process  | \-       | \-    | 11,381    | \-     | \-     | \-    | \-           | 11,381          |
| Cellular Component  | \-       | \-    | 1,391     | \-     | \-     | \-    | \-           | 1,391           |
| Compound            | 9,708     | 11,961 | 1,538     | \-     | 153    | 6,348  | 6,250         | 24,313          |
| Disease             | \-       | 4,747  | 257      | \-     | \-     | \-    | 33           | 4,920           |
| Gene                | 4,973     | 27,111 | 19,145    | 18,316  | 16,321  | 2,551  | 3,181         | 39,220          |
| Molecular Function  | \-       | \-    | 2,884     | \-     | \-     | \-    | \-           | 2,884           |
| Pathway             | \-       | \-    | 1,822     | \-     | \-     | \-    | \-           | 1,822           |
| Pharmacologic Class | \-       | \-    | 345      | \-     | \-     | \-    | \-           | 345            |
| Side Effect         | \-       | \-    | 5,701     | \-     | \-     | \-    | \-           | 5,701           |
| Symptom             | \-       | \-    | 415      | \-     | \-     | \-    | \-           | 415            |
| Tax                 | \-       | 215   | \-       | \-     | \-     | \-    | \-           | 215            |
| Total               | 18,729    | 44,034 | 45,279    | 18,316  | 16,474  | 8,899  | 9,464         | 97,055          |


The following table shows the number of triplets between different entity-type pairs in DRKG for DRKG and various datasources.

| Entity\-type pair                     | Drugbank | GNBR   | Hetionet | STRING  | IntAct | DGIdb | Bibliography | Total interactions |
|:--------------------------------------|-----------:|-------:|---------:|--------:|-------:|------:|-------------:|-------------------:|
| \(Gene, Gene\)                    | \-         | 66,722  | 474,526   | 1,496,708 | 254,346 | \-    | 58,629        | 2,350,931            |
| \(Compound, Gene\)                | 24,801      | 80,803  | 51,429    | \-      | 1,805   | 26,290 | 25,666        | 210,794             |
| \(Disease, Gene\)                 | \-         | 95,400  | 27,977    | \-      | \-     | \-    | 461          | 123,838             |
| \(Atc, Compound\)                 | 15,750      | \-     | \-       | \-      | \-     | \-    | \-           | 15,750              |
| \(Compound, Compound\)            | 1,379,271    | \-     | 6,486     | \-      | \-     | \-    | \-           | 1,385,757            |
| \(Compound, Disease\)             | \-         | 77,782  | 1,145     | \-      | \-     | \-    | \-           | 78,927              |
| \(Gene, Tax\)                     | \-         | 14,663  | \-       | \-      | \-     | \-    | \-           | 14,663              |
| \(Biological Process, Gene\)      | \-         | \-     | 559,504   | \-      | \-     | \-    | \-           | 559,504             |
| \(Disease, Symptom\)              | \-         | \-     | 3,357     | \-      | \-     | \-    | \-           | 3,357               |
| \(Anatomy, Disease\)              | \-         | \-     | 3,602     | \-      | \-     | \-    | \-           | 3,602               |
| \(Disease, Disease\)              | \-         | \-     | 543      | \-      | \-     | \-    | \-           | 543                |
| \(Anatomy, Gene\)                 | \-         | \-     | 726,495   | \-      | \-     | \-    | \-           | 726,495             |
| \(Gene, Molecular Function\)      | \-         | \-     | 97,222    | \-      | \-     | \-    | \-           | 97,222              |
| \(Compound, Pharmacologic Class\) | \-         | \-     | 1,029     | \-      | \-     | \-    | \-           | 1,029               |
| \(Cellular Component, Gene\)      | \-         | \-     | 73,566    | \-      | \-     | \-    | \-           | 73,566              |
| \(Gene, Pathway\)                 | \-         | \-     | 84,372    | \-      | \-     | \-    | \-           | 84,372              |
| \(Compound, Side Effect\)         | \-         | \-     | 138,944   | \-      | \-     | \-    | \-           | 138,944             |
| Total                                 | 1,419,822    | 335,370 | 2,250,197  | 1,496,708 | 256,151 | 26,290 | 84,756        | 5,869,294            |

## Installation
Before using notebooks here, you need to install some dependencies:

### Install PyTorch
Currently all notebooks use PyTorch as Deep Learning backend. For install other version of pytorch please goto [Install PyTorch](https://pytorch.org/)
```
sudo pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install DGL (Optional)
If you want to analyze DRKG with DGL with notebooks at [drkg-with-dgl], you need to install DGL package.
Currently we use the newest stable version of DGL. For install other version of DGL please goto [Install DGL](https://docs.dgl.ai/en/latest/install/index.html)
```
sudo pip3 install dgl-cu101
```

### Install DGL-KE (Optional)
If you want to training the model with notebooks (e.g., using Train_embeddings.ipynb or Edge_score_analysis.ipynb) at [knowledge-graph-embedding-based-analysis-of-drkg], you need to install DGL as in [install-dgl-optional] and install DGL-KE package here.
Currently we use the newest stable version of DGL-KE. DGL-KE can work with DGL > 0.4.3 (either CPU or GPU)
```
sudo pip3 install dglke
```

## DRKG with DGL
We provide a notebook, with example of using DRKG with Deep Graph Library (DGL).

The following notebook provides an example of building a heterograph from DRKG in DGL; and some examples of queries on the DGL heterograph:
 - [loading_drkg_in_dgl.ipynb](drkg_with_dgl/loading_drkg_in_dgl.ipynb)
 
## Basic Graph Analysis of DRKG
We analyzed directly the structure of the extracted DRKG. Since the datasources may contain related information, we want to verify that combining the edge information from different sources is meaningful.

To evaluate the structural similarity among a pair of relation types we compute their Jaccard similarity coefficient and the overlap among the two edge types via the overlap coeffcient. This analysis is given in
 - [Jaccard_scores_among_all_edge_types_in_DRKG.ipynb](raw_graph_analysis/Jaccard_scores_among_all_edge_types_in_DRKG.ipynb)

## Knowledge Graph Embedding Based Analysis of DRKG
We analyze the extracted DRKG by learning a TransE KGE model that utilizes the ![$\ell_2$](https://render.githubusercontent.com/render/math?math=%24%5Cell_2%24) distance. As DRKG combines information from different data sources, we want to verify that meaningful entity and relation embeddings can be generated using knowledge graph embedding technology.

We split the edge triplets in training, validation and test sets as follows 90%, 5%, and 5% and train the KGE model as shown in following notebook:
- [Train_embeddings.ipynb](embedding_analysis/Train_embeddings.ipynb)

Finally, we obtain the entity and relation embeddings for the DRKG. We can do various embedding based analysis as provided in the following notebooks:
 - [Relation_similarity_analysis.ipynb](embedding_analysis/Relation_similarity_analysis.ipynb), analyzing the generate relation embedding similarity.
 - [Entity_similarity_analysis.ipynb](embedding_analysis/Entity_similarity_analysis.ipynb), analyzing the generate entity embedding similarity.
 - [Edge_score_analysis.ipynb](embedding_analysis/Edge_score_analysis.ipynb), evaluating whether the learned KGE model can predict the edges of DRGK
 - [Edge_similarity_based_on_link_recommendation_results.ipynb](embedding_analysis/Edge_similarity_based_on_link_recommendation_results.ipynb), evaluating how similar are the predicted links among different relation types.
 
 ## Licence
This project is licensed under the Apache-2.0 License. However, the DRKG integrates data from many resources and users should consider the licensing of each source (see this [table](https://github.com/shuix007/COVID-19-KG/blob/master/licenses/Readme.md)) . We apply a license attribute on a per node and per edge basis for sources with defined licenses. 


```python

```
