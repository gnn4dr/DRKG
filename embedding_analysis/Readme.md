# Knowledge Graph Embedding Based Analysis of DRKG
We analyze the extracted DRKG by learning a TransE KGE model that utilizes the $\ell_2$ distance. As DRKG combines information from different data sources, we want to verify that meaningful entity and relation embeddings can be generated using knowledge graph embedding technology.

## Train Knowledge Graph Embedding
Before doing the analysis, we need to train the knowledge graph embedding first. Here, we split the edge triplets in training, validation and test sets as follows 90%, 5%, and 5% and train the KGE model as shown in following notebook:

 - [Train_embeddings.ipynb](Train_embeddings.ipynb)
 
## Analyze the Relation Embedding Similarity
We analyze the relation embedding similarity in [Relation_similarity_analysis.ipynb](Relation_similarity_analysis.ipynb). We first use t-SNE to map relation embedding to a 2D space to show the relation embedding distribution and then plot the pair-wise similarity between different edge relation types.

## Analyze the Entity Embedding Similarity
We analyze the entity embedding similarity in [Entity_similarity_analysis.ipynb](Entity_similarity_analysis.ipynb). We first use t-SNE to map relation embedding to a 2D space to show the entity embedding distribution, then plot the embedding distribution of entities of Drugbank drugs, and finally we show the pair-wise similarity between different entities.

## Analyze Edge Score
We analyze whether the learned KGE model can predict the edges of DRGK in [Edge_score_analysis.ipynb](Edge_score_analysis.ipynb). In order to avoid the possible bias of over-fitting the triplets in the training set, we split the whole DRKG into 10 equal folds and train 10 KGE models by picking each fold as the test set and the rest other nine folds are the training set. Following this, the score for each triplet is calculated while this triplet was in the test set. Then we show how edge scores distribute.

## Analyze Link Type Recommendation Similarity
We analyze how similar are the predicted links among different relation types in [Edge_similarity_based_on_link_recommendation_results.ipynb](Edge_similarity_based_on_link_recommendation_results.ipynb). We evaluate how similar are the predicted links among different relation types. This task examines the similarity across relation types for the link prediction task. For seed node $n^{k}_i$ we find the top 10 neighbors under relation $r_j$ with the highest link prediction score. Next, we repeat the same prediction for relation $r_{j'}$ and calculate the Jaccard similarity coefficient among the predicted sets of top 10 neighbors for $r_j$ and $r_{j'}$.