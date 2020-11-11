# compass
Vector representations of gene co-expression in single cell RNAseq.

### Install:
```
python3 -m venv cenv
python3 setup.py install
```

### Loading scanpy datasets into Compass.
```
from compass.data import Context, CompassDataset
context = Context.build(adata)
dataset = CompassDataset(context)
```

### Training gene vectors.
```
from compass.model import CompassTrainer
cmps = CompassTrainer(dataset,output_file="genes.vec", batch_size=200)
cmps.train(10) # run for 10 iterations
```

### Loading results.
```
from compass.embedding import GeneEmbedding, CellEmbedding
gembed = GeneEmbedding("genes.vec", context)
cembed = CellEmbedding(context, gembed)
```

### Compute gene similarities.
```
gembed.compute_similarities("CD8A")
```

### Clustering gene vectors.
```
gene_clusters = gembed.cluster()
```

### Build gene t-SNE embedding and label some genes.
```
gembed.plot(gene_clusters,labels=["C1QC","C1QA","TYROBP"])
```

### Find most representative genes for each cluster based on cosine distance.
```
cluster_definitions = gembed.cluster_definitions(gene_clusters)
```

### Pandas data frame of most representative genes for each cluster.
```
gene_df = gembed.cluster_definitions_as_df(cluster_definitions,top_n=10)
```

### Relabel cluster in results.
```
gembed.relabel_cluster(cluster_definitions, gene_clusters, 6, "Cell Cycle")
```

### Generate an average vector for a set of genes.
```
cd8tcellvec = gembed.generate_vector(["CD8A","CD8B","CD3D","CD3E","CD3G"])
```

### Plot a cosine similarity matrix for a set of genes.
```
gembed.plot_similarity_matrix(["CD8A","CD8B","CD3D","CD3E","CD3G"], png="cd8_matrix.png")
```

### Plot a networkx graph of similarities between a set of genes.
```
plot_similarity_network(["CD8A","CD8B","CD3D","CD3E","CD3G"], png="cd8_graph.png"
```

### Build cell t-SNE embedding with cell type labels.
```
tsne_embedding = cembed.plot(column="cell_type")
```

### Reuse t-SNE embedding and plot another label.
```
cembed.plot(column="batch_label",pcs=tsne_embedding)
```

### Find K cell clusters.
```
clusters = cembed.cluster(k=6)
```

### Locally batch correct on clusters using column as batch label.
```
cembed.batch_correct(column="batch_label",clusters=clusters)
```

### Regenerate corrected t-SNE
```
corrected_tsne_embedding = cembed.plot(column="batchlb")
```

### Plot cosine distance for each cell to a given vector.
```
cembed.plot_distance(cd8tcellvec,png="highlightcd8cells.png")
```






