# GeneVector
Vector representations of gene co-expression in single cell RNAseq.

![alt text](https://github.com/nceglia/genevector/blob/main/data/framework.png?raw=true)


### Install:
```
python3 -m venv gvenv
source gvenv/bin/activate
python3 setup.py install
```

### Loading scanpy dataset into GeneVector.
```
from genevector.data import GeneVectorDataset
from genevector.model import GeneVectorTrainer
from genevector.embedding import GeneEmbedding, CellEmbedding

import scanpy as sc

dataset = GeneVectorDataset(adata, device="cuda")
```

### Training gene vectors.
```
cmps = GeneVector(dataset,
                  output_file="genes.vec",
                  emb_dimension=100,
                  batch_size=100000,
                  initial_lr=0.05,
                  device="cuda")
cmps.train(200) # run for 10 iterations
```

### Loading results.
```
gembed = GeneEmbedding("genes.vec", dataset, vector="average")
cembed = CellEmbedding(context, gembed)
```

### Compute gene similarities.
```
gembed.compute_similarities("CD8A")
```

### Batch Correct and Get Scanpy AnnData Object
```
cembed.batch_correct(column="sample")
adata = cembed.get_adata(min_dist=0.1, n_neighbors=50)
```

### Find optimal cosine threshold and identify meta-genes.
```
cosine_threshold = embed.select_cosine_threshold()
metagenes = embed.identify_metagenes(cosine=cosine_threshold)
```

### Score metagenes over cells.
```
embed.score_metagenes(adata, metagenes)
```

### Generate heatmap of metagenes and scores over a set of conditions.
```
embed.plot_metagenes_scores(adata,metagenes,"condition")
```

### Generate an average vector for a set of genes.
```
cd8tcellvec = gembed.generate_vector(["CD8A","CD8B","CD3D","CD3E","CD3G"])
```







