![alt text](https://github.com/nceglia/genevector/blob/main/logo.png?raw=true)
## Vector representations of gene co-expression in single cell RNAseq.

![alt text](https://github.com/nceglia/genevector/blob/main/framework.png?raw=true)

https://www.biorxiv.org/content/10.1101/2022.04.22.487554v1

### Install:
Make sure torch is installed: https://pytorch.org/get-started/locally/
```
python3 -m venv gvenv
source gvenv/bin/activate
python3 setup.py install
```

## For a workable example: see PBMC notebook in /example.

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
                  initial_lr=0.1,
                  device="cuda")
cmps.train(200) # run for 200 iterations
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
adata = cembed.get_adata()
```

### Get Gene Embedding and Find Metagenes
```
gdata = embed.get_adata()
metagenes = embed.get_metagenes(gdata)
```






