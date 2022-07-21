![alt text](https://github.com/nceglia/genevector/blob/main/logo.png?raw=true)
## Vector representations of gene co-expression in single cell RNAseq.

![alt text](https://github.com/nceglia/genevector/blob/main/framework.png?raw=true | width=100)

https://www.biorxiv.org/content/10.1101/2022.04.22.487554v1

### Install:

Prerequisite software:
Torch: https://pytorch.org/get-started/locally/ 
igraph: https://igraph.org/python/doc/tutorial/install.html (`brew install igraph` on Mac)
igraph-python: `conda install -c conda-forge python-igraph`

```
python3 -m venv gvenv
source gvenv/bin/activate
python3 pip install -r requirements.txt
python3 setup.py install
```

Software has been tested on Macbook (M1 Pro/M1/Intel)

Install time: < 20 min (dependent on Torch).

### Example Tutorial

PBMC workflow with identification of interferon stimulated metagene and cell type annotation is available in /example.

Launch `jupyter notebook` inside /example directory after installing GeneVector.

Runtime: ~2 min for data loading and ~8 min for training (Macbook M1 Pro)


### Basics

#### Loading scanpy dataset into GeneVector.
```
from genevector.data import GeneVectorDataset
from genevector.model import GeneVectorTrainer
from genevector.embedding import GeneEmbedding, CellEmbedding

import scanpy as sc

dataset = GeneVectorDataset(adata, device="cuda")
```

#### Training gene vectors.
```
cmps = GeneVector(dataset,
                  output_file="genes.vec",
                  emb_dimension=100,
                  initial_lr=0.1,
                  threshold=1e-6,
                  device="cuda")
cmps.train(200) # run for 200 iterations or loss delta below 1e-6.
```

#### Loading results.
```
gembed = GeneEmbedding("genes.vec", dataset, vector="average")
cembed = CellEmbedding(dataset, gembed)
```

#### Compute gene similarities.
```
gembed.compute_similarities("CD8A")
```

#### Batch Correct and Get Scanpy AnnData Object
```
cembed.batch_correct(column="sample")
adata = cembed.get_adata()
```

#### Get Gene Embedding and Find Metagenes
```
gdata = embed.get_adata()
metagenes = embed.get_metagenes(gdata)
```






