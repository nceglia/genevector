## Vector representations of gene co-expression in single cell RNAseq.

https://www.biorxiv.org/content/10.1101/2022.04.22.487554v1

## Installation

Install using pip
```
pip install genevector
```
or install from Github
```
python3 -m venv gvenv
source gvenv/bin/activate
python3 pip install -r requirements.txt
python3 setup.py install
```
## Tutorials (see "example" directory)

1. PBMC workflow: Identification of interferon stimulated metagene and cell type annotation.
2. TICA workflow: Cell type assignment.
3. SPECTRUM workflow: Vector arithmetic for site specific metagenes.
4. FITNESS workflow: Identifying increasing metagenes in time series.

Example data can be downloaded within Jupyter notebooks from Google Drive using *gdown*.
```
pip install gdown
```
The data is available for download directly:
[H5ads](https://drive.google.com/drive/folders/1ZRsdnlu9MSaRm4t_w_glD5XTqrY6CnIY?usp=sharing)


## GeneVector Workflow

### Loading scanpy dataset into GeneVector.
GeneVector makes use of Scanpy anndata objects and requires that the raw count data be loaded into .X matrix. It is highly recommended to subset genes using the *seurat_v3* flavor in Scanpy. The ```device="cuda"``` flag should be omitted if there is no GPU available. All downstream analysis requires a GeneVectorDataset object.

```
from genevector.data import GeneVectorDataset
from genevector.model import GeneVectorTrainer
from genevector.embedding import GeneEmbedding, CellEmbedding

import scanpy as sc

dataset = GeneVectorDataset(adata, device="cuda")
```

#### Training gene vectors.
After loading the expression, GeneVector will keep the mutual information between genes *if* a GeneVector object is instantiated (like below). This object is only required if you wish to train a model. Model training times vary depending on datasize. The 10k PBMC dataset can be trained in less than five minutes. ```emb_dimension`` sets the size of the learned gene vectors. Smaller values decrease training time, but values smaller than 50 may not provide optimal results.

```
cmps = GeneVector(dataset,
                  output_file="genes.vec",
                  emb_dimension=100,
                  threshold=1e-6,
                  device="cuda")
cmps.train(200, threshold=1e-6) # run for 200 iterations or loss delta below 1e-6.
```

#### Loading results.
After training, two vector files are produced (for input and output weights). It is recommended to take the average of both weights, but the user is left with the option if choosing a single weight matrix ("1" or "2"). The GeneEmbedding class has several analysis and visualization methods listed below.

```
gembed = GeneEmbedding("genes.vec", dataset, vector="average")
```

#### Compute gene similarities.
A pandas dataframe can be generated or a figure with a specified number of the most similar genes to any gene query in the AnnData object.
```
df = gembed.compute_similarities("CD8A")
gembed.plot_similarities("CD8A",n_genes=10)
```

#### Generate Metagenes.

```
df = gembed.compute_similarities("CD8A")
gembed.plot_similarities("CD8A",n_genes=10)
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






