![alt text](https://github.com/nceglia/genevector/blob/main/logo.png?raw=true)
![alt text](framework.png?raw=true)

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
The data is available for download directly
[here](https://drive.google.com/drive/folders/1ZRsdnlu9MSaRm4t_w_glD5XTqrY6CnIY?usp=sharing).

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

### Training GeneVector
After loading the expression, creating a GeneVector object will compute the mutual information between genes (can take up to 15 min for a dataset of 250k cells). This object is only required if you wish to train a model. Model training times vary depending on datasize. The 10k PBMC dataset can be trained in less than five minutes. ```emb_dimension``` sets the size of the learned gene vectors. Smaller values decrease training time, but values smaller than 50 may not provide optimal results.

```
cmps = GeneVector(dataset,
                  output_file="genes.vec",
                  emb_dimension=100,
                  threshold=1e-6,
                  device="cuda")
cmps.train(1000, threshold=1e-6) # run for 1000 iterations or loss delta below 1e-6.
```

To understand model convergence, a loss plot by epoch can be generated.

```
cmps.plot()
```

### Loading Gene Embedding
After training, two vector files are produced (for input and output weights). It is recommended to take the average of both weights ```vector="average"```). The GeneEmbedding class has several important analysis and visualization methods listed below.

```
gembed = GeneEmbedding("genes.vec", dataset, vector="average")
```

#### 1. Computing gene similarities
A pandas dataframe can be generated using ```compute_similarities``` that includes the most similar genes and their cosine similarities for a given gene query. A barplot figure with a specified number of the most similar genes can be generated using ```plots_similarities```.

```
df = gembed.compute_similarities("CD8A")
gembed.plot_similarities("CD8A",n_genes=10)
```

#### 2. Generating Metagenes
```get_adata``` produces and AnnData object that houses the gene embedding. This allows the use of Scanpy and AnnData visualization functions. The resolution parameter is passed directly to ```sc.tl.leiden``` to cluster the co-expression graph. ```get_metagenes``` returns a dictionary that stores each metagene as a list associated with an id. For a given id, the metagene can be visualized on a UMAP embedding using the ```plot_metagene``` function.

```
gdata = embed.get_adata(resolution=40)
metagenes = embed.get_metagenes(gdata)
embed.plot_metagene(gdata, mg=isg_metagene)
```

### Loading the Cell Embedding

Using the GeneEmbedding object and the GeneVectorDataset object, a CellEmbedding object can be instantiated and used to produce a Scanpy AnnData object with ```get_adata```. This method stores cell embedding under ```X_genevector``` in layers and generates a UMAP embedding using Scanpy. Scanpy functionality can be used to visualize UMAPS (```sc.pl.umap```). 

```
cembed = CellEmbedding(dataset, embed)
adata = cembed.get_adata()
```

The cell embedding can be batch corrected using ```cembed.batch_correct```. The user is required to select a valid column present in the *obs* dataframe and specify a reference label. This is a very fast operation.

```
cembed.batch_correct(column="sample",reference="control")
adata = cembed.get_adata()
```

#### Scoring Cells by Metagene

To score expression for each metagene against all cells, we can call the GeneEmbedding function ```score_metagenes``` with the cell-based AnnData object. To plot a heatmap of all metagenes over a set of cell labels, use the ```plot_metagenes_scores``` function. Metagenes are scored with the Scanpy ```sc.tl.score_genes``` function.

```
embed.score_metagenes(adata, metagenes)
embed.plot_metagenes_scores(adata,mgs,"cell_type")
```

#### Performing Cell Type Assignment

Using a dictionary of cell type annotations to marker genes, each cell can be classified using the CellEmbedding function ```phenotype_probability```. This function returns a new annotated AnnData object, where the resulting classification can be found in ```.obs["genevector"]``` (the user can also supply a column name using ```column=```). A separate column in the obs dataframe is created to hold the pseudo-probabilities for each cell type. These probabilties can be shown on a UMAP using standard the Scanpy function ```sc.pl.umap```.

```
markers = dict()
markers["T Cell"] = ["CD3D","CD3G","CD3E","TRAC","IL32","CD2"]
markers["B/Plasma"] = ["CD79A","CD79B","MZB1","CD19","BANK1"]
markers["Myeloid"] = ["LYZ","CST3","AIF1","CD68","C1QA","C1QB","C1QC"]

annotated_adata = cembed.phenotype_probability(adata,markers)

prob_cols = [x for x in annotated_adata.obs.columns.tolist() if "Pseudo-probability" in x]
sc.pl.umap(annotated_adata,color=prob_cols+["genevector"],size=25)
```

##### *All additional analyses described in the manuscript, including comparisons to LDVAE and CellAssign, can be found in Jupyter notebooks in the examples directory. Results were computed for v0.0.1.*
