import scanpy as sc

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=150,facecolor='white')

from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding

import sys
import os
import yaml
import pickle

#Input Arguments
h5ad        = sys.argv[1] #Input AnnData h5ad file
marker_yaml = sys.argv[2] #Marker genes by each phenotype
output      = sys.argv[3] #Output folder for results

#Output Paths
if not os.path.exists(output):
    os.makedirs(output)
embedding_file = os.path.join(output,"embedding.vec")
output_adata   = os.path.join(output,"genevector.h5ad")
mutual_information = os.path.join(output,"mutual_information.p")

#Get Marker Genes
with open(marker_yaml, 'r') as file:
    markers = yaml.safe_load(file)
print(markers)
print("**** Running GeneVector Workflow ****")

#Setup Objects
adata = sc.read(h5ad)
dataset = GeneVectorDataset(adata, load_expression=True, signed_mi=True)
cmps = GeneVector(dataset,
                  output_file=embedding_file,
                  c=100.,
                  emb_dimension=100)

#Generate Embeddings
cmps.train(3000,threshold=1e-7)
embed = GeneEmbedding(embedding_file, dataset, vector="average")
cembed = CellEmbedding(dataset, embed)
adata = cembed.get_adata()

cembed.cluster(adata,up_markers=markers)

pickle.dump(dataset.mi_scores,open(mutual_information,"wb"))
adata.write(output_adata)
