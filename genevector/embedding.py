"""Gene and cell embedding classes for downstream analysis and visualization."""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import tqdm
import scanpy as sc
import networkx as nx
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from scipy.spatial import distance
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix
import numpy as np
import operator
import collections
import os   
import pandas as pd
import gc
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class GeneEmbedding(object):

    """
    This class provides an interface to the gene embedding, which can be used for tasks such as similarity computation, visualization, etc.

    :param embedding_file: Specifies the path to a set of .vec files generated for model training.
    :type embedding_file: str
    :param dataset: The GeneVectorDataset object that was constructed from the original AnnData object.
    :type dataset: :class:'genevector.dataset.GeneVectorDataset'
    :param vector: Specifies if using the first set of weights ("1"), the second set of weights ("2"), or the average ("average").
    :type vector: str
    """

    def __init__(self, embedding_file, dataset = None, vector="average"):
        """Constructor method
        """
        if vector not in ("1","2","average"):
            raise ValueError("Select the weight vector from: ('1','2','average')")
        if vector == "average":
            print("Loading average of 1st and 2nd weights.")
            avg_embedding = embedding_file.replace(".vec","_avg.vec")
            secondary_weights = embedding_file.replace(".vec","2.vec")
            GeneEmbedding.average_vector_results(embedding_file,secondary_weights,avg_embedding)
            self.embeddings = self.read_embedding(avg_embedding)
        elif vector == "1":
            print("Loading first weights.")
            self.embeddings = self.read_embedding(embedding_file)
        elif vector == "2":
            print("Loading second weights.")
            secondary_weights = embedding_file.replace(".vec","2.vec")
            self.embeddings = self.read_embedding(secondary_weights)
        self.vector = []
        if dataset is not None:
            self.context = dataset.data
        self.context = dataset.data
        self.embedding_file = embedding_file
        self.vector = []
        self.genes = []
        for gene in tqdm.tqdm(self.embeddings.keys()):
            self.vector.append(self.embeddings[gene])
            self.genes.append(gene)

    def read_embedding(self, filename):
        """Read a .vec embedding file into a dict of gene -> vector.

        Parameters
        ----------
        filename : str
            Path to .vec file (first line: dimensions, remaining: gene vectors).

        Returns
        -------
        dict
            Mapping from gene symbol to numpy array.
        """
        embedding = dict()
        lines = open(filename,"r").read().splitlines()[1:]
        for line in lines:
            vector = line.split()
            gene = vector.pop(0)
            embedding[gene] = numpy.array([float(x) for x in vector])
        return embedding

    def get_adata(self, resolution=20.):
        """
        This method returns the AnnData object that contains the gene embedding with leiden clusters for metagenes, the neighbors graph, and the UMAP embedding.

        :param resolution: The resolution to pass to the sc.tl.leiden function.
        :type resolution: float
        :return: An AnnData object with metagenes stored in 'leiden' for the provided resolution, the neighbors graph, and UMAP embedding.
        :rtype: AnnData
        """

        mat = numpy.array(self.vector)
        numpy.savetxt(".tmp.txt",mat)
        gdata = sc.read_text(".tmp.txt")
        os.remove(".tmp.txt")
        gdata.obs.index = self.genes
        sc.pp.neighbors(gdata,use_rep="X")
        sc.tl.leiden(gdata,resolution=resolution)
        sc.tl.umap(gdata)
        return gdata

    def plot_similarities(self, gene, n_genes=10, save=None):
        """
        Plot a horizontal bar plot of cosine similarity of the most similar vectors to 'gene' argument.

        :param gene: The gene symbol of the gene of interest.
        :type gene: str
        :param save: The path to save the figure (optional).
        :type gene: str, optional
        :return: A matplotlib axes object representing the plot.
        :rtype:  matplotlib.figure.axes
        """
        df = self.compute_similarities(gene).head(n_genes)
        _,ax = plt.subplots(1,1,figsize=(3,6))
        sns.barplot(data=df,y="Gene",x="Similarity",palette="magma_r",ax=ax)
        ax.set_title("{} Similarity".format(gene))
        if save != None:
            plt.savefig(save)
        return ax

    def plot_metagene(self, gdata, mg=None, title="Gene Embedding"):
        """
        Plot a UMAP with the genes from a given metagene highlighted and annotated.

        :param gdata: The AnnData object holding the gene embedding (from embedding.get_adata).
        :type gdata: AnnData
        :param mg: The metagene identifier (leiden cluster number) (optional).
        :type mg: str, optional
        :param title: The title of the plot. (optional).
        :type title: str, optional
        """
        highlight = []
        labels = []
        clusters = collections.defaultdict(list)
        for x,y in zip(gdata.obs["leiden"],gdata.obs.index):
            clusters[x].append(y)
            if x == mg:
                highlight.append(str(x))
                labels.append(y)
            else:
                highlight.append("_Other")
        _labels = []
        for gene in labels:
            _labels.append(gene)
        gdata.obs["Metagene {}".format(mg)] = highlight
        _,ax = plt.subplots(1,1,figsize=(8,6))
        sc.pl.umap(gdata,alpha=0.5,show=False,size=100,ax=ax)
        sub = gdata[gdata.obs["Metagene {}".format(mg)]!="_Other"]
        sc.pl.umap(sub,color="Metagene {}".format(mg),title=title,size=200,show=False,add_outline=False,ax=ax)
        for gene, pos in zip(gdata.obs.index,gdata.obsm["X_umap"].tolist()):
            if gene in _labels:
                ax.text(pos[0]+.04, pos[1], str(gene), fontsize=6, alpha=0.9, fontweight="bold")
        plt.tight_layout()

    def get_vector(self, gene):
        """Return the embedding vector for a single gene.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        return self.embeddings[gene]

    def plot_metagenes_scores(self, adata, metagenes, column, plot=None):
        """
        Plot a Seaborn clustermap with the gene module scores for a list of metagenes over a covariate (column). Requires running score_metagenes previously.

        :param adata: The AnnData object holding the cell embedding (from embedding.CellEmbedding.get_adata).
        :type adata: AnnData
        :param metagenes: Dict of metagenes identifiers to plot in clustermap.
        :type metagenes: dict
        :param column: Covariate in obs dataframe of AnnData.
        :type column: str
        :param column: Covariate in obs dataframe of AnnData.
        :type column: str
        :param plot: Filename for saving a figure.
        :type plot: str
        """
        plt.figure(figsize = (5, 13))
        matrix = []
        meta_genes = []
        cfnum = 1
        cfams = dict()
        for cluster, vector in metagenes.items():
            row = []
            cts = []
            for ct in set(adata.obs[column]):
                sub = adata[adata.obs[column]==ct]
                val = numpy.mean(sub.obs[str(cluster)+"_SCORE"].tolist())
                row.append(val)
                cts.append(ct)
            matrix.append(row)
            label = str(cluster)+"_SCORE: " + ", ".join(vector[:10])
            if len(set(vector)) > 10:
                label += "*"
            meta_genes.append(label)
            cfams[cluster] = label
            cfnum+=1
        matrix = numpy.array(matrix)
        df = pandas.DataFrame(matrix,index=meta_genes,columns=cts)
        plt.figure()
        sns.clustermap(df,figsize=(5,9), dendrogram_ratio=0.1,cmap="mako",yticklabels=True, standard_scale=0)
        plt.tight_layout()
        if plot:
            plt.savefig(plot)

    def score_metagenes(self,adata ,metagenes):
        """
        Score a list of metagenes (get_metagenes) over all cells. 

        :param adata: The AnnData object holding the cell embedding (from embedding.CellEmbedding.get_adata).
        :type adata: AnnData
        :param metagenes: Dict of metagenes identifiers to plot in clustermap.
        :type metagenes: dict
        """
        for p, genes in metagenes.items():
            try:
                sc.tl.score_genes(adata,score_name=str(p)+"_SCORE",gene_list=genes)
                scores = numpy.array(adata.obs[str(p)+"_SCORE"].tolist()).reshape(-1,1)
                scaler = MinMaxScaler()
                scores = scaler.fit_transform(scores)
                scores = list(scores.reshape(1,-1))[0]
                adata.obs[str(p)+"_SCORE"] = scores
            except Exception as e:
                adata.obs[str(p)+"_SCORE"] = 0.
            

    def get_metagenes(self, gdata):
        """
        Score a list of metagenes (get_metagenes) over all cells. 

        :param gdata: The AnnData object holding the gene embedding (from embedding.GeneEmbedding.get_adata).
        :type gdata: AnnData
        :return: A dictionary of metagenes (identifier, gene list).
        :rtype:  dict
        """
        metagenes = collections.defaultdict(list)
        for x,y in zip(gdata.obs["leiden"],gdata.obs.index):
            metagenes[x].append(y)
        return metagenes

    def compute_similarities(self, gene, subset=None):
        """
        Compute the cosine similarities between a target gene and all other vectors in the embedding.

        :param gene: Target gene to compute cosine similarities.
        :type gene: str
        :param subset: Only compute against a subset of gene vectors. (optional).
        :type subset: list, optional
        :return: A pandas dataframe holding a gene symbol column ("Gene") and a cosine similarity column ("Similarity").
        :rtype:  pandas.DataFrmae
        """
        if gene not in self.embeddings:
            return None
        embedding = self.embeddings[gene]
        distances = dict()
        if subset:
            targets = set(list(self.embeddings.keys())).intersection(set(subset))
        else:
            targets = list(self.embeddings.keys())
        for target in targets:
            if target not in self.embeddings:
                continue
            v = self.embeddings[target]
            distance = float(cosine_similarity(numpy.array(embedding).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
            distances[target] = distance
        sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
        genes = [x[0] for x in sorted_distances]
        distance = [x[1] for x in sorted_distances]
        df = pandas.DataFrame.from_dict({"Gene":genes, "Similarity":distance})
        return df

    def generate_weighted_vector(self, genes, markers, weights):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(weights[gene] * numpy.array(vec))
            if gene not in genes and gene in markers and gene in weights:
                vector.append(list(weights[gene] * numpy.negative(numpy.array(vec))))
        return list(numpy.sum(vector, axis=0))


    def generate_vector(self, genes):
        """
        Compute an averagve vector representation for a set of genes in the learned gene embedding.

        :param genes: List of genes to generate an average vector embedding.
        :type genes: list
        :return: The average vector for a set of genes in the gene embedding.
        :rtype:  list
        """
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes:
                vector.append(vec)
        assert len(vector) != 0, genes
        return list(numpy.average(vector, axis=0))

    def generate_weighted_vector(self, genes, weights):
        """
        Compute an averagve vector representation for a set of genes in the learned gene embedding with a set of weights.

        :param genes: List of genes to generate an average vector embedding.
        :type genes: list
        :param weights: List of floats in the same order of genes to weight each vector.
        :type genes: list
        :return: The average vector for a set of genes in the gene embedding.
        :rtype:  list
        """
        vector = []
        weight = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(vec)
                weight.append(weights[gene])
        assert len(vector) != 0, genes
        return list(numpy.average(vector, axis=0, weights=weight))


    def cluster_definitions_as_df(self, top_n=20):
        similarities = self.cluster_definitions
        clusters = []
        symbols = []
        for key, genes in similarities.items():
            clusters.append(key)
            symbols.append(", ".join(genes[:top_n]))
        df = pandas.DataFrame.from_dict({"Cluster Name":clusters, "Top Genes":symbols})
        return df

    @staticmethod
    def read_vector(vec):
        lines = open(vec,"r").read().splitlines()
        dims = lines.pop(0)
        vecs = dict()
        for line in lines:
            try:
                line = line.split()
                gene = line.pop(0)
                vecs[gene] = list(map(float,line))
            except Exception as e:
                continue
        return vecs, dims

    def get_similar_genes(self, vector):
        """
        Computes the similarity of each gene in the mebedding to a target vector representation.

        :param vector: Vector representation used to find the gene similarity by cosine cosine.
        :type genes: list or numpy.array
        :return: A pandas dataframe holding the gene symbol column ("Gene") and a cosine similarity column ("Similarity").
        :rtype:  pandas.DataFrmae
        """
        distances = dict()
        targets = list(self.embeddings.keys())
        for target in targets:
            if target not in self.embeddings:
                continue
            v = self.embeddings[target]
            distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
            distances[target] = distance
        sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
        genes = [x[0] for x in sorted_distances]
        distance = [x[1] for x in sorted_distances]
        df = pandas.DataFrame.from_dict({"Gene":genes, "Similarity":distance})
        return df

    def generate_network(self, threshold=0.5):
        """
        Computes networkx graph representation of the gene embedding.

        :param threshold: Minimum cosine similarity to includea as edge in the graph.
        :type genes: float
        :return: A networkx graph with each gene as a node and the edges weighted by cosine similarity.
        :rtype:  networkx.Graph
        """
        G = nx.Graph()
        a = pandas.DataFrame.from_dict(self.embeddings).to_numpy()
        similarities = cosine_similarity(a.T)
        genes = list(self.embeddings.keys())
        similarities[similarities < threshold] = 0
        edges = []
        nz = list(zip(*similarities.nonzero()))
        for n in tqdm.tqdm(nz):
            edges.append((genes[n[0]],genes[n[1]]))
        G.add_nodes_from(genes)
        G.add_edges_from(edges)
        return G

    @staticmethod
    def average_vector_results(vec1, vec2, fname):
        output = open(fname,"w")
        vec1, dims = GeneEmbedding.read_vector(vec1)
        vec2, _ = GeneEmbedding.read_vector(vec2)
        genes = list(vec1.keys())
        output.write(dims+"\n")
        for gene in genes:
            v1 = vec1[gene]
            v2 = vec2[gene]
            meanv = []
            for x,y in zip(v1,v2):
                meanv.append(str((x+y)/2))
            output.write("{} {}\n".format(gene," ".join(meanv)))
        output.close()

class CellEmbedding(object):

    """
    This class provides an interface to the cell embedding, which can be used for tasks such as generating a UMAP visualization, assigning cell types, and identifying the similarity between cells and metagenes.


    :param dataset: The GeneVectorDataset object that was constructed from the original AnnData object.
    :type dataset: genevector.dataset.GeneVectorDataset
    :param embed: The GeneEmbeding object constructed from the dataset.
    :type embed: genevector.dataset.GeneEmbedding
    :param log_normalize: Weights average cell vector computed from all genes by the log normalized expression of each gene.
    :type vector: bool
    """

    def __init__(self, dataset, embed, log_normalize=True):
        """Constructor method
        """
        self.context = dataset.data
        self.embed = embed
        self.data = collections.defaultdict(list) # Stores (vector, weight) tuples for each cell
        self.matrix = [] # Stores average cell vectors
        self.expression = collections.defaultdict(dict) # Potentially for raw/normalized expression values
        adata_copy = self.context.adata.copy()

        if log_normalize:
            sc.pp.normalize_total(adata_copy)
            sc.pp.log1p(adata_copy)

        genes = adata_copy.var.index.to_numpy()
        barcodes = adata_copy.obs.index.to_numpy()

        # Note: The original code had normalized_vectors and normalized_marker_expression
        # initialized here but populated in a separate normalized_expression method.
        # self.normalized_vectors is populated by the call to self.normalized_expression_for_init
        self.normalized_vectors = collections.defaultdict(list)
        
        # Assuming the original 'normalized_expression' method was intended for __init__
        self._initialize_normalized_vectors(adata_copy.X, genes, barcodes)

        print(bcolors.OKGREEN + "Generating Cell Vectors." + bcolors.ENDC)
        cells_with_no_counts = 0
        # Ensure self.data keys are consistent with barcodes for which vectors are made
        temp_cell_vectors = {} # Store vectors before converting to self.matrix to ensure order
        
        processed_barcodes = [] # Keep track of barcodes for which vectors are generated

        for cell_barcode in tqdm.tqdm(barcodes, desc="Processing cell counts"):
            if cell_barcode in self.normalized_vectors and self.normalized_vectors[cell_barcode]:
                vectors, weights = zip(*self.normalized_vectors[cell_barcode])
                if vectors: # Ensure there are vectors to average
                    avg_vector = numpy.average(vectors, axis=0, weights=weights)
                    temp_cell_vectors[cell_barcode] = avg_vector
                    self.data[cell_barcode] = list(zip(vectors, weights)) # Keep original structure for self.data
                    processed_barcodes.append(cell_barcode)
                else:
                    cells_with_no_counts += 1
            else:
                cells_with_no_counts += 1
        
        # Build self.matrix in the order of processed_barcodes,
        # which should align with how self.adata will be created/filtered in get_adata()
        # However, self.data.keys() used later expects all initial barcodes with data.
        # The original code iterates adata.obs.index.tolist() from the input adata_copy for self.matrix.
        # Let's try to keep that logic for self.matrix and self.data more closely.

        self.data = collections.defaultdict(list) # Re-init for clarity, populated below
        self.matrix = [] # Re-init
        
        # Re-iterate for consistent ordering based on original barcodes (adata_copy.obs.index)
        # This matches the original loop more closely for populating self.matrix and self.data
        for cell_barcode in tqdm.tqdm(adata_copy.obs.index.tolist(), desc="Generating cell vectors"):
            if cell_barcode in self.normalized_vectors and self.normalized_vectors[cell_barcode]:
                vectors, weights = zip(*self.normalized_vectors[cell_barcode])
                if vectors:
                    self.data[cell_barcode] = list(zip(vectors, weights)) # For compatibility
                    self.matrix.append(numpy.average(vectors, axis=0, weights=weights))
                else:
                    #This cell had entries in normalized_vectors but they were empty after filtering
                    self.matrix.append(numpy.zeros(self.embed.vector_size)) # Placeholder, assuming vector_size known
                    cells_with_no_counts +=1
            else:
                # This cell had no relevant gene expression
                # Add a zero vector placeholder if this cell ID needs to be in self.matrix
                # The original code implicitly skipped these cells for self.matrix
                # but self.data.keys() would also miss them.
                # For phenotype_probability, consistency between self.data.keys() and self.matrix is key.
                # If get_adata filters by self.data.keys(), these cells are correctly excluded.
                 cells_with_no_counts +=1


        if not self.matrix:
             print(bcolors.WARNING + "No cell vectors were generated. self.matrix is empty." + bcolors.ENDC)
             self.dataset_vector = numpy.zeros(self.embed.vector_size if hasattr(self.embed, "vector_size") else 100) # Default size
        else:
             self.dataset_vector = numpy.zeros(numpy.array(self.matrix).shape[1])
        
        print(f"Found {cells_with_no_counts} Cells with No Counts / No scorable gene expression.")
        print(bcolors.BOLD + "Finished CellEmbedding Initialization." + bcolors.ENDC)


    def normalized_expression(self, normalized_matrix, genes, cells):
        normalized_matrix.eliminate_zeros()
        row_indices, column_indices = normalized_matrix.nonzero()
        nonzero_values = normalized_matrix.data
        entries = list(zip(nonzero_values, row_indices, column_indices))
        for value, i, j in tqdm.tqdm(entries):
            if value > 0 and genes[j].upper() in self.embed.embeddings:
                self.normalized_vectors[cells[i]].append((self.embed.embeddings[genes[j].upper()],value))
                self.normalized_marker_expression[cells[i]][genes[j]] = value

    def batch_correct(self, column, reference):
        """
        Corrects the matrix of cell vectors by computing vector representations for each category in a given variable in the dataset.

        :param column: Covariate signal to eliminate in the the cell embedding.
        :type column: str
        :param column: Covariate category selected as the reference to remain uncorrected.
        :type column: str
        :return: The vectors representing the correction applied to each category to the reference category.
        :rtype:  dict
        """
        if not column:
            raise ValueError("Must supply batch label to correct.")
        column_labels = dict(zip(self.context.cells, self.context.metadata[column]))
        labels = []
        for key in self.data.keys():
            labels.append(column_labels[key])
        batches = collections.defaultdict(list)
        correction_vectors = dict()
        for batch, vec in zip(labels, self.matrix):
            batches[batch].append(vec)
        assert reference in batches, "Reference label not found."
        reference_vector = numpy.average(batches.pop(reference),axis=0)
        print("Generating batch vectors.")
        for batch, vbatches in batches.items():
            if batch != reference:
                batch_vec = numpy.average(vbatches, axis=0)
                offset = numpy.subtract(reference_vector,batch_vec)
                correction_vectors[batch] = offset
                print("Computing correction vector for {}.".format(batch))
        corrected_matrix = []
        gc.collect()
        self.cell_order = []
        print("Applying correction vectors.")
        for batch, xvec in zip(labels, self.matrix):
            if  batch != reference:
                offset = correction_vectors[batch]
                xvec = numpy.add(numpy.array(xvec),offset)
            corrected_matrix.append(xvec)
        self.uncorrected_matrix = self.matrix
        self.matrix = corrected_matrix
        return correction_vectors
    
    @staticmethod
    def get_expression(adata, gene):
        """Extract expression values for a single gene from an AnnData object.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        gene : str
            Gene symbol.

        Returns
        -------
        list
            Expression values for all cells.
        """
        return adata.X[:,adata.var.index.tolist().index(gene)].T.todense().tolist()[0]

    def compare_expression_to_similarity(self, adata, gene):
        vec = self.embed.get_vector(gene)
        adata.obs["{}+".format(gene)] = self.cell_distance(vec)
        adata.obs["{}_exp".format(gene)] = self.get_expression(adata, gene)
        fig, ax =plt.subplots(1,3,figsize=(10,3))
        sc.pl.umap(adata,color=["{}_exp".format(gene)],s=30,ax=ax[0], show=False)
        sc.pl.umap(adata,color="{}+".format(gene),s=30,ax=ax[1], show=False)
        sns.scatterplot(data=adata.obs,x="{}+".format(gene),y='{}_exp'.format(gene),s=5,ax=ax[2])
        fig.tight_layout()

    def get_predictive_genes(self, adata, label, n_genes=10):
        """
        Compute the top n most similar genes to a given variable in the dataset.

        :param adata: anndata object generated from "get_adata", has "X_genevector" in the obsm dataframe.
        :type column: anndata.AnnData
        :param label: Label that defines the cateogies to find predictive genes.
        :type column: str
        :param n_genes: Number of most similar genes to return for each category.
        :type column: int
        :return: The most similar genes to each label stored in a dictionary.
        :rtype:  dict
        """
        vectors = dict()
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        comps = collections.defaultdict(list)
        for bc,x in zip(adata.obs.index,adata.obs[label]):
            comps[x].append(mapped_components[bc])
        mean_vecs = []
        for x, vec in comps.items():
            ovecs = []
            vec = numpy.average(vec,axis=0)
            for oph, ovec in comps.items():
                if oph != x:
                    ovecs.append(numpy.average(ovec,axis=0))
            aovec = numpy.median(ovecs,axis=0)
            vector = numpy.subtract(vec,aovec)
            vector = numpy.subtract(vector,self.dataset_vector)
            vectors[x] = vector
        markers = dict()
        for x, mvec in vectors.items():
            ct_sig = self.embed.get_similar_genes(mvec)[:n_genes]["Gene"].tolist()
            markers[x] = ct_sig
        return markers

    def compare_classification(self,adata,column1, column2):
        df = adata.obs[[column1,column2]]
        df=pd.crosstab(df[column1],df[column2],normalize='index')
        return sns.heatmap(df)

    def phenotype_qc(self, adata, phenotype, genes, norm=True):
        probability = "{} Pseudo-probability".format(phenotype)
        score_name =  "{} Module Score".format(phenotype)
        similarity_name = "{} Similarity".format(phenotype)
        vector = self.embed.generate_vector(genes)
        adata.obs[similarity_name] = self.cell_distance(vector)
        sc.tl.score_genes(adata,gene_list=genes,score_name=score_name)
        df = adata.obs[[probability, score_name, similarity_name]]
        df["Phenotype"] = phenotype
        fig, ax =plt.subplots(1,3,figsize=(10,3))
        sc.pl.umap(adata,color=score_name,s=30,ax=ax[0], show=False)
        sc.pl.umap(adata,color=probability,s=30,ax=ax[1], show=False)
        sc.pl.umap(adata,color=similarity_name,s=30,ax=ax[2], show=False)
        sns.pairplot(df,size=3,hue="Phenotype",kind="scatter",plot_kws={"alpha":0.5,"s":2})
        fig.tight_layout()
        return df
    
    def module_score_r2(self, adata, markers):
        values = []
        phs = []
        for phenotype, genes in markers.items():
            score_name =  "{} Module Score".format(phenotype)
            if score_name not in adata.obs.columns.tolist():
                sc.tl.score_genes(adata,gene_list=genes,score_name=score_name)
            r2 = pearsonr(adata.obs["{} Pseudo-probability".format(phenotype)],
                        adata.obs["{} Module Score".format(phenotype)]).statistic
            values.append(r2)
            phs.append(phenotype)
        df = pandas.DataFrame.from_dict({"Phenotype":phs, "r2":values})
        fig, ax =plt.subplots(1,1,figsize=(7,4))
        sns.stripplot(df,x="Phenotype",y ="r2",s=15,color="#999999",ax=ax)
        ax.set_ylim(0,1)
        ax.set_title("Probability vs Module Score (r2)")

    def plot_probabilities(self,adata,ncols=2,save="probs.pdf",palette="magma"):
        prob_cols = []
        for x in adata.obs.columns:
            if "Pseudo-probability" in x:
                prob_cols.append(x)
        sc.pl.umap(adata,color=prob_cols,s=30,ncols=ncols,cmap=palette,alpha=0.8,save=save)


    def get_inverse_predictive_genes(self, adata, label, n_genes=10):
        """
        Compute the top n least similar genes to a given variable in the dataset.

        :param adata: anndata object generated from "get_adata", has "X_genevector" in the obsm dataframe.
        :type column: anndata.AnnData
        :param label: Label that defines the cateogies to find the least predictive genes.
        :type column: str
        :param n_genes: Number of least similar genes to return for each category.
        :type column: int
        :return: The least similar genes to each label stored in a dictionary.
        :rtype:  dict
        """
        vectors = dict()
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        comps = collections.defaultdict(list)
        for bc,x in zip(adata.obs.index,adata.obs[label]):
            comps[x].append(mapped_components[bc])
        for x, vec in comps.items():
            ovecs = []
            vec = numpy.average(vec,axis=0)
            for oph, ovec in comps.items():
                ovecs.append(numpy.average(ovec,axis=0))
            aovec = numpy.median(ovecs,axis=0)
            vector = numpy.subtract(vec,aovec)
            vector = numpy.subtract(vector,self.dataset_vector)
            vectors[x] = vector
        markers = dict()
        for x, mvec in vectors.items():
            ct_sig = self.embed.get_similar_genes(mvec)[-1.*n_genes:]["Gene"].tolist()
            markers[x] = ct_sig
        return markers

    def normalized_marker_expression(self, normalized_matrix, genes, cells, markers):
        normalized_expression = collections.defaultdict(dict)
        normalized_matrix.eliminate_zeros()
        row_indices, column_indices = normalized_matrix.nonzero()
        nonzero_values = normalized_matrix.data
        entries = list(zip(nonzero_values, row_indices, column_indices))
        for value, i, j in tqdm.tqdm(entries):
            if value > 0 and genes[j].upper() in self.embed.embeddings and genes[j] in markers:
                normalized_expression[cells[i]][genes[j]] = value
        return normalized_expression

    @staticmethod
    def entmax_15(values):
        # Sort values in descending order
        sorted_values = np.sort(values)[::-1]
        # Compute the cumulative sum of the sorted values raised to the power of 2
        # Ensure array is float for power operation if not already
        sorted_values_float = np.array(sorted_values, dtype=float)
        cumsum_sorted = np.cumsum(sorted_values_float**2) # Using **2 based on typical alpha=2 for sparsemax from entmax paper. Original code had **2.
                                                       # Entmax 1.5 should technically be alpha=1.5
                                                       # If original used x^2, it's not entmax(alpha=1.5) but entmax(alpha=2) i.e. sparsemax
                                                       # The original paper for entmax defines it with (alpha-1) * theta + ...
                                                       # For alpha=1.5, it's not just x^2.
                                                       # Given the name "entmax_15" but using **2 suggests a potential mismatch.
                                                       # Assuming the **2 is the intended operation (sparsemax).
                                                       # If alpha=1.5 is strictly needed, the formula is different.
                                                       # For now, keeping original **2 operation as it implies sparsemax (alpha=2).
        
        # Compute the number of elements that will have non-zero probabilities
        # rho is the largest k such that sorted_values[k-1] * k > sum(sorted_values[:k]) - tau (generalized)
        # For sparsemax (alpha=2): sorted_values[k-1] > (cumsum_sorted[k-1]-1)/k
        # The original where clause: sorted_values > (cumsum_sorted - 1) / np.arange(1, len(values) + 1)
        # This is correct for finding rho for sparsemax (alpha=2).
        
        active_set_indices = np.where(sorted_values_float * np.arange(1, len(values) + 1) > (cumsum_sorted - 1))[0]
        if len(active_set_indices) == 0: # All probabilities will be zero (e.g. input is all zero or negative)
             return np.zeros_like(values, dtype=float)
        rho = active_set_indices[-1] # rho is 0-indexed count (k-1)

        # Compute the threshold theta
        theta = (cumsum_sorted[rho] - 1) / (rho + 1)
        
        # Apply the thresholding operation: max(0, values - theta) for sparsemax if values are logits.
        # The original paper uses max(0, v_i - tau)^ (1/(alpha-1)) for entmax(alpha)
        # The provided code uses max(values - theta, 0)**2. This doesn't directly match entmax(1.5).
        # It looks more like a squared hinge loss or a variant of sparsemax if theta is defined differently.
        # If this is custom, it's fine. If it's meant to be standard entmax(1.5), it needs review.
        # Sticking to the provided formula:
        probabilities = np.maximum(np.array(values, dtype=float) - theta, 0)**2 # Element-wise power
        
        # Normalize the probabilities
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities /= sum_probs
        else: # Avoid division by zero if all probabilities are zero
            probabilities = np.zeros_like(values, dtype=float)
            # Optionally distribute uniformly if that's desired for all-zero input after thresholding
            # probabilities[:] = 1.0 / len(values) 
        return probabilities


    @staticmethod
    def normalized_exponential_vector(values, temperature=0.000001):
        assert temperature > 0, "Temperature must be positive"
        values_arr = np.array(values, dtype=float) # Ensure float for division
        exps = np.exp(values_arr / temperature)
        sum_exps = np.sum(exps)
        if sum_exps > 0:
            return exps / sum_exps
        else:
            # Handle cases where all exps are zero (e.g., very small values with small temperature)
            # Return uniform distribution or zeros based on desired behavior
            return np.zeros_like(values_arr)


    def cell_distance(self, target_vec, norm=False):
        """
        Computes the cosine similarity of each cell in self.adata to a target_vec.
        Ensures self.adata is available (set by get_adata()).
        The 'norm' parameter controls if both target_vec and cell vectors are L2 normalized before similarity calculation.
        """
        if not hasattr(self, 'adata') or self.adata is None:
            print(bcolors.WARNING + "self.adata is not set. Call get_adata() first. Returning empty distances." + bcolors.ENDC)
            return []
        
        # This map uses self.data.keys() and self.matrix from __init__
        # It's crucial that self.adata.obs.index aligns with these.
        # get_adata() does: adata = adata[list(self.data.keys())], so they should align.
        cell_id_to_matrix_vector_map = dict(zip(list(self.data.keys()), self.matrix))

        similarities = []
        
        # Process target_vec (convert to float array and optionally normalize)
        processed_target_vec = np.array(target_vec, dtype=float)
        if norm:
            target_norm_val = np.linalg.norm(processed_target_vec)
            if target_norm_val > 0:
                processed_target_vec /= target_norm_val
            # else: target_vec is a zero vector, distance.cosine will handle it

        for cell_id in tqdm.tqdm(self.adata.obs.index, desc="Calculating cell distances"):
            # Get the pre-computed average vector for the cell from self.matrix
            cell_matrix_avg_vector = cell_id_to_matrix_vector_map.get(cell_id)
            
            if cell_matrix_avg_vector is None:
                print(bcolors.WARNING + f"Cell ID {cell_id} from self.adata.obs.index not found in internal matrix map. Skipping." + bcolors.ENDC)
                similarities.append(np.nan) # Or some other placeholder like 0.0
                continue

            processed_cell_vec = np.array(cell_matrix_avg_vector, dtype=float)
            if norm:
                cell_norm_val = np.linalg.norm(processed_cell_vec)
                if cell_norm_val > 0:
                    processed_cell_vec /= cell_norm_val
            
            # distance.cosine calculates 1 - (u.v / (||u||*||v||))
            # so 1 - distance.cosine is the similarity.
            # Handles zero vectors appropriately (e.g., cosine([0,0],[1,1]) might be 0 or undefined based on library)
            # scipy.spatial.distance.cosine returns 1 if u or v is all zero. So similarity = 0.
            sim = 1. - distance.cosine(processed_cell_vec, processed_target_vec)
            similarities.append(sim)
            
        return similarities


    def phenotype_probability(self, adata, phenotype_markers, return_distances=False, method="normalized_exponential", target_col="genevector", temperature=0.001):
        """
        Probabilistically assign phenotypes based on a set of cell type labels and associated markers.
        Loads into the anndata the pseudo-probabilities for each cell type and the deterministic label
        taken from the maximum probability over cell types.

        :param adata: AnnData object. It's assumed this is `self.adata` or is consistent with it,
                      especially regarding `adata.obs.index` if `self.cell_distance` is used.
        :type adata: anndata.AnnData
        :param phenotype_markers: Dictionary of cell type labels (key) to gene markers (list of strings, value).
        :type phenotype_markers: dict
        :param return_distances: If True, return a tuple: (adata, dictionary_of_raw_similarities).
        :type return_distances: bool
        :param method: Probability conversion method: "softmax", "sparsemax", or "normalized_exponential".
        :type method: str
        :param target_col: Column name in `adata.obs` to store the final deterministic cell assignments.
        :type target_col: str
        :param temperature: Temperature parameter for the "normalized_exponential" method.
        :type temperature: float
        :return: AnnData with cell type labels and probabilities. If return_distances is True,
                 returns a tuple (adata, raw_similarities_dict).
        :rtype: anndata.AnnData or tuple
        """
        if method == "softmax":
            print(bcolors.OKBLUE + "Using **SoftMax**" + bcolors.ENDC)
            pfunc = softmax
        elif method == "sparsemax": # Assuming self.entmax_15 is defined
            print(bcolors.OKBLUE + "Using **SparseMax (1.5-entmax)**" + bcolors.ENDC)
            pfunc = self.entmax_15
        elif method == "normalized_exponential":
            print(bcolors.OKBLUE + f"Using Normalized Exponential (Temp: {temperature})" + bcolors.ENDC)
            pfunc = lambda x: self.normalized_exponential_vector(x, temperature)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'softmax', 'sparsemax', 'normalized_exponential'.")

        # Clear any pre-existing "Pseudo-probability" columns from adata.obs
        cols_to_remove = [col for col in adata.obs.columns if "Pseudo-probability" in col]
        if cols_to_remove:
            adata.obs.drop(columns=cols_to_remove, inplace=True)

        # This relies on self.cell_distance, which iterates self.adata.obs.index.
        # Ensure the input 'adata' is consistent with 'self.adata'.
        if not hasattr(self, 'adata') or self.adata is None or adata is not self.adata:
             print(bcolors.WARNING + "Input 'adata' might not be consistent with 'self.adata' used by "
                                   "internal methods like cell_distance. Ensure get_adata() was called and "
                                   "the same AnnData object is used." + bcolors.ENDC)


        phenotype_names = list(phenotype_markers.keys())
        # Stores raw similarities: {phenotype_name: [sim_cell1, sim_cell2, ...]}
        # Order of similarities in lists will correspond to self.adata.obs.index
        raw_similarity_scores = collections.defaultdict(list)

        for pheno_name in tqdm.tqdm(phenotype_names, desc="Computing similarities per phenotype"):
            markers = phenotype_markers[pheno_name]
            if not markers:
                print(bcolors.WARNING + f"No markers provided for phenotype {pheno_name}. Skipping." + bcolors.ENDC)
                # Assign a default low similarity or handle as appropriate
                raw_similarity_scores[pheno_name] = [0.0] * len(self.adata.obs) # Or len(adata.obs) if strictly using input adata
                continue
            
            print(bcolors.OKGREEN + f"Markers for {pheno_name}: {', '.join(markers[:5])}{'...' if len(markers) > 5 else ''}" + bcolors.ENDC)
            phenotype_vector = self.embed.generate_vector(markers) # Assumes gene names are uppercase or handled by generate_vector
            
            # self.cell_distance calculates similarities for cells in self.adata.obs.index
            # norm=False means use raw vectors for cosine similarity.
            similarities_for_pheno = self.cell_distance(phenotype_vector, norm=False)
            raw_similarity_scores[pheno_name] = similarities_for_pheno

        # Prepare matrix for probability calculation: rows are cells, columns are phenotypes
        # The order of cells is implicitly self.adata.obs.index
        # The order of phenotypes is phenotype_names
        num_cells = len(self.adata.obs) # Number of cells for which distances were computed
        similarity_matrix_cells_x_phenos = np.zeros((num_cells, len(phenotype_names)))

        for i, pheno_name in enumerate(phenotype_names):
            if pheno_name in raw_similarity_scores:
                 # Ensure list length matches num_cells, pad if necessary (e.g. if a pheno was skipped)
                pheno_sims = raw_similarity_scores[pheno_name]
                if len(pheno_sims) == num_cells:
                    similarity_matrix_cells_x_phenos[:, i] = pheno_sims
                else:
                    print(bcolors.WARNING + f"Similarity score list length mismatch for {pheno_name}. Expected {num_cells}, got {len(pheno_sims)}. Padding with zeros." + bcolors.ENDC)
                    similarity_matrix_cells_x_phenos[:len(pheno_sims), i] = pheno_sims # Fill what's available

        # Apply probability function per cell (i.e., per row of similarity_matrix_cells_x_phenos)
        probabilities_per_cell = [] # List of np.arrays, each array is probs for one cell
        for i in range(num_cells):
            cell_similarity_vector = similarity_matrix_cells_x_phenos[i, :]
            # Replace NaNs with a low value if any occurred in cell_distance
            cell_similarity_vector = np.nan_to_num(cell_similarity_vector, nan=-1.0) # Or other appropriate fill
            
            prob_dist_for_cell = pfunc(cell_similarity_vector)
            probabilities_per_cell.append(prob_dist_for_cell)
        
        # Store results in the input 'adata' object
        # Probabilities are ordered by self.adata.obs.index. We assign to input 'adata'.
        # This assumes input 'adata.obs.index' is the same as 'self.adata.obs.index'.
        
        assigned_phenotypes_list = []
        # For storing individual phenotype probability columns
        phenotype_prob_data_for_adata = collections.defaultdict(lambda: [np.nan] * len(adata.obs))


        for cell_idx, cell_id in enumerate(adata.obs.index):
            # This loop iterates the *input* adata. We need to ensure probabilities_per_cell maps correctly.
            # If adata is self.adata, cell_idx directly corresponds to probabilities_per_cell[cell_idx].
            if cell_idx < len(probabilities_per_cell):
                prob_vector_for_this_cell = probabilities_per_cell[cell_idx]
                winning_pheno_idx = np.argmax(prob_vector_for_this_cell)
                assigned_phenotypes_list.append(phenotype_names[winning_pheno_idx])

                for pheno_idx, pheno_name in enumerate(phenotype_names):
                    phenotype_prob_data_for_adata[pheno_name][cell_idx] = prob_vector_for_this_cell[pheno_idx]
            else:
                # Should not happen if adata and self.adata are consistent and num_cells matches
                assigned_phenotypes_list.append(np.nan) # Or some default
                print(bcolors.WARNING + f"Mismatch in cell counts when assigning probabilities for cell {cell_id}." + bcolors.ENDC)


        adata.obs[target_col] = pd.Categorical(assigned_phenotypes_list, categories=phenotype_names) # Use Categorical for defined order

        prob_col_names_for_uns = []
        for pheno_name in phenotype_names:
            col_name = f"{pheno_name} Pseudo-probability"
            adata.obs[col_name] = phenotype_prob_data_for_adata[pheno_name]
            prob_col_names_for_uns.append(col_name)
        
        adata.uns["probability_columns"] = prob_col_names_for_uns

        if return_distances:
            # Construct a dictionary of raw similarities aligned with adata.obs.index
            # raw_similarity_scores is already {pheno_name: [list_of_sims_ordered_by_self.adata.obs.index]}
            # If adata is self.adata, this is directly usable.
            distances_dict = {
                "phenotypes": phenotype_names,
                "cell_ids": list(adata.obs.index), # or self.adata.obs.index
                "similarity_matrix_cells_x_phenos": similarity_matrix_cells_x_phenos # rows=cells, cols=phenos
            }
            return adata, distances_dict
        else:
            return adata


    def cosine_sim_qc(self, dists):
        ddf = pd.DataFrame(data = np.array(dist["distances"]),columns=dist['order'])
        sns.pairplot(data=ddf,kind="reg",plot_kws={"scatter_kws":{"s":0.1}})
        return ddf
    

    def cluster(self, adata, up_markers, down_markers=dict()):
        """
        Run GaussianMixture over cosine similarities for up and down markers.

        :param adata: AnnData object with X_genevector embedding.
        :type adata: anndata.AnnData
        :param up_markers: Dictionary of up regulated genes defining phenotypes.
        :type: up_markers: dict
        :param down_markers: Dictionary of down regulated genes defining phenotypes (optional).
        :type: down_markers: dict
        :return: Anndata with clusters stored in metadata ("gcluster") and probabilities ("{} Probability").
        :rtype:  anndata.AnnData
        """
        up_only = set(up_markers.keys()).difference(set(down_markers.keys()))
        down_markers = dict()
        down_only = set(down_markers.keys()).difference(set(up_markers.keys()))
        up_and_down = set(down_markers.keys()).intersection(set(up_markers.keys()))
        phs = list(set(up_markers.keys()).union(set(down_markers.keys())))
        for ph in up_only:
            genes = up_markers[ph]
            vec = self.embed.generate_vector(genes)
            odists = []
            mapped_components = dict(zip(list(self.data.keys()),self.matrix))
            for x in tqdm.tqdm(adata.obs.index):
                dist = 1. - distance.cosine(mapped_components[x], numpy.array(vec))
                odists.append(dist)
            adata.obs[ph] = odists
        for ph in down_only:
            genes = down_markers[ph]
            vec = self.embed.generate_vector(genes)
            odists = []
            mapped_components = dict(zip(list(self.data.keys()),self.matrix))
            for x in tqdm.tqdm(adata.obs.index):
                dist = distance.cosine(mapped_components[x], numpy.array(vec))
                odists.append(dist)
            adata.obs[ph] = odists
        for ph in up_and_down:
            print(ph)
            up_genes = up_markers[ph]
            down_genes = down_markers[ph]
            vec_up = self.embed.generate_vector(up_genes)
            vec_down = self.embed.generate_vector(down_genes)
            vec = numpy.subtract(vec_up,vec_down)
            odists = []
            mapped_components = dict(zip(list(self.data.keys()),self.matrix))
            for x in tqdm.tqdm(adata.obs.index):
                dist = 1.-distance.cosine(mapped_components[x], numpy.array(vec))
                odists.append(dist)
            adata.obs[ph] = odists
        dist = adata.obs[phs].to_numpy()
        gm = GaussianMixture(n_components=len(phs), random_state=42, verbose=True).fit(dist)
        adata.obs["gcluster"] = ["C"+str(x) for x in gm.predict(dist)]
        probs = gm.predict_proba(dist) 
        for c, prob in zip(set(adata.obs["gcluster"]), probs.T):
            adata.obs["{}_probability".format(c)] = prob
        return adata


    def _initialize_normalized_vectors(self, count_matrix, genes, cells):
        """
        Helper method to populate self.normalized_vectors from the count matrix.
        This replaces the `normalized_expression` method body used in the original constructor.
        """
        if not isinstance(count_matrix, csr_matrix):
            count_matrix = csr_matrix(count_matrix)
        
        count_matrix.eliminate_zeros()
        row_indices, column_indices = count_matrix.nonzero()
        nonzero_values = count_matrix.data
        
        # Build a gene to embedding map for quick lookup
        gene_embedding_map = {gene.upper(): emb for gene, emb in self.embed.embeddings.items()}

        for value, r_idx, c_idx in tqdm.tqdm(zip(nonzero_values, row_indices, column_indices), total=len(nonzero_values), desc="Mapping gene expression to embeddings"):
            gene_symbol = genes[c_idx]
            cell_barcode = cells[r_idx]
            
            embedding_vector = gene_embedding_map.get(gene_symbol.upper())
            if value > 0 and embedding_vector is not None:
                self.normalized_vectors[cell_barcode].append((embedding_vector, value))
                # self.normalized_marker_expression can be populated if needed, similar logic
    

    def get_adata(self, min_dist=0.3, n_neighbors=50):
        """
        Return a anndata object to use in downstream analyses that contains the cell embedding matrix 
        (under "X_genevector" in obsm) alongside the neighbors graph and UMAP embedding computed 
        using the cell vectors.

        :param min_dist: UMAP generation parameter.
        :type min_dist: float
        :param n_neighbors: Number of neighbors defined by cosine similarity to include in neighborhood graph.
        :type: n_neighbors: int
        :return: Anndata with cell embedding stored in metadata ("obsm").
        :rtype:  anndata.AnnData
        """
        print(bcolors.OKGREEN + "Loading embedding in X_genevector." + bcolors.ENDC)
        
        # Ensure self.data and self.matrix are populated from __init__
        if not self.data or not self.matrix:
            print(bcolors.FAIL + "CellEmbedding data or matrix not initialized. Run constructor properly." + bcolors.ENDC)
            # Or, attempt to run parts of __init__ if feasible, though better to ensure constructor worked.
            # For now, assume __init__ was successful.

        current_adata = self.context.adata.copy()
        
        # Filter adata to include only cells for which we have embeddings (keys in self.data)
        # This makes current_adata.obs.index consistent with self.data.keys() and self.matrix row order.
        cells_with_embeddings = list(self.data.keys())
        if not cells_with_embeddings:
            print(bcolors.FAIL + "No cells with embeddings found in self.data. Cannot proceed with get_adata." + bcolors.ENDC)
            return current_adata # Or raise an error

        current_adata = current_adata[cells_with_embeddings, :].copy() # Ensure it's a copy after filtering

        # Rebuild X_genevector in the order of current_adata.obs.index
        # self.matrix rows are aligned with self.data.keys() from constructor.
        # And current_adata.obs.index is now self.data.keys().
        cell_id_to_matrix_row_idx = {cell_id: i for i, cell_id in enumerate(cells_with_embeddings)}
        
        x_genevector_list = []
        for cell_id_in_adata in current_adata.obs.index:
            matrix_row_idx = cell_id_to_matrix_row_idx.get(cell_id_in_adata)
            if matrix_row_idx is not None and matrix_row_idx < len(self.matrix):
                 x_genevector_list.append(self.matrix[matrix_row_idx])
            else:
                # This should not happen if cells_with_embeddings was used correctly
                print(bcolors.WARNING + f"Could not find vector for cell {cell_id_in_adata} in self.matrix. Using zero vector.")
                # Determine vector size from first vector in self.matrix or a default
                vec_size = self.matrix[0].shape[0] if self.matrix and len(self.matrix[0]) > 0 else 100
                x_genevector_list.append(np.zeros(vec_size))


        current_adata.obsm['X_genevector'] = np.array(x_genevector_list)
        
        print(bcolors.OKGREEN + "Running Scanpy neighbors and umap." + bcolors.ENDC)
        sc.pp.neighbors(current_adata, use_rep="X_genevector", n_neighbors=n_neighbors)
        sc.tl.umap(current_adata, min_dist=min_dist)
        
        self.adata = current_adata # Store this processed anndata
        return self.adata

    @staticmethod
    def plot_confusion_matrix(adata,label1,label2):
        """
        Plot accuracy of GeneVector cell type assignments with a set of known cell types or clusters.

        :param adata: AnnData object with both genevector cell type labels and ground truth cell type or cluster or assignment.
        :type adata: anndata.AnnData
        :param label1: Target column for GeneVector cell type assignments.
        """
        gv_ct = adata.obs[label1].tolist()
        target_ct = adata.obs[label2].tolist()
        def plot_cm(y_true, y_pred, figsize=(6,6)):
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            annot = np.empty_like(cm).astype(str)
            nrows, ncols = cm.shape
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    p = cm_perc[i, j]
                    if i == j:
                        s = cm_sum[i]
                        annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                    elif c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '%.1f%%\n%d' % (p, c)
            cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
            cm.index.name = 'TICA - Coarse Labels'
            cm.columns.name = 'GeneVector'
            _, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        plot_cm(target_ct,gv_ct)
