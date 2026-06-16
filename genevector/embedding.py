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

from ._logging import get_logger

logger = get_logger(__name__)

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
            logger.info("Loading average of 1st and 2nd weights.")
            avg_embedding = embedding_file.replace(".vec","_avg.vec")
            secondary_weights = embedding_file.replace(".vec","2.vec")
            GeneEmbedding.average_vector_results(embedding_file,secondary_weights,avg_embedding)
            self.embeddings = self.read_embedding(avg_embedding)
        elif vector == "1":
            logger.info("Loading first weights.")
            self.embeddings = self.read_embedding(embedding_file)
        elif vector == "2":
            logger.info("Loading second weights.")
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


    @staticmethod
    def read_vector(vec):
        """Read a .vec file into a dict of gene vectors and a dimension string.

        Parameters
        ----------
        vec : str
            Path to .vec file.

        Returns
        -------
        tuple of (dict, str)
            Mapping from gene symbol to list of floats, and dimension header line.
        """
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
        """Average two .vec embedding files and write the result to a new file.

        Parameters
        ----------
        vec1 : str
            Path to first .vec file.
        vec2 : str
            Path to second .vec file.
        fname : str
            Output path for averaged .vec file.
        """
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

        self.normalized_vectors = collections.defaultdict(list)
        self._initialize_normalized_vectors(adata_copy.X, genes, barcodes)

        logger.info("Generating Cell Vectors.")
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
             logger.warning("No cell vectors were generated. self.matrix is empty.")
             self.dataset_vector = numpy.zeros(self.embed.vector_size if hasattr(self.embed, "vector_size") else 100) # Default size
        else:
             # The dataset vector is the mean cell vector — the shared "background"
             # direction every cell points toward. Previously left as zeros, which
             # silently disabled the contrastive subtraction in get_predictive_genes
             # and made debiased phenotype scoring impossible.
             self.dataset_vector = numpy.array(self.matrix).mean(axis=0)

        logger.info(f"Found {cells_with_no_counts} Cells with No Counts / No scorable gene expression.")
        logger.info("Finished CellEmbedding Initialization.")


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
        logger.info("Generating batch vectors.")
        for batch, vbatches in batches.items():
            if batch != reference:
                batch_vec = numpy.average(vbatches, axis=0)
                offset = numpy.subtract(reference_vector,batch_vec)
                correction_vectors[batch] = offset
                logger.info(f"Computing correction vector for {batch}.")
        corrected_matrix = []
        gc.collect()
        self.cell_order = []
        logger.info("Applying correction vectors.")
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
        """Plot gene expression vs embedding similarity for a single gene.

        Parameters
        ----------
        adata : AnnData
            Annotated data with UMAP coordinates.
        gene : str
            Gene symbol to compare.
        """
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
        """Plot a heatmap comparing two categorical assignments.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        column1 : str
            First obs column (rows of heatmap).
        column2 : str
            Second obs column (columns of heatmap).

        Returns
        -------
        matplotlib.axes.Axes
            Heatmap axes object.
        """
        df = adata.obs[[column1,column2]]
        df=pd.crosstab(df[column1],df[column2],normalize='index')
        return sns.heatmap(df)

    def phenotype_qc(self, adata, phenotype, genes, norm=True):
        """Quality control comparison of pseudo-probability, module score, and embedding similarity.

        Parameters
        ----------
        adata : AnnData
            Annotated data with UMAP and phenotype probabilities.
        phenotype : str
            Phenotype label name.
        genes : list of str
            Marker genes for the phenotype.
        norm : bool
            If True, normalize cell vectors before distance computation.

        Returns
        -------
        pandas.DataFrame
            QC metrics per cell.
        """
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
        """Plot R-squared between pseudo-probability and module score for each phenotype.

        Parameters
        ----------
        adata : AnnData
            Annotated data with pseudo-probabilities and module scores.
        markers : dict
            Mapping of phenotype name to gene list.
        """
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
        """Plot UMAP colored by pseudo-probability for all phenotypes.

        Parameters
        ----------
        adata : AnnData
            Annotated data with pseudo-probability columns.
        ncols : int
            Number of columns in subplot grid.
        save : str
            File path for saving the figure.
        palette : str
            Matplotlib colormap name.
        """
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
            ct_sig = self.embed.get_similar_genes(mvec)[int(-1.*n_genes):]["Gene"].tolist()
            markers[x] = ct_sig
        return markers

    def normalized_marker_expression(self, normalized_matrix, genes, cells, markers):
        """Extract normalized expression for marker genes from a sparse matrix.

        Parameters
        ----------
        normalized_matrix : scipy.sparse matrix
            Normalized expression matrix.
        genes : array-like
            Gene names.
        cells : array-like
            Cell barcodes.
        markers : list of str
            Marker genes to extract.

        Returns
        -------
        dict of dict
            Mapping from cell barcode to {gene: expression_value}.
        """
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
        """Sparse probability mapping using 1.5-entmax (sparsemax variant).

        Applies a sparse transformation that maps real-valued scores to a
        probability distribution where low-scoring entries are driven to
        exactly zero.

        Parameters
        ----------
        values : array-like
            Input scores.

        Returns
        -------
        np.ndarray
            Sparse probability distribution summing to 1.
        """
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
        """Temperature-scaled softmax normalization.

        Parameters
        ----------
        values : array-like
            Input scores.
        temperature : float
            Temperature parameter. Lower values produce sharper distributions.

        Returns
        -------
        np.ndarray
            Probability distribution summing to 1.
        """
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
            logger.warning("self.adata is not set. Call get_adata() first. Returning empty distances.")
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
                logger.warning(f"Cell ID {cell_id} from self.adata.obs.index not found in internal matrix map. Skipping.")
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


    def phenotype_probability(self, adata, phenotype_markers, return_distances=False, method="normalized_exponential", target_col="genevector", temperature=0.001, debias=0.0, contrastive=False, score_norm="none", smooth_graph=None, smooth_alpha=0.5, smooth_adaptive=True, smooth_counts=None, lp_graph=None, lp_alpha=0.0, lp_iter=3):
        """
        Probabilistically assign phenotypes based on a set of cell type labels and associated markers.
        Loads into the anndata the pseudo-probabilities for each cell type and the deterministic label
        taken from the maximum probability over cell types.

        :param adata: AnnData object. It's assumed this is `self.adata` or is consistent with it,
                      especially regarding `adata.obs.index`.
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
        :param debias: Fraction (0–1) of the dataset (background) vector subtracted from each cell
                       vector before scoring. Opt-in (default 0.0 = current behaviour). Mild values
                       (~0.5) de-saturate scores on balanced data; large values can over-correct a
                       dominant population (e.g. epithelial-rich tumours). See :meth:`scoring_report`.
        :type debias: float
        :param contrastive: If True, subtract the mean of competing phenotype vectors from each
                            phenotype vector before scoring (the get_predictive_genes formulation).
                            Opt-in (default False). Helps when phenotypes are not well separated.
        :type contrastive: bool
        :param score_norm: Per-phenotype normalization of the cell×phenotype similarity columns
                           before the probability function. One of "none" (default), "zscore"
                           (subtract each phenotype's mean / divide by std — stops a phenotype that
                           is close to everyone from winning by default), or "rank" (per-phenotype
                           percentile rank). Opt-in; helps on some datasets, default off.
        :type score_norm: str
        :param smooth_graph: Optional scipy sparse adjacency (cells x cells, same order as the cell
                             embedding) to spatially denoise the cell vectors *for scoring only*
                             (does not mutate self.matrix / the UMAP). Opt-in. For persistent
                             denoising that also affects the embedding use :meth:`denoise_cell_vectors`.
        :type smooth_graph: scipy.sparse matrix or None
        :param smooth_alpha: Smoothing weight (or per-cell cap when ``smooth_adaptive``).
        :type smooth_alpha: float
        :param smooth_adaptive: Scale the smoothing weight per cell by inverse count (low-count
                                cells borrow more). Default True.
        :type smooth_adaptive: bool
        :param smooth_counts: Per-cell totals for adaptive smoothing; if None, derived from the
                              loaded expression context.
        :type smooth_counts: array-like or None
        :param lp_graph: Optional scipy sparse adjacency (cells x cells, same order as the cell
                         embedding) for spatial label propagation over the soft probabilities. Opt-in.
        :type lp_graph: scipy.sparse matrix or None
        :param lp_alpha: Label-propagation coupling in [0, 1) (0 disables). Smooths probabilities
                         over ``lp_graph`` as a post-step. Improves spatial coherence; can blur
                         identity in intermixed tissue — keep small (~0.3) and opt-in.
        :type lp_alpha: float
        :param lp_iter: Number of label-propagation iterations.
        :type lp_iter: int
        :return: AnnData with cell type labels and probabilities. If return_distances is True,
                 returns a tuple (adata, raw_similarities_dict).
        :rtype: anndata.AnnData or tuple
        """
        if method == "softmax":
            logger.info("Using SoftMax")
            pfunc = softmax
        elif method == "sparsemax": # Assuming self.entmax_15 is defined
            logger.info("Using SparseMax (1.5-entmax)")
            pfunc = self.entmax_15
        elif method == "normalized_exponential":
            logger.info(f"Using Normalized Exponential (Temp: {temperature})")
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
             logger.warning(
                 "Input 'adata' might not be consistent with 'self.adata' used by "
                 "internal methods like cell_distance. Ensure get_adata() was called and "
                 "the same AnnData object is used."
             )


        phenotype_names = list(phenotype_markers.keys())

        # Cell matrix aligned to self.adata.obs.index (== self.matrix row order).
        C = np.asarray(self.matrix, dtype=float)
        if smooth_graph is not None:
            # Spatially denoise a COPY of the cell vectors for scoring only (self.matrix and
            # the UMAP are untouched). Opt-in; helps sparse / spatially-organised data.
            C = self._graph_smooth(C, smooth_graph, alpha=smooth_alpha,
                                   adaptive=smooth_adaptive, counts=smooth_counts)
        if debias:
            # Subtract a fraction of the shared background direction. Opt-in: mild values
            # de-saturate scores on balanced data; large values over-correct a dominant
            # population. Default debias=0.0 reproduces the original absolute scoring.
            C = C - float(debias) * np.asarray(self.dataset_vector, dtype=float)

        # Phenotype (marker-mean) vectors.
        pheno_vectors = []
        for pheno_name in phenotype_names:
            markers = phenotype_markers[pheno_name] or []
            present = [g for g in markers if g in self.embed.embeddings]
            if not present:
                logger.warning(f"No usable markers for phenotype {pheno_name}; scoring as zeros.")
                pheno_vectors.append(np.zeros(C.shape[1]))
                continue
            missing = [g for g in markers if g not in self.embed.embeddings]
            if missing:
                logger.info(f"{pheno_name}: {len(missing)} marker(s) absent from embedding: {missing[:5]}")
            pheno_vectors.append(np.asarray(self.embed.generate_vector(present), dtype=float))
        V = np.asarray(pheno_vectors, dtype=float)

        if contrastive and len(V) > 1:
            # Contrastive phenotype vectors: subtract the mean of competing phenotypes
            # (and the background if debiasing) — the get_predictive_genes formulation.
            bg = float(debias) * np.asarray(self.dataset_vector, dtype=float) if debias else 0.0
            Vc = np.empty_like(V)
            for i in range(len(V)):
                Vc[i] = V[i] - np.delete(V, i, axis=0).mean(axis=0) - bg
            V = Vc

        # Vectorized cosine similarity (cells x phenotypes).
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        similarity_matrix_cells_x_phenos = Cn @ Vn.T
        num_cells = similarity_matrix_cells_x_phenos.shape[0]

        # Optional per-phenotype (column) normalization of the similarity matrix. De-biases a
        # phenotype that is close to every cell so it doesn't win the argmax by default. Opt-in.
        if score_norm and score_norm != "none":
            S = similarity_matrix_cells_x_phenos
            if score_norm == "zscore":
                S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-9)
            elif score_norm == "rank":
                S = np.argsort(np.argsort(S, axis=0), axis=0) / max(S.shape[0] - 1, 1)
            else:
                raise ValueError(f"Unknown score_norm: {score_norm}. Choose 'none', 'zscore', 'rank'.")
            similarity_matrix_cells_x_phenos = S

        # Apply the probability function per cell (per row).
        probabilities_per_cell = [
            pfunc(np.nan_to_num(similarity_matrix_cells_x_phenos[i, :], nan=-1.0))
            for i in range(num_cells)
        ]

        # Optional spatial label propagation over the soft probabilities (opt-in post-step).
        if lp_graph is not None and lp_alpha and lp_alpha > 0:
            P = self._label_propagate(np.vstack(probabilities_per_cell), lp_graph,
                                      alpha=lp_alpha, n_iter=lp_iter)
            probabilities_per_cell = [P[i] for i in range(P.shape[0])]
        
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
                logger.warning(f"Mismatch in cell counts when assigning probabilities for cell {cell_id}.")


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

    # ─── Spatial / graph helpers (opt-in) ──────────────────────────

    @staticmethod
    def _row_normalize_graph(graph):
        """Row-normalize a sparse adjacency so each row sums to 1 (zero-degree rows stay 0)."""
        from scipy.sparse import diags
        rs = numpy.asarray(graph.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        return diags(1.0 / rs) @ csr_matrix(graph)

    def _cell_total_counts(self):
        """Per-cell total expression aligned to self.matrix / self.data order."""
        keys = list(self.data.keys())
        exp = getattr(self.context, "expression", None)
        if exp:
            return numpy.array([sum(exp.get(k, {}).values()) for k in keys], dtype=float)
        return numpy.ones(len(keys), dtype=float)

    def _label_propagate(self, P, graph, alpha=0.3, n_iter=3):
        """Spatial label propagation: F <- a*Wn@F + (1-a)*P0, iterated; returns smoothed probs."""
        Wn = self._row_normalize_graph(graph)
        P0 = numpy.asarray(P, dtype=float)
        F = P0.copy()
        for _ in range(int(n_iter)):
            F = alpha * numpy.asarray(Wn @ F) + (1.0 - alpha) * P0
        return F

    def _graph_smooth(self, M, graph, alpha=0.5, adaptive=True, counts=None, k0=None, beta=0.0):
        """Message-pass a matrix over a graph: (1-a)*self + a*mean(neighbours) [+ beta*2-hop].

        Pure (does not mutate). ``adaptive=True`` makes a_i = min(cap, k0/(k0+counts_i)) so
        low-count cells borrow more. Shared by denoise_cell_vectors and phenotype_probability.
        """
        M = numpy.asarray(M, dtype=float)
        Wn = self._row_normalize_graph(graph)
        one = numpy.asarray(Wn @ M)
        if adaptive:
            if counts is None:
                counts = self._cell_total_counts()
            counts = numpy.asarray(counts, dtype=float).ravel()
            if k0 is None:
                pos = counts[counts > 0]
                k0 = float(numpy.median(pos)) if pos.size else 1.0
            cap = alpha if 0 < alpha < 1 else 0.85
            a = numpy.minimum(cap, k0 / (k0 + counts)).reshape(-1, 1)
        else:
            a = float(alpha)
        out = (1.0 - a) * M + a * one
        if beta > 0:
            out = out + beta * (numpy.asarray(Wn @ one) - one)
        return out

    def denoise_cell_vectors(self, graph, alpha=0.5, adaptive=True, counts=None,
                             k0=None, beta=0.0):
        """Graph-denoise the cell matrix in gene-vector space (opt-in; modifies self.matrix).

        Message passing over a (usually spatial) graph::

            (1 - a) * self + a * mean(neighbours)   [+ beta * 2-hop]

        With ``adaptive=True`` the per-cell weight ``a_i = min(0.85, k0/(k0+counts_i))`` so
        low-count (noisy) cells borrow heavily from neighbours while high-count cells barely
        move — robust across sparsity regimes. Strongly improves cell typing on sparse and/or
        spatially organised data (e.g. low-depth Xenium), but can contaminate identity in
        intermixed tissue, so it is **opt-in and OFF by default**. It also gives previously
        near-empty cells a usable vector. Call BEFORE :meth:`get_adata` /
        :meth:`phenotype_probability`.

        :param graph: scipy sparse adjacency (cells x cells) aligned to the cell-matrix order.
        :param alpha: smoothing weight (``adaptive=False``) or the per-cell cap (``adaptive=True``).
        :param adaptive: scale the weight per cell by inverse count.
        :param counts: per-cell totals; if None, derived from the loaded expression context.
        :param k0: adaptive midpoint; defaults to the median of positive counts.
        :param beta: optional 2-hop weight.
        :return: the new self.matrix (list of vectors). Original kept in self.uncorrected_matrix.
        """
        out = self._graph_smooth(self.matrix, graph, alpha=alpha, adaptive=adaptive,
                                 counts=counts, k0=k0, beta=beta)
        self.uncorrected_matrix = self.matrix
        self.matrix = [out[i] for i in range(out.shape[0])]
        return self.matrix

    def qc_marker_dict(self, adata, phenotype_markers, layer=None,
                       specificity_threshold=0.5, min_markers=2, verbose=True):
        """Quality-control a marker dictionary before phenotyping.

        Computes a provisional mean-expression labelling and, for every (phenotype, marker),
        reports whether the marker is in the panel, its mean expression, fraction of cells
        expressing it, a fold ``enrichment`` over the global mean, and a proportion-independent
        ``specificity`` (tau index in [0, 1] over the per-type means; 1 = specific to one type,
        0 = uniform). Flags low-specificity markers (tau below ``specificity_threshold``),
        off-target markers (highest in a different type than declared) and phenotypes with too
        few usable markers. Tau is used (not raw fold-enrichment) because fold-enrichment is
        confounded by class proportions and by dense/normalised data.

        :param adata: AnnData with expression (raw counts recommended via ``layer``).
        :param phenotype_markers: dict of phenotype -> marker gene list.
        :param layer: layer to read expression from (e.g. "counts"); defaults to ``.X``.
        :param specificity_threshold: tau below this flags a marker as low-specificity.
        :param min_markers: warn if a phenotype has fewer usable markers than this.
        :param verbose: log warnings.
        :return: pandas.DataFrame with one row per (phenotype, marker) and the QC columns.
        :rtype: pandas.DataFrame
        """
        X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
        X = numpy.asarray(X.todense()) if hasattr(X, "todense") else numpy.asarray(X)
        u2i = {str(g).upper(): i for i, g in enumerate(adata.var.index)}

        # provisional labels via z-scored mean marker expression argmax
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        names, cols = [], []
        for ct, genes in phenotype_markers.items():
            idx = [u2i[g.upper()] for g in genes if g.upper() in u2i]
            if idx:
                names.append(ct)
                cols.append(Z[:, idx].mean(1))
        prov = (numpy.array([names[i] for i in numpy.stack(cols, 1).argmax(1)])
                if cols else numpy.array(["?"] * adata.n_obs))

        ct_list = list(phenotype_markers)
        type_mean = {ct: X[prov == ct].mean(0) if (prov == ct).any() else numpy.zeros(X.shape[1])
                     for ct in ct_list}
        global_mean = X.mean(0) + 1e-9
        frac_expr = (X > 0).mean(0)
        K = max(len(ct_list), 2)

        def tau(gi):
            vals = numpy.array([type_mean[c][gi] for c in ct_list], dtype=float)
            mx = vals.max()
            if mx <= 0:
                return 0.0
            return float(numpy.sum(1.0 - vals / mx) / (K - 1))

        rows = []
        usable = collections.Counter()
        for ct, genes in phenotype_markers.items():
            for g in genes:
                gi = u2i.get(g.upper())
                if gi is None:
                    rows.append(dict(phenotype=ct, marker=g, in_panel=False, mean_expr=numpy.nan,
                                     frac_expressing=numpy.nan, enrichment=numpy.nan,
                                     specificity=numpy.nan, top_type=None, flag="absent"))
                    continue
                usable[ct] += 1
                spec = tau(gi)
                top = max(ct_list, key=lambda c: type_mean[c][gi])
                flag = "ok"
                if top != ct:
                    flag = f"off_target(>{top})"
                elif spec < specificity_threshold:
                    flag = "low_specificity"
                rows.append(dict(phenotype=ct, marker=g, in_panel=True,
                                 mean_expr=round(float(X[:, gi].mean()), 4),
                                 frac_expressing=round(float(frac_expr[gi]), 3),
                                 enrichment=round(float(type_mean[ct][gi] / global_mean[gi]), 3),
                                 specificity=round(spec, 3), top_type=top, flag=flag))
        df = pandas.DataFrame(rows)
        if verbose:
            for ct in phenotype_markers:
                if usable[ct] < min_markers:
                    logger.warning(f"Phenotype '{ct}' has only {usable[ct]} usable marker(s) "
                                   f"(< {min_markers}); assignment will be unreliable.")
            for _, r in df[df["flag"].isin(["low_specificity"]) | df["flag"].str.startswith("off_target")].iterrows():
                logger.warning(f"Marker '{r['marker']}' for '{r['phenotype']}': {r['flag']} "
                               f"(specificity={r['specificity']}, highest in {r['top_type']}).")
            for _, r in df[df["flag"] == "absent"].iterrows():
                logger.warning(f"Marker '{r['marker']}' for '{r['phenotype']}' not in panel.")
        return df


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
            logger.info(ph)
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
        Populates self.normalized_vectors from the count matrix during construction.
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
        logger.info("Loading embedding in X_genevector.")

        # Ensure self.data and self.matrix are populated from __init__
        if not self.data or not self.matrix:
            logger.error("CellEmbedding data or matrix not initialized. Run constructor properly.")
            # Or, attempt to run parts of __init__ if feasible, though better to ensure constructor worked.
            # For now, assume __init__ was successful.

        current_adata = self.context.adata.copy()
        
        # Filter adata to include only cells for which we have embeddings (keys in self.data)
        # This makes current_adata.obs.index consistent with self.data.keys() and self.matrix row order.
        cells_with_embeddings = list(self.data.keys())
        if not cells_with_embeddings:
            logger.error("No cells with embeddings found in self.data. Cannot proceed with get_adata.")
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
                logger.warning(f"Could not find vector for cell {cell_id_in_adata} in self.matrix. Using zero vector.")
                # Determine vector size from first vector in self.matrix or a default
                vec_size = self.matrix[0].shape[0] if self.matrix and len(self.matrix[0]) > 0 else 100
                x_genevector_list.append(np.zeros(vec_size))


        current_adata.obsm['X_genevector'] = np.array(x_genevector_list)
        
        logger.info("Running Scanpy neighbors and umap.")
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
