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

    def __init__(self, embedding_file, dataset, vector="average"):
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
        self.context = dataset.data
        self.embedding_file = embedding_file
        self.vector = []
        self.genes = []
        for gene in tqdm.tqdm(self.embeddings.keys()):
            self.vector.append(self.embeddings[gene])
            self.genes.append(gene)

    def read_embedding(self, filename):
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
        self.data = collections.defaultdict(list)
        self.matrix = []
        self.expression = collections.defaultdict(dict)
        adata = self.context.adata.copy()

        if log_normalize:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        
        genes = adata.var.index.to_numpy()
        barcodes = adata.obs.index.to_numpy()

        matrix = csr_matrix(adata.X)
        self.normalized_vectors = collections.defaultdict(list)
        self.normalized_marker_expression = collections.defaultdict(dict)
        self.normalized_expression = self.normalized_expression(matrix, genes, barcodes)
        self.normalized_expression_values = None

        print(bcolors.OKGREEN + "Generating Cell Vectors." + bcolors.ENDC)
        cells_with_no_counts = 0
        for cell in tqdm.tqdm(adata.obs.index.tolist()):
            try:
                vectors, weights = zip(*self.normalized_vectors[cell])
            except Exception as e:
                cells_with_no_counts += 1
                continue
            self.data[cell] = vectors
            self.matrix.append(numpy.average(vectors,axis=0,weights=weights))
        print("Found {} Cells with No Counts.".format(cells_with_no_counts))
        self.dataset_vector = numpy.zeros(numpy.array(self.matrix).shape[1])
        print(bcolors.BOLD + "Finished." + bcolors.ENDC)

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
        normalized_expression = collections.defaultdict(dict)
        normalized_matrix.eliminate_zeros()
        row_indices, column_indices = normalized_matrix.nonzero()
        nonzero_values = normalized_matrix.data
        entries = list(zip(nonzero_values, row_indices, column_indices))
        for value, i, j in tqdm.tqdm(entries):
            if value > 0 and genes[j].upper() in self.embed.embeddings and genes[j] in markers:
                normalized_expression[cells[i]][genes[j]] = value
        return normalized_expression

    def phenotype_probability(self, adata, phenotype_markers, return_distances=False, expression_weighted=False, target_col="genevector"):
        """
        Probablistically assign phenotypes based on a set of cell type labels and associated markers. 
        Can optionally return the original cosine distances and perform the assignment based on expression weight gene vectors.
        Loads into the anndata the pseudo-probabilities for each cell type and the deterministic label taken from the maximum probability over cell types.

        :param adata: anndata object generated from "get_adata", has "X_genevector" in the obsm dataframe.
        :type column: anndata.AnnData
        :param phenotype_markers: Dictionary of cell type labels (key) to gene markers used to define the cell type as a list (value).
        :type phenotype_markers: dict
        :param return_distances: Change the return type to a tuple that includes a dictionary containing the actual cosine distances alongside the phenotype probabilities.
        :type column: bool
        :param expression_weighted: Compute similarit to each cell using the expression weightedy marker gnene vector.
        :type column: bool
        :param target_col: Column label to load in deterministic cell asssignments in the obs data frame of the anndata object.
        :type target_col: bool
        :return: Anndata with cell type labels and probabilities, or optionally a tuple with the anndata and the raw cosine similarities.
        :rtype:  anndata.AnnData
        """
        def normalized_marker_expression_sub(self, normalized_matrix, genes, cells, markers):
            normalized_expression = collections.defaultdict(dict)
            normalized_matrix.eliminate_zeros()
            row_indices, column_indices = normalized_matrix.nonzero()
            nonzero_values = normalized_matrix.data
            entries = list(zip(nonzero_values, row_indices, column_indices))
            for value, i, j in tqdm.tqdm(entries):
                if value > 0 and genes[j].upper() in self.embed.embeddings and genes[j] in markers:
                    normalized_expression[cells[i]][genes[j]] = value
            return normalized_expression

        if expression_weighted:
            for x in adata.obs.columns:
                if "Pseudo-probability" in x:
                    del adata.obs[x]
            mapped_components = dict(zip(list(self.data.keys()),self.matrix))
            genes = adata.var.index.to_list()
            cells = adata.obs.index.to_list()
            matrix = csr_matrix(adata.X)
            embedding = csr_matrix(self.matrix)
            all_markers = []
            for _, markers in phenotype_markers.items():
                all_markers += markers  
            all_markers = list(set(all_markers))
            normalized_expression = normalized_marker_expression_sub(self, matrix, genes, cells, all_markers)
            probs = dict()

            def generate_weighted_vector(embed, genes, weights):
                vector = []
                for gene, vec in zip(embed.genes, embed.vector):
                    if gene in genes and gene in weights:
                        vector.append(weights[gene] * numpy.array(vec))
                if numpy.sum(vector) == 0:
                    return None
                else:
                    return list(numpy.mean(vector, axis=0))
            matrix = matrix.todense()

            for pheno, markers in phenotype_markers.items():
                dists = []
                print(bcolors.OKBLUE+"Computing similarities for {}".format(pheno)+bcolors.ENDC)
                print(bcolors.OKGREEN+"Markers: {}".format(", ".join(markers))+bcolors.ENDC)
                odists = []
                for x in tqdm.tqdm(adata.obs.index):
                    weights = normalized_expression[x]
                    vector = generate_weighted_vector(self.embed, markers, weights)
                    if vector != None:
                        dist = 1. - distance.cosine(mapped_components[x], numpy.array(vector))
                        odists.append(dist)
                    else:
                        odists.append(0.)
                probs[pheno] = odists
            distribution = []
            celltypes = []
            for k, v in probs.items():
                distribution.append(v)
                celltypes.append(k)
            distribution = list(zip(*distribution))
            probabilities = softmax(numpy.array(distribution),axis=1)
            res = {"distances":distribution, "order":celltypes, "probabilities":probabilities}
            barcode_to_label = dict(zip(list(self.data.keys()), res["probabilities"]))
            ct = []
            probs = collections.defaultdict(list)
            for x in adata.obs.index:
                ctx = res["order"][numpy.argmax(barcode_to_label[x])]
                ct.append(ctx)
                for ph, pb in zip(res["order"],barcode_to_label[x]):
                    probs[ph].append(pb)
            adata.obs[target_col] = ct
            def load_predictions(adata,probs):
                for ph in probs.keys():
                    adata.obs[ph+" Pseudo-probability"] = probs[ph]
                return adata
            adata = load_predictions(adata,probs)
            prob_cols = [x for x in adata.obs.columns if "Pseudo" in x]
            cts = adata.obs[target_col].tolist()
            probs = adata.obs[prob_cols].to_numpy()
            adj_cts = []
            for ct,p in zip(cts,probs):
                if len(set(p)) == 1:
                    adj_cts.append("Unknown")
                else:
                    adj_cts.append(ct)
            adata.obs[target_col] = adj_cts
            if return_distances:
                return adata, res
            else:
                return adata
        else:
            for x in adata.obs.columns:
                if "Pseudo-probability" in x:
                    del adata.obs[x]
            mapped_components = dict(zip(list(self.data.keys()),self.matrix))
            genes = adata.var.index.to_list()
            cells = adata.obs.index.to_list()
            matrix = csr_matrix(adata.X)
            all_markers = []
            for _, markers in phenotype_markers.items():
                all_markers += markers  
            all_markers = list(set(all_markers))
            probs = dict()

            for pheno, markers in phenotype_markers.items():
                print(bcolors.OKBLUE+"Computing similarities for {}".format(pheno)+bcolors.ENDC)
                print(bcolors.OKGREEN+"Markers: {}".format(", ".join(markers))+bcolors.ENDC)
                odists = []
                for x in tqdm.tqdm(adata.obs.index):
                    vector = self.embed.generate_vector(markers)
                    dist = 1. - distance.cosine(mapped_components[x], numpy.array(vector))
                    odists.append(dist)
                probs[pheno] = odists
            distribution = []
            celltypes = []
            for k, v in probs.items():
                distribution.append(v)
                celltypes.append(k)
            distribution = list(zip(*distribution))
            probabilities = softmax(numpy.array(distribution),axis=1)
            res = {"distances":distribution, "order":celltypes, "probabilities":probabilities}
            barcode_to_label = dict(zip(list(self.data.keys()), res["probabilities"]))
            ct = []
            probs = collections.defaultdict(list)
            for x in adata.obs.index:
                ctx = res["order"][numpy.argmax(barcode_to_label[x])]
                ct.append(ctx)
                for ph, pb in zip(res["order"],barcode_to_label[x]):
                    probs[ph].append(pb)
            adata.obs[target_col] = ct
            def load_predictions(adata,probs):
                for ph in probs.keys():
                    adata.obs[ph+" Pseudo-probability"] = probs[ph]
                return adata
            adata = load_predictions(adata,probs)
            if return_distances:
                return adata, res
            else:
                return adata

    def cluster(self, adata, up_markers, down_markers=dict()):
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
        adata.obs["genevector"] = ["C"+str(x) for x in gm.predict(dist)]
        probs = gm.predict_proba(dist) 
        for c, prob in zip(set(adata.obs["genevector"]), probs.T):
            adata.obs["{}_probability".format(c)] = prob
        return adata


    def get_adata(self, min_dist=0.3, n_neighbors=50):
        """
        Return a anndata object to use in downstream analyses that contains the cell embedding matrix (under "X_genevector" in obsm) alongisde the neighbors graph and UMAP embedding computed using the cell vectors.

        :param min_dist: UMAP generation parameter.
        :type min_dist: float
        :param n_neighbors: Number of neighbors defined by cosine dsimilarity to include in neghborhood graph.
        :type: n_neighbors: int
        :return: Anndata with cell embedding stored in metadata ("obsm").
        :rtype:  anndata.AnnData
        """
        print(bcolors.OKGREEN + "Loading embedding in X_genevector." + bcolors.ENDC)
        print(bcolors.OKGREEN + "Running Scanpy neighbors and umap." + bcolors.ENDC)
        adata = self.context.adata.copy()
        adata = adata[list(self.data.keys())]
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        x_genevector = []
        for x in adata.obs.index:
            x_genevector.append(mapped_components[x])
        adata.obsm['X_genevector'] = numpy.array(x_genevector)
        sc.pp.neighbors(adata, use_rep="X_genevector", n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist)
        self.adata = adata
        return adata

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
