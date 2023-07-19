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
    A class used to represent the learned gene embedding.

    This class provides an embedding (a representation in a lower-dimensional space) for genes,
    which can be used for tasks such as similarity computation, visualization, etc.

    ...

    Methods
    -------
    __init__(self, embedding_file, dataset, vector):
        Initializes the GeneEmbedding object with an embedding file learned from GeneVector model and a GeneVectorDataset object generated from an AnnData object.
    
    get_adata():
        Returns the AnnData object holding the learned gene embedding.

    plot_similarities():
        Plots a similarity matrix of the genes based on their embeddings.
    """

    def __init__(self, embedding_file, dataset, vector="average"):
        """
        Initialize the GeneEmbedding object.

        Parameters
        ----------
        embedding_file : str
            Specifies the path to a set of .vec files generated for model training.
        dataset : GeneVectorDataset
            The GeneVectorDataset object that was constructed from the original AnnData object.
        vector : str
            Specifies if using the first set of weights ("1"), the second set of weights ("2"), or the average ("average"). This should be set to "average".
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
        Get the AnnData object holding the learned gene embedding.

        This method returns the AnnData object that contains the gene embedding with leiden clusters for metagenes, the neighbors graph, and the UMAP embedding.
       
        Parameters
        ----------
        resolution : float
            The resolution to pass to the sc.tl.leiden function.
        
        Returns
        -------
        AnnData
            An AnnData object with metagenes stored in 'leiden' for the provided resolution, the neighbors graph, and UMAP embedding.
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

        Parameters
        ----------
        gene : str
            The gene symbol of the gene of interest.
        n_genes : int
            The number of most similar genes to plot.
        save : str
            The path to save the figure (optional).

        Returns
        -------
        matplotlib.figure.ax
            A matplotlib axes object representing the plot.
        """
        df = self.compute_similarities(gene).head(n_genes)
        _,ax = plt.subplots(1,1,figsize=(3,6))
        sns.barplot(data=df,y="Gene",x="Similarity",palette="magma_r",ax=ax)
        ax.set_title("{} Similarity".format(gene))
        if save != None:
            plt.savefig(save)
        return ax

    def plot_metagene(self, gdata, mg=None, title="Gene Embedding"):
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
        metagenes = collections.defaultdict(list)
        for x,y in zip(gdata.obs["leiden"],gdata.obs.index):
            metagenes[x].append(y)
        return metagenes

    def compute_similarities(self, gene, subset=None, feature_type=None):
        if gene not in self.embeddings:
            return None
        if feature_type:
            subset = []
            for gene in list(self.embeddings.keys()):
                if feature_type == self.context.feature_types[gene]:
                    subset.append(gene)
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

    def clusters(self, clusters):
        average_vector = dict()
        gene_to_cluster = collections.defaultdict(list)
        matrix = collections.defaultdict(list)
        total_average_vector = []
        for gene, cluster in zip(self.context.expressed_genes, clusters):
            if gene in self.embeddings:
                matrix[cluster].append(self.embeddings[gene])
                gene_to_cluster[cluster].append(gene)
                total_average_vector.append(self.embeddings[gene])
        self.total_average_vector = list(numpy.average(total_average_vector, axis=0))
        for cluster, vectors in matrix.items():
            xvec = list(numpy.average(vectors, axis=0))
            average_vector[cluster] = numpy.subtract(xvec,self.total_average_vector)
        return average_vector, gene_to_cluster

    def generate_weighted_vector(self, genes, markers, weights):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(weights[gene] * numpy.array(vec))
            if gene not in genes and gene in markers and gene in weights:
                vector.append(list(weights[gene] * numpy.negative(numpy.array(vec))))
        return list(numpy.sum(vector, axis=0))


    def generate_vector(self, genes):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes:
                vector.append(vec)
        assert len(vector) != 0, genes
        return list(numpy.average(vector, axis=0))

    def generate_weighted_vector(self, genes, weights):
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

    def __init__(self, dataset, embed, log_normalize=True):

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

    def compute_cell_similarities(self, barcode_to_label):
        vectors = dict()
        cell_similarities = dict()
        vectors, _ = self._cell_vectors(barcode_to_label)
        for label, vector in vectors.items():
            distances = dict()
            for label2, vector2 in vectors.items():
                xdist = []
                distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(vector2).reshape(1, -1))[0])
                xdist.append(distance)
                distances[label2] = distance
            cell_similarities[label] = distances
        return cell_similarities

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

    def phenotype_probability(self, adata, phenotype_markers, return_distances=True, expression_weighted=True, target_col="genevector"):
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


    def get_adata(self, min_dist=0.3, n_neighbors=50):
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


    def group_cell_vectors(self, label):
        barcode_to_label = dict(zip(self.context.adata.obs.index.tolist(),self.context.adata.obs[label]))
        label_vector = dict()
        labels = []
        for cell, vectors in self.data.items():
            vector = list(numpy.median(vectors, axis=0))
            labels.append(barcode_to_label[cell])
            label_vector[barcode_to_label[cell]] = vector
        for cell, vectors in self.data.items():
            _vectors = []
            for vector in vectors:
                _vectors.append(numpy.subtract(vector, label_vector[barcode_to_label[cell]]))
            vectors = _vectors
            vector = list(numpy.median(vectors, axis=0))
            label_vector[barcode_to_label[cell]] = vector
        return label_vector, labels
