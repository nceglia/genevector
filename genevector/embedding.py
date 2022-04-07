from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas
import matplotlib.cm as cm
import umap
import tqdm
import scanpy as sc
import matplotlib.gridspec as gridspec
import networkx as nx
import matplotlib as mpl

import numpy
import operator
import random
import pickle
import collections
import sys
import os

class GeneEmbedding(object):

    def __init__(self, embedding_file, dataset, vector="1"):
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

    def select_cosine_threshold(self,plot=None):
        gene_sets = set()
        cosine = []
        sill = []
        cosine_max = 0.0
        max_score = 0.0
        for i in numpy.linspace(0.0,1,100):
            clustering = AgglomerativeClustering(affinity="cosine",
                                                 linkage="complete",
                                                 distance_threshold=i,
                                                 n_clusters=None).fit(self.vector)
            clusters = collections.defaultdict(list)
            if len(set(clustering.labels_)) > 1 and len(set(clustering.labels_)) != len(clustering.labels_):
                score = metrics.silhouette_score(self.vector, clustering.labels_, metric='cosine')
                sill.append(score)
                cosine.append(i)
                if score > max_score:
                    cosine_max = i
                    max_score = score
        self.cosine_threshold = cosine_max

        if plot:
            sns.set(font_scale=0.6)
            params = {'legend.fontsize': 'small',
                      'axes.labelsize': 'small',
                      'axes.titlesize':'small'}
            fig, ax = plt.subplots(1,1,figsize=(5,1))
            sns.lineplot(x=cosine,y=sill,color="g",ax=ax)
            plt.vlines(cosine_max,ymin=0,ymax=max(sill)+0.05,color="blue")
            ax.set_title("PBMC")
            ax.set_xlim(0,1)
            ax.set_ylim(0,max(sill)+0.05)
            ax.set_ylabel("Silhouette Coefficient")
            ax.set_xlabel("Cosine Similarity Threshold")
            plt.tight_layout()
            plt.savefig(plot)
        return cosine_max

    def read_embedding(self, filename):
        embedding = dict()
        lines = open(filename,"r").read().splitlines()[1:]
        for line in lines:
            vector = line.split()
            gene = vector.pop(0)
            embedding[gene] = [float(x) for x in vector]
        return embedding

    def plot_metagenes_scores(self, adata, metagenes, column, plot=None):
        similarity_matrix = []
        plt.figure(figsize = (5, 13))
        barcode_to_label = dict(zip(adata.obs.index, adata.obs[column]))
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
        sns.set(font_scale=0.3)
        sns.clustermap(df,figsize=(5,9), dendrogram_ratio=0.1,cmap="mako",yticklabels=True, standard_scale=0)
        plt.tight_layout()
        if plot:
            plt.savefig(plot)

    def score_metagenes(self,adata ,metagenes):
        for p, genes in metagenes.items():
            sc.tl.score_genes(adata,score_name=str(p)+"_SCORE",gene_list=genes)
            scores = numpy.array(adata.obs[str(p)+"_SCORE"].tolist()).reshape(-1,1)
            scaler = MinMaxScaler()
            scores = scaler.fit_transform(scores)
            scores = list(scores.reshape(1,-1))[0]
            adata.obs[str(p)+"_SCORE"] = scores

    def identify_metagenes(self, cosine=None, upper_bound=40):
        if cosine == None:
            cosine = self.select_cosine_threshold()
        clustering = AgglomerativeClustering(affinity="cosine",
                                             linkage="complete",
                                             distance_threshold=cosine,
                                             n_clusters=None).fit(self.vector)
        clusters = collections.defaultdict(list)
        for l,g in zip(clustering.labels_,self.genes):
            clusters[l].append(g)
        markers = dict()
        for cluster, genes in clusters.items():
            if len(set(genes)) > 1 and len(set(genes)) < upper_bound:
                markers["MG_{}".format(cluster)] = list(set(genes))
        self.cluster_definitions = markers
        return markers

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

    def generate_vector(self, genes):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes:
                vector.append(vec)
        assert len(vector) != 0, genes
        return list(numpy.median(vector, axis=0))

    def cluster_definitions_as_df(self, top_n=20):
        similarities = self.cluster_definitions
        clusters = []
        symbols = []
        for key, genes in similarities.items():
            clusters.append(key)
            symbols.append(", ".join(genes[:top_n]))
        df = pandas.DataFrame.from_dict({"Cluster Name":clusters, "Top Genes":symbols})
        return df

    def plot(self, png=None, method="TSNE", labels=[], pcs=None, remove=[]):
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1,1,1)
        pcs = self.plot_reduction(self.cluster_labels, ax, labels=labels, method=method, pcs=pcs, remove=remove)
        plt.show()
        return pcs

    def marker_labels(self,top_n=5):
        markers = []
        cluster_definitions = self.cluster_definitions
        marker_labels = dict()
        for gclust, genes in cluster_definitions.items():
            print(gclust, ",".join(genes[:5]))
            markers += genes[:top_n]
            for gene in genes[:top_n]:
                marker_labels[gene] = gclust
        return markers, marker_labels

    def plot_reduction(self, clusters, ax, method="TSNE", labels=[], pcs=None, remove=[]):
        if type(pcs) != numpy.ndarray:
            if method == "TSNE":
                print("Running t-SNE")
                pca = TSNE(n_components=2, n_jobs=-1, metric="cosine")
                pcs = pca.fit_transform(self.vector)
                pcs = numpy.transpose(pcs)
                print("Finished.")
            else:
                print("Running UMAP")
                trans = umap.UMAP(random_state=42,metric='cosine').fit(self.vector)
                x = trans.embedding_[:, 0]
                y = trans.embedding_[:, 1]
                pcs = [x,y]
                print("Finished.")
        if len(remove) != 0:
            _pcsx = []
            _pcsy = []
            _clusters = []
            for x, y, c in zip(pcs[0],pcs[1],clusters):
                if c not in remove:
                    _pcsx.append(x)
                    _pcsy.append(y)
                    _clusters.append(c)
            pcs = []
            pcs.append(_pcsx)
            pcs.append(_pcsy)
            clusters = _clusters
        data = {"x":pcs[0],"y":pcs[1], "Cluster":clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y",hue="Cluster", ax=ax)
        plt.xlabel("{}-1".format(method))
        plt.ylabel("{}-2".format(method))
        ax.set_xticks([])
        ax.set_yticks([])
        if len(labels):
            for x, y, gene in zip(pcs[0], pcs[1], self.context.expressed_genes):
                if gene in labels:
                    ax.text(x+.02, y, str(gene), fontsize=8)
        return pcs


    @staticmethod
    def read_vector(vec):
        lines = open(vec,"r").read().splitlines()
        dims = lines.pop(0)
        vecs = dict()
        for line in lines:
            line = line.split()
            gene = line.pop(0)
            vecs[gene] = list(map(float,line))
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

    def __init__(self, dataset, embed):
        self.context = dataset.data
        cell_to_gene = list(self.context.cell_to_gene.items())
        self.embed = embed
        self.expression = self.context.expression
        self.data = collections.defaultdict(list)
        self.weights = collections.defaultdict(list)
        self.pcs = dict()
        embed_genes = list(embed.embeddings.keys())
        self.matrix = []

        for cell, genes in tqdm.tqdm(cell_to_gene):
            cell_weights = self.expression[cell]
            cell_genes = []
            weights = []
            for g,w in cell_weights.items():
                if w != 0.0:
                    cell_genes.append(g)
                    weights.append(w)
            weights = numpy.array(weights)
            scaler = MinMaxScaler()
            scaled_weights = scaler.fit_transform(numpy.array(weights).reshape(-1,1))
            scaled_weights = [x + 1.0 for x in list(scaled_weights.reshape(1,-1)[0])]
            vectors = numpy.array([embed.embeddings[gene] for gene in cell_genes])
            self.matrix.append(numpy.average(vectors,axis=0,weights=scaled_weights))
            self.data[cell] = vectors
        self.dataset_vector = numpy.zeros(numpy.array(self.matrix).shape[1])

    def batch_correct(self, column=None, resolution=1, atten=1.0):
        if not column:
            raise ValueError("Must supply batch label to correct.")
        _clusters = []
        for cluster in self.matrix:
            _clusters.append("C1")
        self.clusters = _clusters
        column_labels = dict(zip(self.context.cells, self.context.metadata[column]))
        labels = []
        for key in self.data.keys():
            labels.append(column_labels[key])
        local_correction = collections.defaultdict(lambda : collections.defaultdict(list))
        correction_vectors = collections.defaultdict(dict)
        for cluster, batch, vec in zip(self.clusters, labels, self.matrix):
            local_correction[cluster][batch].append(vec)
        for cluster, batches in local_correction.items():
            cluster_vec = []
            batch_keys = list(batches.keys())
            base_batch = batch_keys.pop(0)
            max_distance = 1.0
            cluster_vec = numpy.average(batches[base_batch], axis=0)
            for batch in batch_keys:
                bvec = list(numpy.average(batches[batch], axis=0))
                distance = float(cosine_similarity(numpy.array(bvec).reshape(1, -1),numpy.array(cluster_vec).reshape(1, -1))[0])
                offset = numpy.subtract(cluster_vec,bvec)
                bvec = numpy.add(bvec,offset)
                distance = float(cosine_similarity(numpy.array(bvec).reshape(1, -1),numpy.array(cluster_vec).reshape(1, -1))[0])
                correction_vectors[cluster][batch] = offset

        self.matrix = []
        self.sample_vector = collections.defaultdict(list)
        i = 0
        self.cell_order = []
        for cell, vectors in self.data.items():
            cluster = self.clusters[i]
            xvec = list(numpy.average(vectors, axis=0))
            batch = column_labels[cell]
            if cluster in correction_vectors and batch in correction_vectors[cluster]:
                offset = correction_vectors[cluster][batch]
                offset = numpy.multiply(offset,atten)
                xvec = numpy.add(xvec,offset)
            self.matrix.append(xvec)
            self.cell_order.append(cell)
            i += 1

    def cluster(self, k=12):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.matrix)
        clusters = kmeans.labels_
        _clusters = []
        for cluster in clusters:
            _clusters.append("C"+str(cluster))
        self.clusters = _clusters
        return _clusters

    def cluster_definitions(self):
        gene_similarities = dict()
        vectors = collections.defaultdict(list)
        for vec, label in zip(self.matrix, self.clusters):
            vectors[label].append(vec)
        for label, vecs in vectors.items():
            distances = dict()
            cell_vector = list(numpy.mean(vecs, axis=0))
            for gene, vector in self.embed.embeddings.items():
                distance = float(cosine_similarity(numpy.array(cell_vector).reshape(1, -1),numpy.array(vector).reshape(1, -1))[0])
                distances[gene] = distance
            sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
            gene_similarities[label] = [x[0] for x in sorted_distances]
            print(label, sorted_distances[:10])
        return gene_similarities

    def cluster_definitions_as_df(self, similarities, top_n=20):
        clusters = []
        symbols = []
        for key, genes in similarities.items():
            clusters.append(key)
            symbols.append(", ".join(genes[:top_n]))
        df = pandas.DataFrame.from_dict({"Cluster Name":clusters, "Top Genes":symbols})
        return df

    def compute_cell_similarities(self, barcode_to_label):
        vectors = dict()
        cell_similarities = dict()
        vectors, labels = self._cell_vectors(barcode_to_label)
        for label, vector in vectors.items():
            distances = dict()
            for label2, vector2 in vectors.items():
                xdist = []
                distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(vector2).reshape(1, -1))[0])
                xdist.append(distance)
                distances[label2] = distance
            cell_similarities[label] = distances
        return cell_similarities

    def plot_reduction(self, ax, pcs=None, method="TSNE", clusters=None, labels=None):
        if type(pcs) != numpy.ndarray:
            if method == "TSNE":
                if method not in self.pcs:
                    print("Running t-SNE")
                    pca = TSNE(n_components=2, n_jobs=-1, metric="cosine")
                    pcs = pca.fit_transform(self.matrix)
                    pcs = numpy.transpose(pcs)
                    print("Finished.")
                    self.pcs[method] = pcs
                else:
                    print("Loading TSNE")
                    pcs = self.pcs[method]
            else:
                if method not in self.pcs:
                    print("Running UMAP")
                    trans = umap.UMAP(random_state=42,metric='cosine').fit(self.matrix)
                    x = trans.embedding_[:, 0]
                    y = trans.embedding_[:, 1]
                    pcs = [x,y]
                    print("Finished.")
                    self.pcs[method] = pcs
                else:
                    print("Loading UMAP")
                    pcs = self.pcs[method]
        data = {"x":pcs[0],"y":pcs[1],"Cluster": clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Cluster', ax=ax,linewidth=0.1,s=13,alpha=1.0)
        return pcs


    def plot(self, png=None, pcs=None, method="TSNE", column=None):
        if column:
            column_labels = dict(zip(self.context.cells,self.context.metadata[column]))
            labels = []
            for key in self.data.keys():
                labels.append(column_labels[key])
        else:
            labels = self.clusters
        plt.figure(figsize = (8, 8))
        ax1 = plt.subplot(1,1,1)
        pcs = self.plot_reduction(ax1, pcs=pcs, clusters=labels, method=method)
        plt.xlabel("{}-1".format(method))
        plt.ylabel("{}-2".format(method))
        ax1.set_xticks([])
        ax1.set_yticks([])
        if png:
            plt.savefig(png)
            plt.close()
        else:
            plt.show()
        return pcs

    def plot_distance(self, vector, pcs=None, threshold=0.0, method="TSNE", title=None, show=True):
        if type(pcs) != numpy.ndarray:
            if method not in self.pcs:
                if method == "TSNE":
                    pca = TSNE(n_components=2)
                    pcs = pca.fit_transform(self.matrix)
                    pcs = numpy.transpose(pcs)
                else:
                    trans = umap.UMAP(random_state=42,metric='cosine').fit(self.matrix)
                    x = trans.embedding_[:, 0]
                    y = trans.embedding_[:, 1]
                    pcs = [x,y]
            else:
                pcs = self.pcs[method]
        distances = []
        dataset_distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(self.dataset_vector).reshape(1, -1))[0])
        for cell_vector in self.matrix:
            distance = float(cosine_similarity(numpy.array(cell_vector).reshape(1, -1),numpy.array(vector).reshape(1, -1))[0])
            d = distance-dataset_distance
            if d < threshold:
                d = -1.0
            distances.append(d)
        data = {"x":pcs[0],"y":pcs[1],"Distance": distances}
        df = pandas.DataFrame.from_dict(data)
        if show:
            plt.figure(figsize = (8,8))
            ax = plt.subplot(1,1, 1)
            sns.scatterplot(data=df,x="x", y="y", hue='Distance', ax=ax,linewidth=0.00,s=7,alpha=0.7)
            if title != None:
                plt.title(title)
        return distances

    def phenotype_probability(self, phenotype_markers, method="softmax"):
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        adata = self.context.adata.copy()
        adata = adata[list(self.data.keys())]
        probs = dict()
        for pheno, markers in phenotype_markers.items():
            dists = []
            vector = self.embed.generate_vector(markers)
            ovecs = []
            for oph, ovec in phenotype_markers.items():
                ovec = embed.generate_vector(ovec)
                ovecs.append(ovec)
            aovec = numpy.average(ovecs,axis=0)
            vector = numpy.subtract(vector,aovec)
            for x in tqdm.tqdm(adata.obs.index):
                dist = 1.0 - distance.cosine(mapped_components[x],vector)
                dists.append(dist)
            probs[pheno] = dists
        distribution = []
        celltypes = []
        for k, v in probs.items():
            distribution.append(v)
            celltypes.append(k)
        distribution = list(zip(*distribution))
        classif = []
        probabilities = []
        if method=="normalized":
            scaler = MinMaxScaler()
            res = scaler.fit_transform(numpy.array(distribution))
            for ct in res:
                ct = ct / ct.sum()
                probabilities.append(ct)
                assign = celltypes[numpy.argmax(ct)]
                classif.append(assign)
        if method=="softmax":
            scaler = StandardScaler()
            probabilities = softmax(scaler.fit_transform(numpy.array(distribution)),axis=1)
            for ct in probabilities:
                assign = celltypes[numpy.argmax(ct)]
                classif.append(assign)
        umap_pts = dict(zip(list(self.data.keys()),classif))
        return {"distances":distribution, "order":celltypes, "probabilities":probabilities}

    def get_adata(self, min_dist=0.3, n_neighbors=50):
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
