from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    def __init__(self, embedding_file, compass_dataset, vector="1"):
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
        self.context = compass_dataset.data
        self.embedding_file = embedding_file
        self.vector = []
        self.genes = []
        for gene in tqdm.tqdm(self.embeddings.keys()):
            # if gene in self.embeddings:
            self.vector.append(self.embeddings[gene])
            self.genes.append(gene)

    def read_embedding(self, filename):
        embedding = dict()
        lines = open(filename,"r").read().splitlines()[1:]
        for line in lines:
            vector = line.split()
            gene = vector.pop(0)
            embedding[gene] = [float(x) for x in vector]
        return embedding

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

    def cluster(self, threshold=0.75, lower_bound=1):
        cluster_definitions = collections.defaultdict(list)
        G = embed.generate_network(threshold=threshold)
        G.remove_edges_from(networkx.selfloop_edges(G))
        for i, connected_component in enumerate(networkx.connected_components(G)):
            subg = G.subgraph(connected_component)
            if len(subg.nodes()) > lower_bound:
                # if len(subg.nodes()) == 2:
                #     cluster_definitions[str(i+j+100)] += list(subg.nodes())
                #     continue
                clique_tree = networkx.tree.junction_tree(subg)
                clique_tree.remove_nodes_from(list(networkx.isolates(clique_tree)))
                for j, cc in enumerate(nx.connected_components(clique_tree)):
                    for clique_cc in cc:
                        cluster_definitions[str(i+j)] += list(set(clique_cc))
        self.cluster_definitions = cluster_definitions
        return self.cluster_definitions

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
        # if png:
        #     plt.savefig(png)
        #     plt.close()
        # else:
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

    def subtract_vector(self, vector):
        for gene, vec in self.embeddings.items():
            vec = numpy.subtract(vec-vector)
            self.embeddings[gene] = vec

    def relabel_clusters(self, clusters, annotations):
        _clusters = []
        for cluster in clusters:
            if cluster in annotations:
                _clusters.append(annotations[cluster])
            else:
                _clusters.append(cluster)
        self.cluster_labels = _clusters
        return _clusters

    def plot_similarity_matrix(self, top_n=5, png=None):
        markers, marker_labels = self.marker_labels(top_n=top_n)
        cmap = matplotlib.cm.tab20
        node_color = {}
        type_color = {}
        ctypes = []
        for marker in markers:
            if marker_labels:
                ctypes = []
                for value in marker_labels.values():
                    ctypes.append(value)
                ctypes = list(set(ctypes))
                node_color[marker] = cmap(ctypes.index(marker_labels[marker]))
                type_color[marker_labels[marker]] = cmap(ctypes.index(marker_labels[marker]))
        mm = pandas.DataFrame(markers, index=markers)
        mm["Gene Cluster"] = mm[0]
        row_colors = mm["Gene Cluster"].map(node_color)
        similarity_matrix = []
        markers = set(list(self.embeddings.keys())).intersection(set(markers))
        markers = list(markers)
        for marker in markers:
            row = []
            res = self.compute_similarities(marker, subset=markers)
            resdict = dict(zip(res["Gene"],res["Similarity"]))
            for gene in markers:
                row.append(resdict[gene])
            similarity_matrix.append(row)
        from matplotlib.patches import Patch
        plt.figure()
        matrix = numpy.array(similarity_matrix)
        df = pandas.DataFrame(matrix,index=markers,columns=markers)
        sns.clustermap(df,cbar_pos=None,figsize=(12,12), dendrogram_ratio=0.1, cmap="mako",row_colors=row_colors,yticklabels=True,xticklabels=True)
        handles = [Patch(facecolor=type_color[name]) for name in type_color]
        plt.legend(handles, type_color, title='Gene Cluster',
                bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
        plt.tight_layout()
        if png:
            plt.savefig("marker_similarity.png")
        else:
            plt.show()

    def similarity_network(self,top_n=5):
        markers, marker_labels = self.marker_labels(top_n=top_n)
        G = nx.Graph()
        for marker in markers:
            G.add_node(marker)
        for marker in markers:
            res = self.compute_similarities(marker)
            resdict = dict(zip(res["Gene"],res["Similarity"]))
            for gene, similarity in resdict.items():
                if gene != marker:
                    if gene not in G.nodes():
                        G.add_node(gene)
                    G.add_edge(marker, gene, weight=similarity)
        return G


    def plot_similarity_network(self, top_n=5, png=None):
        markers, marker_labels = self.marker_labels(top_n=top_n)
        cmap = matplotlib.cm.tab20
        G = nx.petersen_graph()
        node_color = []
        node_order = []
        node_size = []
        edge_order = []
        edge_color = []
        edge_labels = dict()
        for marker in markers:
            node_order.append(marker)
            if marker_labels:
                ctypes = []
                for value in marker_labels.values():
                    ctypes.append(value)
                ctypes = list(set(ctypes))
                node_color.append(ctypes.index(marker_labels[marker]))
            node_size.append(400)
            G.add_node(marker)
        for marker in markers:
            res = self.compute_similarities(marker)
            resdict = dict(zip(res["Gene"],res["Similarity"]))
            i = 0
            for gene, similarity in resdict.items():
                if i > 9: break
                if gene != marker:
                    if gene not in G.nodes():
                        node_size.append(0)
                        G.add_node(gene)
                        node_order.append(gene)
                        node_color.append(len(set(marker_labels.values())))
                    G.add_edge(marker, gene, weight=similarity)
                    edge_color.append(similarity)
                    edge_order.append((marker,gene))
                    edge_labels[(marker,gene)] = str(round(similarity,2))
                    i += 1
        # print(node_color)
        # c = max(nx.connected_components(G), key=len)
        # G = G.subgraph(c).copy()
        for i in range(10):
            G.remove_node(i)
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(1,1,1)
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato",args="-Goverlap=scale -Elen=5 -Eweight=0.2")
        nx.draw(G,pos,ax=ax, cmap=cmap,nodelist=node_order,
                             node_size=node_size,
                             edgelist=edge_order,
                             node_color=node_color,
                             edge_color=edge_color,
                             edge_vmin=0,
                             edge_vmax=1.0,
                             edge_cmap=plt.cm.Greys,
                             with_labels=True, width=1,font_size=7)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels, font_size=6)
        plt.axis('off')
        plt.tight_layout()
        if png:
            plt.savefig(png)
        else:
            plt.show()
        return G

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

    @staticmethod
    def cluster_network(genes, nxg, threshold=0.0, title="", display=True):
        G = nxg.subgraph(genes)
        for subg in nx.connected_components(G):
            if len(subg) > 1:
                if display:
                    fig = plt.figure(figsize=(14,6))
                    ax = plt.subplot(1,2,1)
                    subG = G.subgraph(list(subg))
                    centrality = dict(nx.betweenness_centrality(subG))
                    low, *_, high = sorted(centrality.values())
                    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
                    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
                    pos = nx.nx_agraph.graphviz_layout(subG, prog="neato",args="-Goverlap=scale -Elen=5 -Eweight=0.2")
                    nx.draw(subG,pos,with_labels=True,
                            node_color=[mapper.to_rgba(i) for i in centrality.values()],
                            node_size=100,ax=ax,font_size=16,
                            edge_color=[[0.5,0.5,0.5,0.5] for _ in subG.edges()])
                    vector = embed.generate_vector(list(subg))
                    ax = plt.subplot(1,2,2)
                    pcs = self.pcs["UMAP"]
                    distances = []
                    dataset_distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(self.dataset_vector).reshape(1, -1))[0])
                    for cell_vector in self.matrix:
                        distance = float(cosine_similarity(numpy.array(cell_vector).reshape(1, -1),numpy.array(vector).reshape(1, -1))[0])
                        d = distance-dataset_distance
                        if d < threshold:
                            d = threshold
                        distances.append(d)
                    data = {"x":pcs[0],"y":pcs[1],"Distance": distances}
                    df = pandas.DataFrame.from_dict(data)
                    sns.scatterplot(data=df,x="x", y="y", hue='Distance', ax=ax,linewidth=0.00,s=7,alpha=0.7)
                    if title != None:
                        plt.title(title)
                    plt.show()

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

    def __init__(self, compass_dataset, embed):
        self.context = compass_dataset.data
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
        self.cluster(k=resolution)
        column_labels = dict(zip(self.context.cells,self.context.metadata[column]))
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

    def annotate_clusters(self, annotations):
        _clusters = []
        for cluster in self.clusters:
            if cluster in annotations:
                _clusters.append(annotations[cluster])
            else:
                _clusters.append(cluster)
        self.clusters = _clusters

    def subtract_vector(self, vector):
        corrected_matrix = []
        for cell_vector in self.matrix:
            corrected_matrix.append(numpy.subtract(cell_vector, vector))
        self.matrix = corrected_matrix

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

    def plot_distance_grid(self, vectors, labels, pcs=None, threshold=-1.0, method="UMAP"):
        plt.figure(figsize = (8,8))
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
        i = 0
        num_plots = len(vectors)
        import math
        square = int(math.ceil(math.sqrt(num_plots)))
        for vector, label in zip(vectors, labels):
            ax = plt.subplot(square,square,i+1)
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
            sns.scatterplot(data=df,x="x", y="y", hue='Distance', ax=ax,linewidth=0.00,s=7,alpha=0.7)
            ax.set_title(label)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.get_legend().remove()
            i += 1
        plt.tight_layout()
        plt.show()

    def plot_gene_tsne(self, title, ax, genes, pcs=None):
        expression = [0 for _ in range(len(list(self.data.keys())))]
        for gene in genes:
            for i, cell in enumerate(self.data.keys()):
                if gene in self.expression[cell]:
                    expression[i] += self.expression[cell][gene]
        if type(pcs) != numpy.ndarray:
            pca = TSNE(n_components=2)
            pcs = pca.fit_transform(self.matrix)
            pcs = numpy.transpose(pcs)
        data = {"x":pcs[0],"y":pcs[1],"Gene Expression": expression}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Gene Expression', ax=ax,linewidth=0.00,s=7,alpha=0.7)
        ax.set_title(title,fontsize=16)
        return pcs

    def plot_gene_expression(self, genes, pcs=None, png=None):
        plt.figure(figsize = (8,8))
        ax = plt.subplot(1,1, 1)
        pcs = self.plot_gene_tsne(",".join(genes[:10]), ax, genes, pcs=pcs)
        ax.set_xticks([])
        ax.set_yticks([])
        if not png:
            plt.show()
        else:
            plt.savefig(png)
            plt.close()
        return pcs

    def plot_similarity_matrix(self, vectors, column):
        similarity_matrix = []
        plt.figure(figsize = (12, 10))
        barcode_to_label = dict(zip(self.context.metadata.index, self.context.metadata[column]))
        ctypes = self._cell_vectors()
        matrix = []
        clusters = list(vectors.keys())
        celltypes = list(cytpes.keys())
        for cluster, genes in vectors.items():
            vector = embed.generate_vector(genes)
            row = []
            for cell in ctypes.keys():
                distance = float(cosine_similarity(numpy.array(ctypes[cell]).reshape(1, -1),numpy.array(vector).reshape(1, -1))[0])
                row.append()
            matrix.append(row)
        matrix = numpy.array(matrix)
        df = pandas.DataFrame(matrix,index=celltypes,columns=celltypes)
        sns.clustermap(df,figsize=(17,8))
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,"celltype_similarities_{}.png".format(sample)))

    def phenotype_probability(self, phenotype_markers, method="softmax"):
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        adata = self.context.adata.copy()
        adata = adata[list(self.data.keys())]
        probs = dict()
        for pheno, markers in phenotype_markers.items():
            dists = []
            vector = self.embed.generate_vector(markers)
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
            probabilities = softmax(distribution,axis=1)
            for ct in probabilities:
                assign = celltypes[numpy.argmax(ct)]
                classif.append(assign)
        umap_pts = dict(zip(list(self.data.keys()),classif))
        nu = []
        for bc in adata.obs.index.tolist():
            nu.append(umap_pts[bc])
        adata.obs["cell_type"] = nu
        return {"distances":distribution, "order":celltypes, "probabilities":probabilities}

    def get_adata(self, min_dist=0.3, n_neighbors=50):
        adata = self.context.adata.copy()
        adata = adata[list(self.data.keys())]
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        x_compass = []
        for x in adata.obs.index:
            x_compass.append(mapped_components[x])
        adata.obsm['X_compass'] = numpy.array(x_compass)
        sc.pp.neighbors(adata, use_rep="X_compass", n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist)
        self.adata = adata
        return adata

    def clique_family_distances(self):
        from scipy.spatial import distance
        adata = self.adata
        mapped_components = dict(zip(list(cembed.data.keys()),cembed.matrix))
        gene_cluster_vectors = dict()
        for gclust, genes in cluster_definitions.items():
            vector = embed.generate_vector(genes)
            gene_cluster_vectors[gclust] = vector
        for gclust, vector in gene_cluster_vectors.items():
            print("Computed CF: ",gclust)
            dists = []
            for x in adata.obs.index:
                dist = 1.0 - distance.cosine(mapped_components[x],vector)
                dists.append(dist)
            adata.obs[gclust] = dists
        self.adata = adata
        return adata

    def plot_clique(self,clique_genes,distance=0.5,threshold=-1.0, title=None):
        import matplotlib as mpl
        G = self.embed.generate_network(threshold=distance)
        fig = plt.figure(figsize=(14,6))
        ax = plt.subplot(1,2,1)
        subG = G.subgraph(clique_genes)

        centrality = dict(nx.betweenness_centrality(subG))
        low, *_, high = sorted(centrality.values())
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
        pos = nx.nx_agraph.graphviz_layout(subG, prog="neato",args="-Goverlap=scale -Elen=5 -Eweight=0.2")

        nx.draw(subG,pos,with_labels=True,
                node_color=[mapper.to_rgba(i) for i in centrality.values()],
                node_size=100,ax=ax,font_size=16,
                edge_color=[[0.5,0.5,0.5,0.5] for _ in subG.edges()])
        clique_genes = set(clique_genes).difference(set(list(nx.isolates(subG))))
        vector = self.embed.generate_vector(list(clique_genes))
        ax = plt.subplot(1,2,2)
        pcs = self.pcs["UMAP"]
        distances = []
        dataset_distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(self.dataset_vector).reshape(1, -1))[0])
        for cell_vector in self.matrix:
            distance = float(cosine_similarity(numpy.array(cell_vector).reshape(1, -1),numpy.array(vector).reshape(1, -1))[0])
            d = distance-dataset_distance
            if d < threshold:
                d = threshold
            distances.append(d)
        data = {"x":pcs[0],"y":pcs[1],"Distance": distances}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Distance', ax=ax,linewidth=0.00,s=7,alpha=0.7)
        if title != None:
            plt.title(title)
        plt.show()
        return clique_genes


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
