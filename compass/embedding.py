from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import matplotlib.cm as cm
import umap
import tqdm
import scanpy as sc

import numpy
import operator
import random
import pickle
import collections
import sys
import os

class GeneEmbedding(object):

    def __init__(self, embedding_file, context):
        self.vector = []
        self.context = context
        self.context = context
        self.embedding_file = embedding_file
        self.embeddings = self.read_embedding(self.embedding_file)
        self.vector = []
        self.genes = []
        for gene in tqdm.tqdm(self.context.expressed_genes):
            if gene in self.embeddings:
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

    def compute_similarities(self, gene):
        if gene not in self.embeddings:
            return None
        embedding = self.embeddings[gene]
        distances = dict()
        for target in self.embeddings.keys():
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

    def cluster(self, n=12):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(self.vector)
        clusters = kmeans.labels_
        clusters = zip(self.context.expressed_genes, clusters)
        _clusters = []
        for gene, cluster in clusters:
            _clusters.append("G"+str(cluster))
        return _clusters

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
        return list(numpy.median(vector, axis=0))

    def cluster_definitions(self, clusters):
        average_vector, gene_to_cluster = self.clusters(clusters)
        similarities = collections.defaultdict(dict)
        for cluster, vector in average_vector.items():
            distances = dict()
            for target in gene_to_cluster[cluster]:
                v = self.embeddings[target]
                distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
                distances[target] = distance
            sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
            similarities[cluster] = [x[0] for x in sorted_distances if x[0]]
        return similarities

    def cluster_definitions_as_df(self, similarities, top_n=20):
        clusters = []
        symbols = []
        for key, genes in similarities.items():
            clusters.append(key)
            symbols.append(", ".join(genes[:top_n]))
        df = pandas.DataFrame.from_dict({"Cluster Name":clusters, "Top Genes":symbols})
        return df

    def plot(self, clusters, png=None, method="TSNE", labels=[], pcs=None):
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1,1,1)
        pcs = self.plot_reduction(clusters, ax, labels=labels, method=method, pcs=pcs)
        if png:
            plt.savefig(png)
            plt.close()
        else:
            plt.show()
        return pcs

    def plot_reduction(self, clusters, ax, method="TSNE", labels=[], pcs=None):
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
        data = {"x":pcs[0],"y":pcs[1], "Cluster":clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y",hue="Cluster", ax=ax)
        if len(labels):
            for x, y, gene in zip(pcs[0], pcs[1], self.context.expressed_genes):
                if gene in labels:
                    ax.text(x+.02, y, str(gene), fontsize=8)
        return pcs

    def subtract_vector(self, vector):
        for gene, vec in self.embeddings.items():
            vec = numpy.subtract(vec-vector)
            self.embeddings[gene] = vec

    @staticmethod
    def relabel_cluster(similarities, clusters, old_label, new_label):
        genes = similarities[old_label]
        del similarities[old_label]
        similarities[new_label] = genes
        _clusters = []
        for cluster in clusters:
            if cluster == old_label:
                _clusters.append(new_label)
            else:
                _clusters.append(cluster)
        return similarities, _clusters

class CellEmbedding(object):

    def __init__(self, context, embed):

        cell_to_gene = list(context.cell_to_gene.items())
        self.context = context
        self.embed = embed
        self.expression = context.expression
        self.data = collections.defaultdict(list)
        self.weights = collections.defaultdict(list)

        for cell, genes in tqdm.tqdm(cell_to_gene):
            if len(genes) < 2: continue
            if cell in self.expression:
                cell_weights = self.expression[cell]
                for gene in set(genes).intersection(set(embed.embeddings.keys())):
                    if gene in cell_weights:
                        weight = self.expression[cell][gene]
                        if weight > 0:
                            self.data[cell].append(embed.embeddings[gene])
                            self.weights[cell].append(weight)
        self.matrix = []
        dataset_vector = []
        for cell, vectors in self.data.items():
            weights = self.weights[cell]
            xvec = list(numpy.average(vectors, axis=0, weights=weights))
            self.matrix.append(xvec)
            dataset_vector += vectors

        self.dataset_vector = numpy.average(dataset_vector, axis=0)
        _matrix = []
        for vec in self.matrix:
            _matrix.append(numpy.subtract(vec, self.dataset_vector))
        self.matrix = _matrix

    def batch_correct(self, column=None, clusters=None):
        if not column or not clusters:
            raise ValueError("Must supply batch column and clusters!")
        column_labels = dict(zip(self.context.cells,self.context.metadata[column]))
        labels = []
        for key in self.data.keys():
            labels.append(column_labels[key])
        local_correction = collections.defaultdict(lambda : collections.defaultdict(list))
        correction_vectors = collections.defaultdict(dict)
        for cluster, batch, vec in zip(clusters, labels, self.matrix):
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
                if max_distance > distance:
                    max_distance = distance
                offset = numpy.subtract(cluster_vec,bvec)
                bvec = numpy.add(bvec,offset)
                distance = float(cosine_similarity(numpy.array(bvec).reshape(1, -1),numpy.array(cluster_vec).reshape(1, -1))[0])
                correction_vectors[cluster][batch] = offset

        self.matrix = []
        self.sample_vector = collections.defaultdict(list)
        i = 0
        self.cell_order = []
        for cell, vectors in self.data.items():
            cluster = clusters[i]
            xvec = list(numpy.average(vectors, axis=0))
            batch = column_labels[cell]
            if cluster in correction_vectors and batch in correction_vectors[cluster]:
                offset = correction_vectors[cluster][batch]
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

    def subtract_vector(self, vector):
        corrected_matrix = []
        for cell_vector in self.matrix:
            corrected_matrix.append(numpy.subtract(cell_vector, vector))
        self.matrix = corrected_matrix

    def compute_gene_similarities(self):
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

    def group_cell_vectors(self, barcode_to_label):
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

    def compute_cell_similarities(self, barcode_to_label):
        vectors = dict()
        cell_similarities = dict()
        vectors, labels = self.group_cell_vectors(barcode_to_label)
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
                print("Running t-SNE")
                pca = TSNE(n_components=2, n_jobs=-1, metric="cosine")
                pcs = pca.fit_transform(self.matrix)
                pcs = numpy.transpose(pcs)
                print("Finished.")
            else:
                print("Running UMAP")
                trans = umap.UMAP(random_state=42,metric='cosine').fit(self.matrix)
                x = trans.embedding_[:, 0]
                y = trans.embedding_[:, 1]
                pcs = [x,y]
                print("Finished.")
        data = {"x":pcs[0],"y":pcs[1],"Cluster": clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Cluster', ax=ax,linewidth=0.00,s=7,alpha=0.7)
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

    def plot_distance(self, vector, pcs=None):
        plt.figure(figsize = (8,8))
        ax = plt.subplot(1,1, 1)
        if type(pcs) != numpy.ndarray:
            pca = TSNE(n_components=2)
            pcs = pca.fit_transform(self.matrix)
            pcs = numpy.transpose(pcs)
        distances = []
        dataset_distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(self.dataset_vector).reshape(1, -1))[0])
        for cell_vector in self.matrix:
            distance = float(cosine_similarity(numpy.array(cell_vector).reshape(1, -1),numpy.array(vector).reshape(1, -1))[0])
            distances.append(distance-dataset_distance)
        data = {"x":pcs[0],"y":pcs[1],"Distance": distances}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Distance', ax=ax,linewidth=0.00,s=7,alpha=0.7,legend=False)
        return pcs

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
        sns.scatterplot(data=df,x="x", y="y", hue='Gene Expression', ax=ax,linewidth=0.00,s=7,alpha=0.7,legend=False)
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
