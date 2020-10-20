from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
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

import data

class GeneEmbedding(object):

    def __init__(self, embedding_file, context):
        self.vector = []
        self.context = context
        self.context = context
        self.embedding_file = embedding_file
        self.embeddings = self.read_embedding(self.embedding_file)
        self.vector = []
        for gene in tqdm.tqdm(self.context.expressed_genes):
            if gene in self.embeddings:
                self.vector.append(self.embeddings[gene])

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
        return sorted_distances

    def cluster(self):
        kmeans = AgglomerativeClustering(n_clusters=36, affinity="cosine", linkage="average")
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
            average_vector[cluster] = xvec
            #average_vector[cluster] = numpy.subtract(xvec,self.total_average_vector)
        return average_vector, gene_to_cluster

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
            print(cluster, similarities[cluster][:15])
        return similarities
    
    def plot(self, clusters, png=None, method="TSNE", labels=[]):
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1,1,1)
        self.plot_reduction(clusters, ax, labels=labels)
        if png:
            plt.savefig(png)
            plt.close()
        else:
            plt.show()

    def plot_reduction(self, clusters, ax, method="TSNE", labels=[]):
        if method == "TSNE":
            pca = TSNE(n_components=2, metric="cosine")
            pcs = pca.fit_transform(self.vector)
        elif method == "UMAP":
            trans = umap.UMAP(random_state=42).fit(self.matrix)
            x = trans.embedding_[:, 0]
            y = trans.embedding_[:, 1]
            pcs = [x,y]
        else:
            raise ValueError("Method '{}' not supported!".format(method))
        pcs = numpy.transpose(pcs)
        data = {"x":pcs[0],"y":pcs[1], "Cluster":clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y",hue="Cluster", ax=ax)
        if len(labels):
            for x, y, gene in zip(pcs[0], pcs[1], self.context.expressed_genes):
                if gene in labels:
                    ax.text(x+.02, y, str(gene), fontsize=8)

class CellEmbedding(object):

    def __init__(self, context, embed, adata, resolution=10):
        
        cell_to_gene = list(context.cell_to_gene.items())

        self.context = context
        self.embed = embed

        self.celltypes = dict(zip(context.cells, context.metadata["cell_type"]))
        self.samples = dict(zip(context.cells, context.metadata["sample"]))
        
        self.expression = self.expression(adata)
        # self.expression = pickle.load(open(""))

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
        self.celltype = []
        self.sample = []
        
        self.celltype_vector = collections.defaultdict(list)
        self.sample_vector = collections.defaultdict(list)

        dataset_vector = []
        for cell, vectors in self.data.items():
            weights = self.weights[cell]
            xvec = list(numpy.average(vectors, axis=0, weights=weights))
            self.matrix.append(xvec)
            self.sample.append(self.samples[cell])
            self.celltype.append(self.celltypes[cell])
            dataset_vector += vectors
            self.celltype_vector[self.celltypes[cell]] += vectors
            self.sample_vector[self.samples[cell]].append(xvec)


        self.clusters = self.cluster(k=resolution)
        clusters = self.clusters
        local_correction = collections.defaultdict(lambda : collections.defaultdict(list))
        correction_vectors = collections.defaultdict(dict)
        for cluster, batch, vec in zip(clusters, self.sample, self.matrix):
            local_correction[cluster][batch].append(vec)
        for cluster, batches in local_correction.items():
            cluster_vec = []
            weights = []
            batch_keys = list(batches.keys())
            base_batch = batch_keys.pop(0)
            max_distance = 1.0
            cluster_vec = numpy.average(batches[base_batch], axis=0)
            for batch in batch_keys:
                bvec = list(numpy.average(batches[batch], axis=0))
                distance = float(cosine_similarity(numpy.array(bvec).reshape(1, -1),numpy.array(cluster_vec).reshape(1, -1))[0])
                print(distance)
                if max_distance > distance:
                    max_distance = distance
                offset = numpy.subtract(cluster_vec,bvec)
                bvec = numpy.add(bvec,offset)
                distance = float(cosine_similarity(numpy.array(bvec).reshape(1, -1),numpy.array(cluster_vec).reshape(1, -1))[0])
                print(distance)
                _offset = []
                correction_vectors[cluster][batch] = offset


        self.matrix = []
        self.celltype_vector = collections.defaultdict(list)
        self.sample_vector = collections.defaultdict(list)
        i = 0
        for cell, vectors in self.data.items():
            cluster = clusters[i]
            weights = self.weights[cell]
            xvec = list(numpy.average(vectors, axis=0))
            batch = self.samples[cell]
            # if cluster in correction_vectors and batch in correction_vectors[cluster]:
            #     offset = correction_vectors[cluster][batch]
            #     xvec = numpy.add(xvec,offset)
            self.matrix.append(xvec)
            self.celltype_vector[self.celltypes[cell]] += vectors
            self.sample_vector[self.samples[cell]].append(xvec)
            i += 1

        self.dataset_vector = numpy.average(dataset_vector, axis=0)
        _matrix = []
        for vec in self.matrix:
            _matrix.append(numpy.subtract(vec, self.dataset_vector))
        self.matrix = _matrix

    def expression(self, adata):
        data = collections.defaultdict(dict)
        genes = [x.upper() for x in adata.var.index]
        normalized_matrix = adata.X
        nonzero = (normalized_matrix > 0).nonzero()
        cells = adata.obs.index
        print("Loading Expression.")
        nonzero_coords = list(zip(nonzero[0],nonzero[1]))
        normalized_matrix = adata.X.todok()
        for cell, gene in tqdm.tqdm(nonzero_coords):
            symbol = genes[gene]
            barcode = cells[cell]
            data[barcode][symbol] = normalized_matrix[cell,gene]
        return data

    def cluster(self, k=12):
        kmeans = AgglomerativeClustering(n_clusters=k)
        kmeans.fit(self.matrix)
        clusters = kmeans.labels_
        _clusters = []
        for cluster in clusters:
            _clusters.append("C"+str(cluster))
        return _clusters

    def compute_gene_similarities(self):
        gene_similarities = dict()
        vectors = collections.defaultdict(list)
        for vec, label in zip(self.matrix, self.clusters):
            vectors[label].append(vec)
        for label, vecs in vectors.items():
            distances = dict()
            cell_vector = list(numpy.median(vecs, axis=0))
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

    def plot_tsne(self, ax, pcs=None, clusters=None, labels=None):
        if type(pcs) != numpy.ndarray:
            pca = TSNE(n_components=2, n_jobs=-1, metric="cosine")
            pcs = pca.fit_transform(self.matrix)
            pcs = numpy.transpose(pcs)
            print("TSNE DONE")
            # trans = umap.UMAP(random_state=42).fit(self.matrix)
            # x = trans.embedding_[:, 0]
            # y = trans.embedding_[:, 1]
            # pcs = [x,y]
        data = {"x":pcs[0],"y":pcs[1],"Cluster": clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Cluster', ax=ax,linewidth=0.00,s=7,alpha=0.7)
        output = open('cells.tsv',"w")
        output.write("barcode,cluster\n")
        for barcode, cluster in zip(self.context.cells,clusters):
            output.write("{},{}\n".format(barcode, cluster))
        output.close()
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
        ax.set_title(title,fontsize=7)
        return pcs



def main():
    output_path = "./"
    sample = "SPECTRUM-OV-002"
    k = 10
    adata = sc.read("spectrum002.h5ad")
    context = data.Context.build(adata)

    
    # adata = adata.raw.to_adata()
    embed = GeneEmbedding("out.vec", context)
    print(embed.compute_similarities("CD8A")[:20])
    print("Clustering.")
    clusters = embed.cluster()

    print("Generating Representative Genes.")
    embed.plot(clusters, png="genes.png", labels=["CD8A","CD8B","GNLY","IFNG","CD3E","CD3D","CD2","CD7"])
    gene_clusters = embed.cluster_definitions(clusters)


    print("Building Cell Embedding...")
    cembed = CellEmbedding(context, embed, adata, resolution=k)

    plt.figure(figsize = (12, 6))
    ax1 = plt.subplot(1,2,1)
    pcs = cembed.plot_tsne(ax1, clusters=cembed.celltype)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax3 = plt.subplot(1,2,2)
    cembed.plot_tsne(ax3, clusters=cembed.sample,pcs=pcs)

    ax1.legend(fontsize='8')
    ax3.legend(fontsize='8')
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.savefig(os.path.join(output_path,"qcelltypes_{}.png".format(sample,k)))
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.close()

    # gclusters = list(gene_clusters.keys())
    
 
    exit(0)

    # cembed.compute_gene_similarities()

    # plt.figure(figsize = (40, 40))
    # ctypes = cembed.compute_cell_similarities(cembed.tissues)
    # labels = list(ctypes.keys())
    # matrix = []
    # for cell1 in labels:
    #     row = []
    #     for cell2 in labels:
    #         row.append(ctypes[cell1][cell2])
    #     matrix.append(row)
    # matrix = numpy.array(matrix)
    # df = pandas.DataFrame(matrix,index=labels,columns=labels)
    # sns.clustermap(df,figsize=(17,8))
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path,"tissue_similarities_{}.png".format(sample)))

if __name__ == '__main__':
    main()
