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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import softmax
from scipy.spatial import distance
import numpy
import tqdm
from scipy.sparse import csr_matrix


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import softmax
from scipy.spatial import distance
import numpy
import tqdm

import numpy
import operator
import random
import pickle
import collections
import sys
import os
import pandas as pd


import gc

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

    def read_embedding(self, filename):
        embedding = dict()
        lines = open(filename,"r").read().splitlines()[1:]
        for line in lines:
            vector = line.split()
            gene = vector.pop(0)
            embedding[gene] = [float(x) for x in vector]
        return embedding

    def get_adata(self, resolution=20):
        mat = numpy.array(self.vector)
        numpy.savetxt(".tmp.txt",mat)
        gdata = sc.read_text(".tmp.txt")
        os.remove(".tmp.txt")
        gdata.obs.index = self.genes
        sc.pp.neighbors(gdata,use_rep="X")
        sc.tl.leiden(gdata,resolution=resolution)
        sc.tl.umap(gdata)
        return gdata

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
        fig,ax = plt.subplots(1,1,figsize=(8,6))
        sc.pl.umap(gdata,alpha=0.5,show=False,size=100,ax=ax)
        sub = gdata[gdata.obs["Metagene {}".format(mg)]!="_Other"]
        sc.pl.umap(sub,color="Metagene {}".format(mg),title=title,size=200,show=False,add_outline=False,ax=ax)

        for gene, pos in zip(gdata.obs.index,gdata.obsm["X_umap"].tolist()):
            if gene in _labels:
                ax.text(pos[0]+.04, pos[1], str(gene), fontsize=6, alpha=0.9, fontweight="bold")
        plt.tight_layout()

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
        self.embed = embed
        self.expression = self.context.expression
        self.data = collections.defaultdict(list)
        self.weights = collections.defaultdict(list)
        self.pcs = dict()
        self.matrix = []

        adata = self.context.adata.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

        weights = collections.defaultdict(list)

        for cell in tqdm.tqdm(adata.obs.index.tolist()):
            if type(adata.X[adata.obs.index.tolist().index(cell)]) == csr_matrix:
                exp = adata.X[adata.obs.index.tolist().index(cell)].T.todense()
                exp = [float(x[0]) for x in exp]
            else:
                exp = adata.X[adata.obs.index.tolist().index(cell)]
            cell_weights = dict(zip(adata.var.index.tolist(),exp))
            weights = []
            vectors = []
            for g,w in cell_weights.items():
                if g in embed.embeddings:
                    weights.append(w)
                    vectors.append(embed.embeddings[g])
            weights = numpy.array(weights)
            self.matrix.append(numpy.average(vectors,axis=0,weights=weights))
            self.data[cell] = vectors
        self.dataset_vector = numpy.zeros(numpy.array(self.matrix).shape[1])

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
        self.matrix = corrected_matrix

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
        mean_vecs = []
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

    def phenotype_probability(self, adata, up_phenotype_markers, down_phenotype_markers, target_col="genevector"):
        mapped_components = dict(zip(list(self.data.keys()),self.matrix))
        adata = adata[list(self.data.keys())]
        probs = dict()
        for pheno, markers in up_phenotype_markers.items():
            dists = []
            vector = self.embed.generate_vector(markers)
            if pheno in down_phenotype_markers:
                dvector = self.embed.generate_vector(down_phenotype_markers[pheno])
                vector = numpy.subtract(vector, dvector)
            ovecs = []
            for oph, ovec in up_phenotype_markers.items():
                if oph != pheno:
                    ovec = self.embed.generate_vector(ovec)
                    ovecs.append(ovec)
            aovec = numpy.mean(ovecs,axis=0)
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
        scaler = StandardScaler()
        probabilities = softmax(scaler.fit_transform(numpy.array(distribution)),axis=1)
        for ct in probabilities:
            assign = celltypes[numpy.argmax(ct)]
            classif.append(assign)
        umap_pts = dict(zip(list(self.data.keys()),classif))
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
                print(ph)
                adata.obs[ph+" Pseudo-probability"] = probs[ph]
            return adata
        adata = load_predictions(adata,probs)
        return adata

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

    @staticmethod
    def plot_confusion_matrix(adata,label1,label2):
        import numpy as np
        from sklearn.metrics import confusion_matrix

        gv = adata.obs[label1].tolist()
        gt = adata.obs[label2].tolist()
        def plot_cm(y_true, y_pred, figsize=(10,10)):
            cm = confusion_matrix(y_true, y_pred, labels=numpy.unique(y_true))
            cm_sum = numpy.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            annot = numpy.empty_like(cm).astype(str)
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
            cm = pandas.DataFrame(cm, index=numpy.unique(y_true), columns=numpy.unique(y_true))
            cm.index.name = 'Gene Vector'
            cm.columns.name = 'Ground Truth'
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        plot_cm(gv,gt)

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
