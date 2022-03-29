import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import scanpy as sc
import numpy
from scipy.sparse import csr_matrix, find
import operator
import itertools
from itertools import permutations
import argparse
import tqdm
import random
import pickle
import collections
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import itertools
from sklearn.linear_model import LinearRegression
import gc
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
from sklearn import feature_extraction
from multiprocess import Pool
import copy
import pandas
import numpy
import numpy as np
import tqdm

class Context(object):

    def __init__(self):
        pass

    @classmethod
    def build(context_class, adata, subsample=None, expression=None, frequency_lower_bound = 10, threads=2):
        try:
            adata.var.index = [x.decode("utf-8") for x in adata.var.index]
        except Exception as e:
            pass
        context = context_class()
        if subsample:
            sc.pp.subsample(adata,fraction=subsample)
        context.adata = adata
        context.threads = threads
        context.genes = [x.upper() for x in list(context.adata.var.index)]
        context.normalized_matrix = context.adata.X
        context.metadata = context.adata.obs
        context.frequency_lower_bound = frequency_lower_bound
        try:
            for column in context.metadata.columns:
                if type(context.metadata[column][0]) == bytes:
                    context.metadata[column] = [x.decode("utf-8") for x in context.metadata[column]]
        except Exception as e:
            pass
        context.cells = context.adata.obs.index
        context.cell_index, context.index_cell = Context.index_cells(context.cells)
        context.data, context.cell_to_gene = context.expression(context.normalized_matrix, \
                            context.genes, \
                            context.index_cell,
                            expression=expression)
        context.expressed_genes = context.get_expressed_genes(context.data)
        context.gene_index, context.index_gene = Context.index_geneset(context.expressed_genes)
        context.gene2id = context.gene_index
        context.id2gene = context.index_gene
        context.gene_count = len(context.gene_frequency.keys())
        context.adata = adata
        return context

    @classmethod
    def load(context_class, path):
        context = context_class()
        serialized = pickle.load(open(path, "rb"))
        context.unserialize(serialized)
        context.path = os.path.split(path)[0]
        return context

    @staticmethod
    def index_geneset(genes):
        gene_index = {w: idx for (idx, w) in enumerate(genes)}
        index_gene = {idx: w for (idx, w) in enumerate(genes)}
        return gene_index, index_gene

    @staticmethod
    def index_cells(cells):
        cell_index = {w: idx for (idx, w) in enumerate(cells)}
        index_cell = {idx: w for (idx, w) in enumerate(cells)}
        return cell_index, index_cell

    def get_expressed_genes(self, data):
        return list(data.keys())

    def get_expressed_genes_frequency(self, data):
        return self.gene_frequency

    def inverse_filter(self, data):
        cell_to_gene = collections.defaultdict(list)
        for gene, cells in data.items():
            for cell in cells:
                cell_to_gene[cell].append(gene)
        return cell_to_gene

    def expression(self, normalized_matrix, genes, cells, expression=None):
        gene_index, index_gene = Context.index_geneset(genes)
        self.gene_frequency = collections.defaultdict(int)
        data = collections.defaultdict(list)
        if expression == None:
            self.expression = collections.defaultdict(dict)
            nonzero = find(normalized_matrix > 0)
            print("Loading Expression.")

            nonindexed_expression = collections.defaultdict(dict)
            for cell, gene_i, val in tqdm.tqdm(list(zip(*nonzero))):
                symbol = index_gene[gene_i]
                nonindexed_expression[cell][symbol] = normalized_matrix[cell,gene_i]

            print("Reindexing Cooc")
            for cell, genes in tqdm.tqdm(list(nonindexed_expression.items())):
                barcode = cells[cell]
                for index, val in genes.items():
                    self.expression[barcode][index] = val
                    data[index].append(barcode)
                    self.gene_frequency[index] += 1
        else:
            self.expression = pickle.load(open(expression,"rb"))
            for cell, genes in tqdm.tqdm(list(self.expression.items())):
                for gene, val in genes.items():
                    data[gene].append(cell)
                    self.gene_frequency[gene] += 1

        data = self.filter_on_frequency(data)
        return data, self.inverse_filter(data)

    def filter_on_frequency(self, data):
        remove = []
        for gene, frequency in self.gene_frequency.items():
            if frequency < self.frequency_lower_bound:
                del data[gene]
                remove.append(gene)
        for gene in remove:
            del self.gene_frequency[gene]
        return data

    def serialize(self):
        serialized = dict()
        for attr, value in self.__dict__.items():
            if attr != "adata" and attr != "inv_data" and attr != "data":
                serialized[attr] = value
        return serialized

    def unserialize(self, serialized):
        for attribute, value in serialized.items():
            setattr(self, attribute, value)

    def save(self, filename):
        serialized = self.serialize()
        pickle.dump(serialized, open(filename,"wb"))

    def frequency_histogram(self):
        f = np.array(list(self.gene_frequency.values())) / len(self.cells)
        plt.hist(f, 200, density=True, facecolor='g', alpha=0.75)
        plt.grid(True)
        plt.show()

    def frequency(self, gene):
        return self.gene_frequency[gene] / len(self.cells)

def calculate_mi_parallel(payload):
    mi_scores = dict()
    jdf = payload[0]
    gene = payload[1]
    genes = payload[2]
    gene_profile = jdf.loc[gene]
    if list(gene_profile > 0).count(True) < 3:
        return mi_scores
    for other in genes:
        e = numpy.array([gene_profile,jdf.loc[other]]).T
        e = e[e.min(axis=1) > 0].astype(int)
        if len(e[:,0]) != 0 or e.shape[0] > 3:
            mi_scores[other] = calculate_mi(e, gene, other)
    print(gene)
    return mi_scores

def calculate_mi(e, gene1, gene2, bins=50, x_max=3, alpha=0.25):
    xbins = []
    ybins = []
    for i in numpy.linspace(0,1,bins)[1:]:
        x1 = int(np.quantile(e[:,0], i))
        if x1 not in xbins:
            xbins.append(x1)
        y1 = int(np.quantile(e[:,1], i))
        if y1 not in ybins:
            ybins.append(y1)
    if len(xbins) < 3 or len(ybins) < 3:
        return 0.0
    try:
        hgram, xedges, yedges = numpy.histogram2d(e[:,0],e[:,1],bins=(xbins,ybins))
    except Exception as e:
        return 0.0
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    expected_pmi = np.mean(np.log(pxy[nzs] / px_py[nzs]))
    ppmi = max([expected_pmi,0])
    return ppmi

class CompassDataset(Dataset):

    def __init__(self, adata, device="cpu"):
        self.data = Context.build(adata)
        self._word2id = self.data.gene2id
        self._id2word = self.data.id2gene
        self._vocab_len = len(self._word2id)
        self.device = device

    def generate_mi_scores_parallel(self,processes=10):
        df = pandas.DataFrame.from_dict(self.data.expression)
        df = df.fillna(0)
        self.jdf = df
        genes = self.jdf.index.tolist()
        total_genes = genes
        mi_scores = collections.defaultdict(lambda : collections.defaultdict(float))
        num_genes = len(genes)
        payloads = []
        while len(genes) > 0:
            gene = genes.pop(0)
            payload = (self.jdf, gene, copy.deepcopy(genes))
            if len(payload) == 3:
                payloads.append(payload)
        with Pool(processes) as p:
            results = p.map(calculate_mi_parallel, payloads)
            for p, r in zip(payloads,results):
                for gene, res in r.items():
                    mi_scores[p[1]][gene] = res
                    mi_scores[gene][p[1]] = res
        self.mi_scores = mi_scores

    def create_inputs_outputs(self, use_mi=False, thresh=0.0, min_coexp=10):
        print("Generating Correlation matrix.")
        import pandas

        from sklearn import feature_extraction
        vectorizer = feature_extraction.DictVectorizer(sparse=True)
        corr_matrix = vectorizer.fit_transform(list(self.data.expression.values()))
        corr_matrix[corr_matrix != 0] = 1

        all_genes = vectorizer.feature_names_

        gene_index = {w: idx for (idx, w) in enumerate(all_genes)}
        index_gene = {idx: w for (idx, w) in enumerate(all_genes)}
        self.data.gene2id = gene_index
        self.data.id2gene = index_gene
        self.data.expressed_genes = all_genes

        corr_matrix = pandas.DataFrame(data=corr_matrix.todense(),columns=all_genes)
        corr_matrix = corr_matrix[all_genes]
        corr_df = corr_matrix

        print("Decomposing")
        coocc = numpy.array(corr_df.T.dot(corr_df))

        corr_matrix = numpy.transpose(corr_matrix.to_numpy())
        cov = numpy.corrcoef(corr_matrix)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        import collections
        self.correlation = collections.defaultdict(dict)

        for gene, row in tqdm.tqdm(zip(all_genes, cov)):
            for cgene, value in zip(all_genes, row):
                wi = self.data.gene2id[gene]
                ci = self.data.gene2id[cgene]
                self._i_idx.append(wi)
                self._j_idx.append(ci)
                self.correlation[gene][cgene] = value
                if use_mi:
                    value = self.mi_scores[gene][cgene]
                if thresh > thresh: #and coocc[wi,ci] > min_coexp: #self.mi_scores[gene][cgene]
                    self._xij.append(value * coocc[wi,ci])
                else:
                    self._xij.append(0.0)
        if self.device == "cuda":
            self._i_idx = torch.cuda.LongTensor(self._i_idx).cuda()
            self._j_idx = torch.cuda.LongTensor(self._j_idx).cuda()
            self._xij = torch.cuda.FloatTensor(self._xij).cuda()
        else:
            self._i_idx = torch.LongTensor(self._i_idx).to("cpu")
            self._j_idx = torch.LongTensor(self._j_idx).to("cpu")
            self._xij = torch.FloatTensor(self._xij).to("cpu")
        self.coocc = coocc
        self.cov = cov


    def get_batches(self, batch_size):
        if self.device == "cuda":
            rand_ids = torch.cuda.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        else:
            rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
