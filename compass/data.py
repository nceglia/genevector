import numpy as np
import torch
from torch.utils.data import Dataset

import scanpy as sc
import numpy
from scipy.sparse import csr_matrix
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


class Context(object):

    def __init__(self):
        pass

    @classmethod
    def build(context_class, adata, subsample=None):
        try:
            adata.var.index = [x.decode("utf-8") for x in adata.var.index]
        except Exception as e:
            pass
        context = context_class()
        if subsample:
            sc.pp.subsample(adata,fraction=subsample)
        context.adata = adata
        context.genes = [x.upper() for x in list(context.adata.var.index)]
        context.normalized_matrix = context.adata.X
        context.metadata = context.adata.obs
        context.frequency_lower_bound = 1
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
                            context.index_cell)
        context.expressed_genes = context.get_expressed_genes(context.data)
        context.gene_index, context.index_gene = Context.index_geneset(context.expressed_genes)
        context.negatives = []
        context.discards = []
        context.negpos = 0
        context.negative_table_size = 1e8
        context.gene2id = context.gene_index
        context.id2gene = context.index_gene
        context.gene_count = len(context.gene_frequency.keys())
        context.adata = adata
        context.init_negative_table()
        # context.idx_pairs = context.coexpression()
        
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
    
    def set_lower_bound_on_frequency(self, frequency):
        self.frequency_lower_bound = frequency

    @staticmethod
    def filter_gene(symbol):
        symbol = symbol.lower()
        if symbol.startswith("rp") \
           or symbol.startswith("mt-") \
           or "." in symbol \
           or "rik" in symbol.lower() \
           or "linc" in symbol.lower() \
           or "orf" in symbol.lower() \
           or "ercc" in symbol.lower() \
           or symbol.startswith("gm"):
            return True
        return False

    def expression(self, normalized_matrix, genes, cells):
        gene_index, index_gene = Context.index_geneset(genes)
        self.expression = collections.defaultdict(dict)
        nonzero = (normalized_matrix > 0).nonzero()
        print("Loading Expression.")
        nonzero_coords = list(zip(nonzero[0],nonzero[1]))
        self.gene_frequency = collections.defaultdict(int)
        data = collections.defaultdict(list)
        nonzero_cells = list(set(nonzero[0]))
        for cell in tqdm.tqdm(nonzero_cells):
            barcode = cells[cell]
            row = normalized_matrix.getrow(cell)
            for index in row.nonzero()[1]:
                symbol = index_gene[index]
                if not Context.filter_gene(symbol):
                    self.expression[barcode][symbol] = row[0,index]
                    data[symbol].append(barcode)
                    self.gene_frequency[symbol] += 1
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

    def init_negative_table(self):
        pow_frequency = np.array(list(self.gene_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * self.negative_table_size)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negative_targets(self, target, size):
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    # def coexpression(self, min_pair_num=10, max_pair_num=1000):
    #     idx_pairs = collections.defaultdict(int)
    #     for cell, genes in tqdm.tqdm(self.cell_to_gene.items()):
    #         pairs = list(permutations(genes,2))
    #         pairs = [(self.gene_index[pair[0]],self.gene_index[pair[1]]) for pair in pairs if pair[0] != pair[1]]
    #         for pair in pairs:
    #             if idx_pairs[pair] < max_pair_num:
    #                 idx_pairs[pair] += 1
    #     return idx_pairs

class CompassDataset(Dataset):

    def __init__(self, data, discard_probability=0.1, negative_targets=5):
        self.data = data
        self.negative_targets = negative_targets
        self.discard_probability = discard_probability

    def __len__(self):
        return len(self.data.cells)
    
    def update_discard_probability(self, probability):
        self.discard_probability = probability
    
    def update_negative_targets(self, negative_targets):
        self.negative_targets = negative_targets

    def __getitem__(self, idx):
        cell_id = self.data.index_cell[idx]
        genes = self.data.cell_to_gene[cell_id]
        word_ids = [self.data.gene2id[w] for w in genes if w in self.data.gene2id and np.random.rand() < self.discard_probability]
        idx_pairs = list(permutations(word_ids,2))
        return [(u, v, self.data.get_negative_targets(v, 5)) for u, v in idx_pairs if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
