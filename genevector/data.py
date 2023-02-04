import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy
from scipy import sparse
import itertools
import pickle
import os
import pandas
from sklearn import feature_extraction
import pandas
import tqdm
import collections

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

class Context(object):

    def __init__(self):
        pass

    @classmethod
    def build(context_class, adata, threads=2):
        try:
            adata.var.index = [x.decode("utf-8") for x in adata.var.index]
        except Exception as e:
            pass
        context = context_class()
        context.adata = adata
        context.threads = threads
        context.genes = [x.upper() for x in list(context.adata.var.index)]
        context.normalized_matrix = context.adata.X
        context.metadata = context.adata.obs
        try:
            for column in context.metadata.columns:
                if type(context.metadata[column][0]) == bytes:
                    context.metadata[column] = [x.decode("utf-8") for x in context.metadata[column]]
        except Exception as e:
            pass
        context.cells = context.adata.obs.index
        context.cell_index, context.index_cell = Context.index_cells(context.cells)
        context.data = context.expression(context.normalized_matrix, \
                                          context.genes, \
                                          context.cells)
        context.gene_index, context.index_gene = Context.index_geneset(adata.var.index.tolist())
        context.gene2id = context.gene_index
        context.id2gene = context.index_gene
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

    def expression(self, normalized_matrix, genes, cells):
        cells = cells.to_numpy()
        index_gene = numpy.array(genes)
        data = collections.defaultdict(list)
        self.expression = collections.defaultdict(dict)
        print("Loading Expression.")
        normalized_matrix = normalized_matrix.tocsc()
        normalized_matrix.eliminate_zeros()
        print("Finding nonzero indices.")
        row_indices, column_indices = normalized_matrix.nonzero()
        print("Loading nonzero entries.")
        nonzero_values = normalized_matrix.data
        print("Indexing expression.")
        entries = list(zip(nonzero_values, row_indices, column_indices))
        for value, i, j in tqdm.tqdm(entries):
            barcode = cells[i]
            symbol = index_gene[j]
            self.expression[barcode][symbol] = value
            data[symbol].append(barcode)
        print("Finished.")
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



class GeneVectorDataset(Dataset):

    def __init__(self, adata, device="cpu", mi_scores=None):
        adata.X = sparse.csr_matrix(adata.X)
        self.data = Context.build(adata)
        self._word2id = self.data.gene2id
        self._id2word = self.data.id2gene
        self._vocab_len = len(self._word2id)
        self.device = device
        self.mi_scores = mi_scores

    def generate_mi_scores(self, bins = 40):
        from fast_histogram import histogram2d
        print(bcolors.OKGREEN + "Getting gene pairs combinations." + bcolors.ENDC)
        mi_scores = collections.defaultdict(lambda : collections.defaultdict(float))
        bcs = dict()
        vgenes = []
        for gene, bc in self.data.data.items():
            bcs[gene] = set(bc)
            vgenes.append(gene)
        pairs = list(itertools.combinations(vgenes, 2))
        counts = collections.defaultdict(lambda : collections.defaultdict(int))

        maxs = dict(zip([x.upper() for x in self.data.adata.var.index.tolist()],numpy.array(self.data.adata.X.max(axis=1).T.todense())[0]))
        for c, p in self.data.expression.items():
            for g,v in p.items():
                counts[g][c] += int(v)
        print(bcolors.OKGREEN + "Computing MI for each pair." + bcolors.ENDC)
        for p1,p2 in tqdm.tqdm(pairs):
            common = bcs[p1].intersection(bcs[p2])
            if len(common) ==0: continue
            maxp1 = maxs[p1]
            maxp2 = maxs[p2]
            if maxp1 < 2 or maxp2 < 2: continue
            c1 = counts[p1]
            c2 = counts[p2]
            x = [c1[bc] for bc in common]
            y = [c2[bc] for bc in common]
            rangex = [0,maxs[p1]]
            rangey = [0,maxs[p2]]
            pxy = histogram2d(x,y,bins=bins,range=[rangex,rangey])
            #pxy, _, _ = numpy.histogram2d(x,y, density=True)
            pxy = pxy / pxy.sum()
            px = np.sum(pxy, axis=1)
            px = px / px.sum()
            py = np.sum(pxy, axis=0)
            py = py / py.sum()
            px_py = px[:, None] * py[None, :]
            nzs = pxy > 0
            mi = np.sum(pxy[nzs] * numpy.log2((pxy[nzs] / px_py[nzs])))
            mi_scores[p1][p2] = mi
            mi_scores[p2][p1] = mi
        self.mi_scores = mi_scores


    def create_inputs_outputs(self, c=100., bins=30):
        print(bcolors.WARNING+"*****************"+bcolors.ENDC)
        print(bcolors.HEADER+"Loading Dataset."+bcolors.ENDC)
        print(bcolors.WARNING+"*****************\n"+bcolors.ENDC)
        if self.mi_scores == None:
            self.generate_mi_scores(bins=bins)
        gene_index = {w: idx for (idx, w) in enumerate(self.data.genes)}
        index_gene = {idx: w for (idx, w) in enumerate(self.data.genes)}
        self.data.gene2id = gene_index
        self.data.id2gene = index_gene

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        print(bcolors.OKGREEN + "Loading Batches for Training." + bcolors.ENDC)

        for gene in tqdm.tqdm(self.data.genes):
            for cgene in self.data.genes:
                if gene == cgene: continue
                wi = self.data.gene2id[gene]
                ci = self.data.gene2id[cgene]
                self._i_idx.append(wi)
                self._j_idx.append(ci)
                value = self.mi_scores[gene][cgene] * c**2
                if value > 0:
                    self._xij.append(value)
                else:
                    self._xij.append(0.)

        if self.device == "cuda":
            self._i_idx = torch.cuda.LongTensor(self._i_idx).cuda()
            self._j_idx = torch.cuda.LongTensor(self._j_idx).cuda()
            self._xij = torch.cuda.FloatTensor(self._xij).cuda()
        else:
            self._i_idx = torch.LongTensor(self._i_idx).to("cpu")
            self._j_idx = torch.LongTensor(self._j_idx).to("cpu")
            self._xij = torch.FloatTensor(self._xij).to("cpu")

        print(bcolors.OKCYAN + "Ready to train." + bcolors.ENDC)
    
    def get_batches(self, batch_size):
        if self.device == "cuda":
            rand_ids = torch.cuda.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        else:
            rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
