import numpy as np
import torch
from torch.utils.data import Dataset
import numpy
from scipy import sparse
import itertools
import pickle
import os
from multiprocessing import Pool
import tqdm
import collections
from scipy.stats import entropy
from sklearn import feature_extraction
import pandas

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
        print(bcolors.OKGREEN + "Loading Expression." + bcolors.ENDC)
        normalized_matrix.eliminate_zeros()
        row_indices, column_indices = normalized_matrix.nonzero()
        nonzero_values = normalized_matrix.data
        print(bcolors.BOLD+"Indexing expression."+bcolors.ENDC)
        entries = list(zip(nonzero_values, row_indices, column_indices))
        for value, i, j in tqdm.tqdm(entries):
            barcode = cells[i]
            symbol = index_gene[j]
            self.expression[barcode][symbol] = value
            data[symbol].append(barcode)
        print(bcolors.OKGREEN+"Finished."+bcolors.ENDC)
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

    """
    This class provides extends the torch Dataset class with functionality to compute mutual information between genes and generate batches of input and output data for each gene pair for training..

    :param adata: The AnnData Scanpy object that holds the dataset with expression data in .X.
    :type adata: AnnData
    :param device: The device to load torch dataset ("cpu","cuda","mips" for torch metal acceleration).
    :type device: str
    :param mi_scores: Optionallu side load a dictionary of two levels containing the training target for each gene pair.
    :type mi_scores: dict of dict
    :param processes: Not functional, adding support for multiprocessing MI computation.
    :type processes: int
    """

    """
    This class provides extends the torch Dataset class with functionality to compute mutual information between genes and generate batches of input and output data for each gene pair for training..

    :param adata: The AnnData Scanpy object that holds the dataset with expression data in .X.
    :type adata: AnnData
    :param device: The device to load torch dataset ("cpu","cuda","mips" for torch metal acceleration).
    :type device: str
    :param mi_scores: Optionallu side load a dictionary of two levels containing the training target for each gene pair.
    :type mi_scores: dict of dict
    :param processes: Not functional, adding support for multiprocessing MI computation.
    :type processes: int
    """

    def __init__(self, adata, device="cpu", mi_scores=None, processes=1, apply_qc=True, entropy_threshold=1.):
        adata.var.index = [str(x).upper() for x in adata.var.index.tolist()]
        adata.X = sparse.csr_matrix(adata.X)
        if apply_qc:
            adata = self.quality_control(adata,entropy_threshold=entropy_threshold)
        self.adata = adata
        self.data = Context.build(adata)
        self._word2id = self.data.gene2id
        self._id2word = self.data.id2gene
        self._vocab_len = len(self._word2id)
        self.device = device
        self.mi_scores = mi_scores
        self.processes = processes

    @staticmethod
    def get_gene_entropy(adata):
        X = adata.X.todense()
        X = numpy.array(X.T)
        gene_to_row = list(zip(adata.var.index.tolist(), X))
        gene_entropy = dict()
        for g, exp in tqdm.tqdm(gene_to_row):
            counts = np.unique(exp, return_counts = True)
            gene_entropy[g] = entropy(counts[1][1:])
        return gene_entropy

    @staticmethod
    def quality_control(adata, entropy_threshold = 1.):
        adata.var_names_make_unique()
        print(bcolors.BOLD + "Removing Genes..."+ bcolors.ENDC)
        gene_entropy = GeneVectorDataset.get_gene_entropy(adata)
        vgenes = [x for x,y in gene_entropy.items() if y > entropy_threshold]
        adata = adata[:,vgenes]
        print(bcolors.OKGREEN + "Selecting {} Genes with greater than {} nats entropy.".format(len(vgenes), entropy_threshold)+ bcolors.ENDC)
        return adata.copy()

    # def generate_mi_scores(self):
    #     mi_scores = collections.defaultdict(lambda : collections.defaultdict(float))
    #     bcs = dict()
    #     maxs = dict(zip([x.upper() for x in self.data.adata.var.index.tolist()],numpy.array(self.data.adata.X.max(axis=1).T.todense())[0]))
    #     vgenes = []
    #     for gene, bc in self.data.data.items():
    #         bcs[gene] = set(bc)
    #         if maxs[gene.upper()] > 1:
    #             vgenes.append(gene.upper())

    #     exp = dict()
    #     bins = dict()
    #     indices = dict()
    #     adata = self.data.adata
    #     print(bcolors.OKGREEN + "Computing Expression Bins." + bcolors.ENDC)
    #     for gene in tqdm.tqdm(vgenes):
    #         exp[gene] = numpy.array(adata.X[:,adata.var.index.tolist().index(gene)].todense().T.tolist()[0])
    #         bins[gene] = self.rna_expr_percentile_hist(exp[gene])
    #         indices[gene] = self.rna_expr_to_bin_inds(exp[gene],bins[gene])
        
    #     pairs = list(itertools.combinations(vgenes, 2))
    #     self.num_pairs = len(pairs)

    #     joints = []
    #     for p1,p2 in tqdm.tqdm(pairs):
    #         nbins1 = bins[p1].shape[0]+1
    #         nbins2 = bins[p2].shape[0]+1
    #         joints.append((indices[p1],indices[p2],nbins1,nbins2))

    #     print(bcolors.OKGREEN + "Computing Joint Distributions and Mutual Information." + bcolors.ENDC)
    #     import time
    #     start_time = time.time()
    #     with Pool(processes=self.processes) as pool:
    #         result = pool.starmap(self.mutual_info, joints)
    #         for ps, mi in zip(pairs, result):
    #             mi_scores[ps[0]][ps[1]] = mi
    #             mi_scores[ps[1]][ps[0]] = mi
    #     print("Finished in %s seconds." % (time.time() - start_time))
    #     self.mi_scores = mi_scores

    def load_targets(self, targets):
        self.mi_scores = targets

    def generate_mi_scores(self):
        print(bcolors.OKGREEN + "Getting gene pairs combinations." + bcolors.ENDC)
        if self.mi_scores == None:
            mi_scores = collections.defaultdict(lambda : collections.defaultdict(float))
        else:
            mi_scores = self.mi_scores
        bcs = dict()
        maxs = dict(zip([x.upper() for x in self.data.adata.var.index.tolist()],numpy.array(self.data.adata.X.max(axis=1).T.todense())[0]))
        vgenes = []
        for gene, bc in self.data.data.items():
            bcs[gene] = set(bc)
            if maxs[gene.upper()] > 1:
                vgenes.append(gene)
        ipairs = list(itertools.combinations(vgenes, 2))
        pairs = []
        for p1,p2 in ipairs:
            if p1 in mi_scores and p2 in mi_scores[p1]:
                continue
            if p2 in mi_scores and p1 in mi_scores[p2]:
                continue
            pairs.append((p1,p2))
        counts = collections.defaultdict(lambda : collections.defaultdict(int))
        self.num_pairs = len(pairs)
        
        for c, p in self.data.expression.items():
            for g,v in p.items():
                counts[g][c] += int(v)
        print(bcolors.OKGREEN + "Computing MI for each pair." + bcolors.ENDC)
        for p1,p2 in tqdm.tqdm(pairs):
            common = bcs[p1].intersection(bcs[p2])
            if len(common) ==0: continue
            
            c1 = counts[p1]
            c2 = counts[p2]
            x = [c1[bc] for bc in common]
            y = [c2[bc] for bc in common]
            
            pxy, _, _ = numpy.histogram2d(x,y, density=True)
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


    @staticmethod
    def mutual_info():
        pxy, _, _ = numpy.histogram2d(x,y, density=True)
        pxy = pxy / pxy.sum()
        px = np.sum(pxy, axis=1)
        px = px / px.sum()
        py = np.sum(pxy, axis=0)
        py = py / py.sum()
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * numpy.log2((pxy[nzs] / px_py[nzs])))
        return mi

    @staticmethod
    def mutual_info(rna_ind_exprA, rna_ind_exprB, nbinsA, nbinsB):
        pxy = numpy.zeros((nbinsA,nbinsB))
        indxs = list(zip(rna_ind_exprA, rna_ind_exprB))
        for indA, indB in set(indxs):
            pxy[indA, indB] = indxs.count((indA,indB))
        pxy = pxy[1:,1:]
        pxy = pxy / pxy.sum()
        px = np.sum(pxy, axis=1)
        px = px / px.sum()
        py = np.sum(pxy, axis=0)
        py = py / py.sum()
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        pxy = pxy[nzs]
        px_py = px_py[nzs]
        mi = np.sum(pxy * np.log2((pxy / px_py)))
        return mi

    @staticmethod
    def rna_expr_percentile_hist(rna_expr, min_frac_coverage = .05):
        rna_expr = np.array(sorted(rna_expr))
        non_zero_ind_start = rna_expr.searchsorted(0, 'right')
        n_tot = len(rna_expr)
        min_coverage = int(np.ceil((n_tot - non_zero_ind_start)*min_frac_coverage))
        bins_out = [0]
        i = min_coverage
        while i < n_tot - min_coverage + 1:
            if rna_expr[i] > bins_out[-1]:
                bins_out.append(rna_expr[i])
                i += min_coverage
            else:
                i = rna_expr.searchsorted(rna_expr[i], 'right')
        return np.array(bins_out)

    @staticmethod
    def rna_expr_to_bin_inds(rna_expr, bins):
        return [0 if x == 0 else bins.searchsorted(x) for x in rna_expr]

    @staticmethod
    def rna_ind_vecs_to_joint_dist(rna_ind_exprA, rna_ind_exprB, nbinsA, nbinsB):
        joint_dist = numpy.zeros((nbinsA,nbinsB))
        for indA, indB in zip(rna_ind_exprA,rna_ind_exprB):
            joint_dist[indA, indB] += 1
        return joint_dist

    def create_inputs_outputs(self, c=1.):
        print(bcolors.WARNING+"*****************"+bcolors.ENDC)
        print(bcolors.HEADER+"Loading Dataset."+bcolors.ENDC)
        print(bcolors.WARNING+"*****************\n"+bcolors.ENDC)
        if self.mi_scores == None:
            self.generate_mi_scores()
        print(bcolors.FAIL+"MI Loaded."+bcolors.ENDC)


        gene_index = {w: idx for (idx, w) in enumerate(self.data.genes)}
        index_gene = {idx: w for (idx, w) in enumerate(self.data.genes)}
        self.data.gene2id = gene_index
        self.data.id2gene = index_gene

        print(bcolors.FAIL+"Finding coefficients."+bcolors.ENDC)
        
        # cov = numpy.corrcoef(self.adata.X.T.todense(),rowvar=True)

        matrix = numpy.array(self.adata.X.T.todense())
        correlation_matrix = numpy.corrcoef(matrix, rowvar=True)
        correlation_dict = {}
        names=self.adata.var.index.tolist()
        for i, row_name in enumerate(names):
            correlation_dict[row_name] = {}
            for j, col_name in enumerate(names):
                correlation_dict[row_name][col_name] = correlation_matrix[i, j]
        print(bcolors.FAIL+"Finished."+bcolors.ENDC)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        pairs = list(itertools.combinations(names,2))
        self.num_pairs = len(pairs)

        print(bcolors.OKGREEN + "Loading Batches for Training." + bcolors.ENDC)
    
        for genes in tqdm.tqdm(pairs):
            gene = genes[0]
            cgene = genes[1]
            wi = self.data.gene2id[gene]
            ci = self.data.gene2id[cgene]
            self._i_idx.append(wi)
            self._j_idx.append(ci)
            mivalue = self.mi_scores[gene][cgene] * c
            value = correlation_dict[gene][cgene]
            self._xij.append(mivalue * value)

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
