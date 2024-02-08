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

    """
    This class provides an interface for parsing expression from AnnData objects.
    """

    def __init__(self):
        pass

    @classmethod
    def build(context_class, adata, threads=2, load_expression=True):
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
        print("Running...")
        context.cells = context.adata.obs.index
        context.cell_index, context.index_cell = Context.index_cells(context.cells)
        if load_expression:
            context.data = context.expression(context.normalized_matrix, \
                                            context.genes, \
                                            context.cells)
        else:
            print("Skipping expression load.")
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
    :param device: The device to load torch dataset ("cpu","cuda:0","mips" for torch metal acceleration).
    :type device: str
    :param mi_scores: Optionallu side load a dictionary of two levels containing the training target for each gene pair.
    :type mi_scores: dict of dict
    :param processes: Not functional, adding support for multiprocessing MI computation.
    :type processes: int
    """

    def __init__(self, adata, device="cpu", mi_scores=None, load_expression=True, signed_mi=True):
        """Constructor method
        """
        adata.var.index = [str(x).upper() for x in adata.var.index.tolist()]
        adata.X = sparse.csr_matrix(adata.X)
        self.adata = adata
        self.data = Context.build(adata, load_expression=load_expression)
        self._word2id = self.data.gene2id
        self._id2word = self.data.id2gene
        self._vocab_len = len(self._word2id)
        self.device = device
        self.mi_scores = mi_scores
        self.signed_mi = signed_mi

    @staticmethod
    def get_gene_entropy(adata):
        """
        Compute individual gene entropy.

        :param adata: The AnnData Scanpy object that holds the dataset with expression data in .X.
        :type adata: AnnData
        :return: Dictionary of gene to entropy.
        :rtype: dict
        """
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
        """
        Select genes with an entropy above the given threshold. Used in place of highly variable gene selection.

        :param adata: The AnnData Scanpy object that holds the dataset with expression data in .X.
        :type adata: AnnData
        :param entropy_threshold: Minimum entropy for a gene to be included in training and downstream analyses.
        :type entropy_threshold: float

        :return: Filtered AnnData object.
        :rtype: anndata.AnnData
        """
        adata.var_names_make_unique()
        print(bcolors.BOLD + "Removing Genes..."+ bcolors.ENDC)
        gene_entropy = GeneVectorDataset.get_gene_entropy(adata)
        vgenes = [x for x,y in gene_entropy.items() if y > entropy_threshold]
        adata = adata[:,vgenes]
        print(bcolors.OKGREEN + "Selecting {} Genes with greater than {} nats entropy.".format(len(vgenes), entropy_threshold)+ bcolors.ENDC)
        return adata.copy()

    def load_targets(self, targets):
        """
        Load precomputed target values. Can be mutual information.

        :param targets: Dictionary of dictionaries mapping target value to gene pairs.
        :type targets: dict
        """
        self.mi_scores = targets

    # def generate_mi_scores(self):
    #     print(bcolors.OKGREEN + "Getting gene pairs combinations." + bcolors.ENDC)
    #     mi_scores = self.mi_scores if self.mi_scores is not None else {}

    #     maxs = {gene.upper(): val for gene, val in zip(self.data.adata.var.index, np.array(self.data.adata.X.max(axis=0).T.todense()).ravel())}
    #     bcs = {gene: set(bc) for gene, bc in self.data.data.items()}
    #     vgenes = [gene for gene in self.data.data if maxs[gene.upper()] > 0]

    #     pairs = [(p1, p2) for p1, p2 in itertools.combinations(vgenes, 2) if not mi_scores.get(p1, {}).get(p2) and not mi_scores.get(p2, {}).get(p1)]
    #     self.num_pairs = len(pairs)

    #     counts = collections.defaultdict(lambda: collections.defaultdict(int))
    #     for c, p in self.data.expression.items():
    #         for g, v in p.items():
    #             counts[g][c] += int(v)

    #     print(bcolors.OKGREEN + "Computing MI for each pair." + bcolors.ENDC)
    #     if self.mi_scores:
    #         print("Found {} valid MI scores.".format(len(self.mi_scores)))

    #     for p1, p2 in tqdm.tqdm(pairs):
    #         common = bcs[p1].intersection(bcs[p2])
    #         if not common:
    #             continue

    #         c1, c2 = counts[p1], counts[p2]
    #         x, y = np.array([c1[bc] for bc in common]), np.array([c2[bc] for bc in common])

    #         pxy, _, _ = np.histogram2d(x, y, density=True)
    #         pxy /= pxy.sum()
    #         px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    #         px /= px.sum()
    #         py /= py.sum()
    #         px_py = np.outer(px, py)
    #         nzs = pxy > 0
    #         mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    #         if p1 not in mi_scores:
    #             mi_scores[p1] = {}
    #         if p2 not in mi_scores:
    #             mi_scores[p2] = {}
    #         mi_scores[p1][p2] = mi_scores[p2][p1] = mi

    #     self.mi_scores = mi_scores

    def generate_mi_scores(self):
        print(bcolors.OKGREEN + "Getting gene pairs combinations." + bcolors.ENDC)
        if self.mi_scores == None:
            mi_scores = collections.defaultdict(lambda : collections.defaultdict(float))
        else:
            mi_scores = self.mi_scores
        bcs = dict()
        maxs = dict(zip([x.upper() for x in self.data.adata.var.index.tolist()],numpy.array(self.data.adata.X.max(axis=0).T.todense()).T.tolist()[0]))
        vgenes = []
        for gene, bc in self.data.data.items():
            bcs[gene] = set(bc)
            if maxs[gene.upper()] > 0:
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
            if p1 not in mi_scores:
                mi_scores[p1] = dict()
                if p2 in mi_scores[p1]:
                    mi_scores[p1][p2] = dict()
            if p2 not in mi_scores:
                mi_scores[p2] = dict()
                if p2 not in mi_scores[p1]:
                    mi_scores[p2][p1] = dict()

            common = bcs[p1].intersection(bcs[p2])
            if len(common) == 0: continue
            
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

    def create_inputs_outputs(self, compute_mi = False):
        print(bcolors.WARNING+"*****************"+bcolors.ENDC)
        print(bcolors.HEADER+"Loading Dataset."+bcolors.ENDC)
        print(bcolors.WARNING+"*****************\n"+bcolors.ENDC)

        entropy = self.get_gene_entropy(self.adata)

        ent = []
        for g in self.data.genes:
            ent.append(entropy[g])

        if self.mi_scores == None or compute_mi:
            self.generate_mi_scores()
            
            if self.signed_mi:
                print("...Directional MI....")
                correlation_matrix = self.adata.to_df().corr()
                self.correlation = correlation_matrix.to_dict()

                modified_value_dict = {}

                for row_name in self.mi_scores.keys():
                    modified_value_dict[row_name] = {}
                    for col_name in self.mi_scores[row_name].keys():
                        original_value = self.mi_scores[row_name][col_name]
                        modified = self.correlation[row_name][col_name] * original_value
                        modified_value_dict[row_name][col_name] = round(modified,5)
                self.mi_scores = modified_value_dict
            
        print(bcolors.FAIL+"MI Loaded."+bcolors.ENDC)

        gene_index = {w: idx for (idx, w) in enumerate(self.data.genes)}
        index_gene = {idx: w for (idx, w) in enumerate(self.data.genes)}
        self.data.gene2id = gene_index
        self.data.id2gene = index_gene

        names=self.adata.var.index.tolist()

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        self._ei = list()
        for gene in names:
            self._ei.append(entropy)

        pairs = list(itertools.combinations(names,2))
        self.num_pairs = len(pairs)

        print(bcolors.OKGREEN + "Loading Batches for Training." + bcolors.ENDC)
    
        for gene in names:
            for cgene in names:
                wi = self.data.gene2id[gene]
                ci = self.data.gene2id[cgene]
                self._i_idx.append(wi)
                self._j_idx.append(ci)
                if cgene in self.mi_scores[gene]:
                    mivalue = round(self.mi_scores[gene][cgene],5)
                else:
                    mivalue =0.
                self._xij.append(mivalue)

        if self.device == "cuda":
            self._i_idx = torch.cuda.LongTensor(self._i_idx).cuda()
            self._j_idx = torch.cuda.LongTensor(self._j_idx).cuda()
            self._xij = torch.cuda.FloatTensor(self._xij).cuda()
            self._ent = torch.FloatTensor(ent).cuda()
        else:
            self._i_idx = torch.LongTensor(self._i_idx).to(self.device)
            self._j_idx = torch.LongTensor(self._j_idx).to(self.device)
            self._xij = torch.FloatTensor(self._xij).to(self.device)
            self._ent = torch.FloatTensor(ent).to(self.device)
        print(bcolors.OKCYAN + "Ready to train." + bcolors.ENDC)
    
    def get_batches(self, batch_size):
        if self.device == "cuda":
            rand_ids = torch.cuda.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        else:
            rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
