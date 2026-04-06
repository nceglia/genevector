"""GeneVector dataset and expression context for single-cell data."""

import numpy as np
import torch
from torch.utils.data import Dataset
import numpy
from scipy import sparse
import itertools
import pickle
import os
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

    def __init__(self, adata, device="cpu", mi_scores=None, load_expression=True,
                 signed_mi=True, target="mi", target_kwargs=None,
                 mi_backend="auto", use_cache=True):
        """Constructor method

        Parameters
        ----------
        adata : AnnData
            The AnnData Scanpy object with expression data in .X.
        device : str
            Device for torch tensors ("cpu", "cuda", "mps").
        mi_scores : dict, optional
            Side-load precomputed target scores.
        load_expression : bool
            Whether to load expression into Context dicts.
        signed_mi : bool
            If True, multiply MI by correlation sign for directional MI.
        target : str or callable
            Name of registered target function, or a callable with
            signature ``f(X, gene_names, **kwargs) -> dict[dict[float]]``.
            Default: "mi" (mutual information).
        target_kwargs : dict, optional
            Extra keyword arguments passed to the target function.
        mi_backend : str
            Backend for MI computation: "auto", "numpy", "numba", "gpu".
            Only used when target="mi".
        use_cache : bool
            If True, cache computed target scores to disk and reload
            on subsequent runs with the same data+parameters.
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
        self.target = target
        self.target_kwargs = target_kwargs or {}
        self.mi_backend = mi_backend
        self.use_cache = use_cache
        self.num_pairs = None

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

    def save_target_scores(self, filepath):
        """Save computed target scores to a specific .npz file."""
        from .cache import save_scores
        key = filepath.replace(".npz", "").replace("/", "_")
        save_scores(key, self.mi_scores, self.data.genes)

    def load_target_scores(self, filepath):
        """Load target scores from a specific .npz file."""
        data = np.load(filepath, allow_pickle=False)
        matrix = data["scores"]
        gene_names = list(data["genes"])
        n = len(gene_names)
        self.mi_scores = collections.defaultdict(lambda: collections.defaultdict(float))
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] != 0:
                    self.mi_scores[gene_names[i]][gene_names[j]] = round(float(matrix[i, j]), 5)

    def _generate_mi_scores_legacy(self):
        """Legacy MI computation (kept for reference/testing)."""
        print(bcolors.OKGREEN + "Getting gene pairs combinations." + bcolors.ENDC)
        mi_scores = collections.defaultdict(lambda : collections.defaultdict(float))
        bcs = dict()
        vgenes = []
        for gene, bc in self.data.data.items():
            bcs[gene] = set(bc)
            vgenes.append(gene)
        pairs = list(itertools.combinations(vgenes, 2))
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

    def _compute_target_scores(self):
        """Dispatch to the appropriate target function."""
        from .metrics import get_target_function, TARGETS

        X = self.adata.X
        gene_names = self.data.genes

        # --- Check cache ---
        if self.use_cache:
            from .cache import compute_cache_key, load_scores, save_scores

            target_name = self.target if isinstance(self.target, str) else "custom"
            cache_key = compute_cache_key(
                X, gene_names, target_name, self.target_kwargs, self.signed_mi
            )
            cached_scores, _ = load_scores(cache_key)
            if cached_scores is not None:
                self.mi_scores = cached_scores
                self.num_pairs = len(gene_names) * (len(gene_names) - 1) // 2
                return

        # --- Compute ---
        if callable(self.target):
            print(bcolors.OKGREEN + "Computing custom target scores." + bcolors.ENDC)
            self.mi_scores = self.target(X, gene_names, **self.target_kwargs)
        elif isinstance(self.target, str):
            print(bcolors.OKGREEN + f"Computing '{self.target}' target scores." + bcolors.ENDC)
            fn = get_target_function(self.target)
            kwargs = {
                "signed": self.signed_mi,
                "backend": self.mi_backend,
                "device": self.device,
                **self.target_kwargs,
            }
            self.mi_scores = fn(X, gene_names, **kwargs)
        else:
            raise TypeError(f"target must be str or callable, got {type(self.target)}")

        self.num_pairs = len(gene_names) * (len(gene_names) - 1) // 2

        # --- Save to cache ---
        if self.use_cache:
            save_scores(cache_key, self.mi_scores, gene_names)

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

    def create_inputs_outputs(self, c=100.):
        """Compute target scores, build training tensors, and prepare for model training.

        Parameters
        ----------
        c : float
            Scaling factor applied to target scores (score * c^2).
        """
        print(bcolors.WARNING + "*****************" + bcolors.ENDC)
        print(bcolors.HEADER + "Loading Dataset." + bcolors.ENDC)
        print(bcolors.WARNING + "*****************\n" + bcolors.ENDC)

        # --- Entropy ---
        entropy = self.get_gene_entropy(self.adata)
        ent = [entropy[g] for g in self.data.genes]

        # --- Compute or load target scores ---
        if self.mi_scores is None:
            self._compute_target_scores()
        else:
            print(bcolors.OKCYAN + "Using preloaded target scores." + bcolors.ENDC)

        print(bcolors.FAIL + "Scores Loaded." + bcolors.ENDC)

        # --- Rebuild gene index ---
        gene_index = {w: idx for (idx, w) in enumerate(self.data.genes)}
        index_gene = {idx: w for (idx, w) in enumerate(self.data.genes)}
        self.data.gene2id = gene_index
        self.data.id2gene = index_gene

        # --- Build training tensors (vectorized) ---
        self._build_training_tensors(c=c)

        # --- Entropy tensor ---
        if self.device == "cuda":
            self._ent = torch.FloatTensor(ent).cuda()
        else:
            self._ent = torch.FloatTensor(ent).to(self.device)

        print(bcolors.OKCYAN + "Ready to train." + bcolors.ENDC)

    def _build_training_tensors(self, c=100.):
        """Build i_idx, j_idx, xij tensors using vectorized numpy ops."""
        genes = self.data.genes
        n = len(genes)

        # build dense score matrix from mi_scores dict
        score_matrix = np.zeros((n, n), dtype=np.float32)
        for i, g1 in enumerate(genes):
            if g1 in self.mi_scores:
                for j, g2 in enumerate(genes):
                    if i != j and g2 in self.mi_scores[g1]:
                        score_matrix[i, j] = self.mi_scores[g1][g2]

        score_matrix *= c ** 2

        if not self.signed_mi:
            score_matrix[score_matrix < 0] = 0.

        # build index arrays for all off-diagonal pairs
        idx = np.arange(n)
        i_grid, j_grid = np.meshgrid(idx, idx, indexing='ij')
        off_diag = i_grid != j_grid

        i_idx = i_grid[off_diag].ravel()
        j_idx = j_grid[off_diag].ravel()
        xij = score_matrix[off_diag].ravel()

        if self.device == "cuda":
            self._i_idx = torch.cuda.LongTensor(i_idx)
            self._j_idx = torch.cuda.LongTensor(j_idx)
            self._xij = torch.cuda.FloatTensor(xij)
        else:
            self._i_idx = torch.LongTensor(i_idx).to(self.device)
            self._j_idx = torch.LongTensor(j_idx).to(self.device)
            self._xij = torch.FloatTensor(xij).to(self.device)
    
    def get_batches(self, batch_size):
        """Yield randomized mini-batches of (target_values, i_indices, j_indices).

        Parameters
        ----------
        batch_size : int
            Number of gene pairs per batch.

        Yields
        ------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            Target values, row gene indices, column gene indices.
        """
        if self.device == "cuda":
            rand_ids = torch.cuda.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        else:
            rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
