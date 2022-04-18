import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import scanpy as sc
from torch.autograd import Variable

import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal

def mse_loss(inputs, targets, device):
    loss = F.mse_loss(inputs, targets, reduction='none')
    if device == "cuda":
        loss = loss.cuda()
    return torch.mean(loss).to(device)


class GeneVectorModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        super(GeneVectorModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.wi.weight.data.uniform_(-1., 1.)
        self.wj.weight.data.uniform_(-1., 1.)

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        x = torch.sum(w_i * w_j, dim=1)
        return x

    def save_embedding(self, id2word, file_name, layer):
        if layer == 0:
            embedding = self.wi.weight.cpu().data.numpy()
        else:
            embedding = self.wj.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.embedding_dim))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

class GeneVector(object):
    def __init__(self, dataset, output_file, emb_dimension=100, batch_size=1000, initial_lr=0.01, device="cpu", use_mi=True, distance=None, scale=100):
        self.dataset = dataset
        self.dataset.create_inputs_outputs(use_mi=use_mi, distance=distance, scale=scale)
        self.output_file_name = output_file
        self.emb_size = len(self.dataset.data.gene2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.use_cuda = torch.cuda.is_available()
        self.model = GeneVectorModel(self.emb_size, self.emb_dimension)
        self.device = device
        if self.device == "cuda" and not self.use_cuda:
            raise ValueError("CUDA requested but no GPU available.")
        elif self.device == "cuda":
            self.model.cuda()
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=initial_lr)


    def train(self, epochs):
        n_batches = int(len(self.dataset._xij) / self.batch_size)
        loss_values = list()
        for e in range(1, epochs+1):
            batch_i = 0
            for x_ij, i_idx, j_idx in self.dataset.get_batches(self.batch_size):
                batch_i += 1
                self.optimizer.zero_grad()
                outputs = self.model(i_idx, j_idx)
                loss = mse_loss(outputs, x_ij, self.device)
                loss.backward()
                self.optimizer.step()
                loss_values.append(loss.item())
                if batch_i % 100 == 0:
                    print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, epochs, batch_i, n_batches, np.mean(loss_values[-20:])))
        print("Saving model...")
        self.model.save_embedding(self.dataset.data.id2gene, self.output_file_name, 0)
        self.model.save_embedding(self.dataset.data.id2gene, self.output_file_name.replace(".vec","2.vec"), 1)
