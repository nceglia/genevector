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

def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.to("cuda:0")

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).to("cuda:0")

class CompassModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        super(CompassModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.to("cuda:0")
        self.wj.to("cuda:0")
        self.bi.to("cuda:0")
        self.bj.to("cuda:0")

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()



    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

    def save_embedding(self, id2word, file_name):
        embedding = self.wi.weight.gpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.embedding_dim))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

class CompassTrainer(object):
    def __init__(self, dataset, output_file, emb_dimension=100, batch_size=1000, initial_lr=0.01, x_max=100, alpha=0.75):
        self.dataset = dataset
        self.output_file_name = output_file
        self.emb_size = len(self.dataset.data.gene2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.model = CompassModel(self.emb_size, self.emb_dimension)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=initial_lr)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0")
        #if self.use_cuda:
        self.model.to("cuda:0")
        print("\n\n\n\n{}\n\n\n\n".format(self.model.device))
        self.x_max = x_max
        self.alpha = alpha

    def train(self, epochs):
        n_batches = int(len(self.dataset._xij) / self.batch_size)
        loss_values = list()
        for e in range(1, epochs+1):
            batch_i = 0
            for x_ij, i_idx, j_idx in self.dataset.get_batches(self.batch_size):
                batch_i += 1
                self.optimizer.zero_grad()
                outputs = self.model(i_idx, j_idx)
                weights_x = weight_func(x_ij, self.x_max, self.alpha)
                loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
                loss.backward()
                self.optimizer.step()
                loss_values.append(loss.item())
                if batch_i % 100 == 0:
                    print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, epochs, batch_i, n_batches, np.mean(loss_values[-20:])))  
        print("Saving model...")
        self.model.save_embedding(self.dataset.data.id2gene, self.output_file_name)
