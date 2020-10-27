import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import scanpy as sc
from torch.autograd import Variable

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v, clip=10):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=clip, min=-clip)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=clip, min=-clip)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

class CompassTrainer(object):
    def __init__(self, dataset, output_file, emb_dimension=100, batch_size=1000, initial_lr=0.01):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate)
        self.output_file_name = output_file
        self.emb_size = len(self.dataset.data.gene2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        #self.use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()
    
    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate)

    def train(self, iterations, lr=None, negative_targets=5, discard_probability=0.05):
        for iteration in range(iterations):
            print("Epoch: {}".format(iteration+1))
            if not lr:
                lr = self.initial_lr
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=lr)
            self.dataset.update_discard_probability(discard_probability)
            self.dataset.update_negative_targets(negative_targets)
            self.dataloader = self.create_dataloader(self.dataset)
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    running_loss =+ loss.item()
            print("Loss:", running_loss)
            self.skip_gram_model.save_embedding(self.dataset.data.id2gene, self.output_file_name)
