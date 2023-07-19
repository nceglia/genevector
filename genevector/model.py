import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy
import matplotlib.pyplot as plt

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
        self.wi.weight.data.uniform_(-1, 1.)
        self.wj.weight.data.uniform_(-1.,1.)

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
    def __init__(self, dataset, output_file, emb_dimension=100, batch_size=None, c=1., device="cpu", correlation=False):
        self.dataset = dataset
        if correlation == False:
            self.dataset.create_inputs_outputs(c=c)
        else:
            self.dataset.generate_correlation(c=c)
        self.output_file_name = output_file
        self.emb_size = len(self.dataset.data.gene2id)
        self.emb_dimension = emb_dimension
        if batch_size == None and self.dataset.num_pairs:
            self.batch_size = self.dataset.num_pairs
        elif batch_size != None:
            self.batch_size = batch_size
        else:
            self.batch_size = 1e6
        self.use_cuda = torch.cuda.is_available()
        self.model = GeneVectorModel(self.emb_size, self.emb_dimension)
        self.device = device
        if self.device == "cuda" and not self.use_cuda:
            raise ValueError("CUDA requested but no GPU available.")
        elif self.device == "cuda":
            self.model.cuda()
        self.optimizer = optim.Adadelta(self.model.parameters())
        self.loss = nn.MSELoss()
        self.epoch = 0
        self.loss_values = list()
        self.mean_loss_values = []

    def train(self, epochs, threshold=None, update_interval=20):
        last_loss = 0.
        for _ in range(1, epochs+1):
            batch_i = 0
            for x_ij, i_idx, j_idx in self.dataset.get_batches(self.batch_size):
                batch_i += 1
                self.optimizer.zero_grad()
                outputs = self.model(i_idx, j_idx)
                loss = self.loss(outputs, x_ij)
                loss.backward()
                self.optimizer.step()
                self.loss_values.append(loss.item())
            self.mean_loss_values.append(numpy.mean(self.loss_values[-10:]))
            curr_loss = numpy.mean(self.loss_values[-10:])
            if self.epoch % int(update_interval) == 0:
                print(bcolors.OKGREEN + "**** Epoch" + bcolors.ENDC,
                    self.epoch, 
                    bcolors.HEADER+"\tLoss:"+bcolors.ENDC,
                    round(np.mean(self.loss_values[-30:]),5))
            if type(threshold) == float and abs(curr_loss - last_loss) < threshold:
                print(bcolors.OKCYAN + "Training complete!" + bcolors.ENDC)
                return
            last_loss = curr_loss
            self.epoch += 1
        print(bcolors.WARNING+"Saving model..."+bcolors.ENDC)
        self.model.save_embedding(self.dataset.data.id2gene, self.output_file_name, 0)
        self.model.save_embedding(self.dataset.data.id2gene, self.output_file_name.replace(".vec","2.vec"), 1)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.gnn.load_state_dict(torch.load(filepath))
        self.gnn.eval()

    def plot(self, fname=None, log=False):
        fig, ax = plt.subplots(1,1,figsize=(12,5),facecolor='#FFFFFF')
        ax.plot(self.mean_loss_values, color="purple")
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        if log:
            ax.set_xscale('log')
        if fname != None:
            fig.savefig(fname)
