# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter

from torch.nn import functional as F


def get_embedding_weight(num_embeddings, embedding_dim, use_cuda):
    t = torch.rand(num_embeddings, embedding_dim)
    t = t - torch.mean(t, 0).view(1, -1).repeat(num_embeddings, 1)
    if use_cuda:
        return Parameter(t).cuda()
    else:
        return Parameter(t)


class VariationalEmbedding(nn.Module):
    def __init__(self, n_tokens, embdim, _weight=None, _spread=None):
        super(VariationalEmbedding, self).__init__()
        self.weight = Parameter(_weight)
        self.spread = Parameter(_spread)
        self.n_tokens = self.weight.size(0)
        self.embdim = embdim

        assert list(_weight.shape) == [n_tokens, embdim], \
            'Shape of weight does not match num_embeddings and embedding_dim'

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, indices):
        mu = F.embedding(indices, self.weight)
        std = F.embedding(indices, self.spread)
        return self.reparameterize(mu, std)


class EmbeddingToIndex(nn.Module):
    def __init__(self, n_tokens, embdim, _weight=None):
        super(EmbeddingToIndex, self).__init__()
        self.weight = Parameter(_weight)
        self.n_tokens = self.weight.size(0)
        self.embdim = embdim

        assert list(_weight.shape) == [n_tokens, embdim], \
            'Shape of weight does not match num_embeddings and embedding_dim'
    
    def forward(self, X):  
        mb_size = X.size(0)
        adotb = torch.matmul(X, self.weight.permute(1, 0))

        if len(X.size()) == 3:
            seqlen = X.size(1)
            adota = torch.matmul(X.view(-1, seqlen, 1, self.embdim),
                                 X.view(-1, seqlen, self.embdim, 1))
            adota = adota.view(-1, seqlen, 1).repeat(1, 1, self.n_tokens)
        else: # (X.size()) == 2:
            adota = torch.matmul(X.view(-1, 1, self.embdim), X.view(-1, self.embdim, 1))
            adota = adota.view(-1, 1).repeat(1, self.n_tokens)

        bdotb = torch.bmm(self.weight.unsqueeze(-1).permute(0, 2, 1), self.weight.unsqueeze(-1)).permute(1, 2, 0)

        if len(X.size()) == 3:
            bdotb = bdotb.repeat(mb_size, seqlen, 1)
        else:
            bdotb = bdotb.reshape(1, self.n_tokens).repeat(mb_size, 1)

        dist = adota - 2 * adotb + bdotb

        return torch.min(dist, dim=len(dist.size()) - 1)[1]


class MakeMissing(nn.Module):
    def __init__(self, dropoutrate):
        super(MakeMissing, self).__init__()
        self._name = 'MakeMissing'
        self.dropoutrate = dropoutrate

    def forward(self, inputs):
        rvals = torch.rand(inputs.size()).cuda()
        r = torch.gt(torch.rand(inputs.size()).cuda(), self.dropoutrate*torch.ones_like(inputs)).float()
        return r*inputs - rvals*(torch.ones_like(r)-r)


class ReLUSigmoid(nn.Module):
    def __init__(self):
        super(ReLUSigmoid, self).__init__()
        self._name = 'ReLUSigmoid'

    def forward(self, inputs):
        r = torch.lt(inputs[:, 0].cuda(), torch.zeros_like(inputs[:, 0])).float()
        #rvals = torch.rand(inputs[:, 0].size()).cuda()
        #return -r*rvals + (torch.ones_like(r)-r)*torch.sigmoid(inputs[:, 1])
        return -r + (torch.ones_like(r) - r) * torch.sigmoid(inputs[:, 1])

class PlaceHolder(nn.Module):
    def __init__(self):
        super(PlaceHolder, self).__init__()
        self._name = 'place_holder'

    def forward(self, inputs):
        if len(inputs.size()) == 1:
            return inputs.view(-1, 1)
        else:
            return inputs

#choose an item from a tuple or list
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class SwapAxes(nn.Module):
    def __init__(self, reordered_axes):
        super(SwapAxes, self).__init__()
        self._name = 'SwapAxes'
        self.reordered_axes = reordered_axes

    def forward(self, inputs):
        return inputs.permute(self.reordered_axes)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self._name = 'Flatten'

    def forward(self, inputs):
        return inputs.reshape(inputs.size(0), -1)

class DropAll(nn.Module):
    def __init__(self, dropoutrate):
        super(DropAll, self).__init__()
        self._name = 'DropAll'
        self.dropoutrate = dropoutrate

    def forward(self, inputs):
        if self.training:
            r = torch.lt(torch.rand((inputs.size(0))).cuda(), self.dropoutrate).float()
            r = r.reshape([-1] + [1 for i in inputs.size()[1:]]).repeat((1,)+inputs.size()[1:])
            return (1-r)*inputs #-r*torch.ones_like(inputs) + (1-r)*inputs
        else:
            return inputs


def calc_input_size(input_dict, recurrent_hidden_size, num_layers, bidirectional, continuous_input_dim):
    input_size = 0

    if bidirectional:
        bidir = 2
    else:
        bidir = 1

    for inp in input_dict['discrete']:
        input_size += continuous_input_dim
        print(inp, continuous_input_dim)
    for inp in input_dict['continuous']:
        input_size += continuous_input_dim
        print(inp, continuous_input_dim)
    for inp in input_dict['text']:
        input_size += continuous_input_dim #bidir*recurrent_hidden_size*num_layers
        print(inp, continuous_input_dim)
    for inp in input_dict['onehot']:
        input_size += continuous_input_dim #input_dict['onehot'][inp]
        print(inp, continuous_input_dim) #input_dict['onehot'][inp])

    return input_size



class Decoder(nn.Module):
    def __init__(self, input_dict, maxlens, dim=24, recurrent_hidden_size=8):
        super(Decoder, self).__init__()
        self._name = 'decoder'
        self.dim = dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.input_dict = input_dict
        self.maxlens = maxlens
        self.hidden_dim = 8*self.dim
        self.bidirectional_rnn = True
        self.num_layers = 2

        self.preprocess = nn.Sequential(
                #nn.utils.weight_norm(nn.Linear(self.dim, int(self.hidden_dim/2)), dim=None),
                nn.Linear(self.dim, int(self.hidden_dim/2)),
                nn.ReLU(True),
                )

        self.main = nn.Sequential(
                #nn.utils.weight_norm(nn.Linear(int(self.hidden_dim/2), self.hidden_dim), dim=None),
                nn.Linear(int(self.hidden_dim / 2), self.hidden_dim),
                nn.ReLU(True),
                #nn.utils.weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim), dim=None),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(True),
                )

        '''
        self.main = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(int(self.hidden_dim/2), self.hidden_dim, 1)),
                #nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
                nn.ReLU(True),
                nn.utils.weight_norm(nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)),
                #nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
                nn.ReLU(True),
                Flatten()
                )
        '''


        self.outputs = {dtype: {} for dtype in self.input_dict.keys()}

        for col in self.input_dict['discrete']:
            #self.outputs['discrete'][col] = nn.utils.weight_norm(nn.Linear(self.hidden_dim, 1))
            self.outputs['discrete'][col] = self.get_numeric(self.hidden_dim)
            self.add_module(col + '_output', self.outputs['discrete'][col])

        for col in self.input_dict['continuous']:
            #self.outputs['continuous'][col] = nn.utils.weight_norm(nn.Linear(self.hidden_dim, 1))
            self.outputs['continuous'][col]= self.get_numeric(self.hidden_dim)
            self.add_module(col + '_output', self.outputs['continuous'][col])

        for col, emb_size in self.input_dict['text'].items():
            #self.outputs['text'][col] = nn.GRU(self.hidden_dim, emb_size, bidirectional=False, batch_first=True)
            self.outputs['text'][col] = self.get_rec(emb_size)
            self.add_module(col + '_output', self.outputs['text'][col])

        for col, emb_size in self.input_dict['onehot'].items():
            #self.outputs['onehot'][col] = nn.utils.weight_norm(nn.Linear(self.hidden_dim, emb_size))
            self.outputs['onehot'][col] = self.get_lin(self.hidden_dim, emb_size)
            self.add_module(col + '_output', self.outputs['onehot'][col])


    def get_numeric(self, hidden_size):
        net = nn.Sequential(
            #nn.utils.weight_norm(nn.Linear(hidden_size, int(hidden_size/2))),
            #nn.Linear(hidden_size, int(hidden_size / 2)),
            #nn.ReLU(True),
            #nn.utils.weight_norm(nn.Linear(hidden_size, int(hidden_size/1))),
            nn.Linear(hidden_size, int(hidden_size / 1)),
            nn.ReLU(True),
            #nn.utils.weight_norm(nn.Linear(hidden_size, int(hidden_size / 1))),
            nn.Linear(hidden_size, int(hidden_size / 1)),
            nn.ReLU(True),
            #nn.utils.weight_norm(nn.Linear(int(hidden_size / 1), 2)),
            #nn.utils.weight_norm(nn.Linear(int(hidden_size / 1), 1)),
            nn.Linear(int(hidden_size / 1), 1),
            nn.Sigmoid()
            #ReLUSigmoid()
            )
        return net

    def get_lin(self, hidden_size, emb_size):
        net = nn.Sequential(
            #nn.utils.weight_norm(nn.Linear(hidden_size, int(hidden_size/2))),
            #nn.Linear(hidden_size, int(hidden_size / 2)),
            #nn.ReLU(True),
            #nn.utils.weight_norm(nn.Linear(hidden_size, int(hidden_size/4))),
            nn.Linear(hidden_size, int(hidden_size / 4)),
            nn.ReLU(True),
            #nn.utils.weight_norm(nn.Linear(int(hidden_size / 4), emb_size)),
            nn.Linear(int(hidden_size / 4), emb_size),
            #nn.Tanh()
            )
        return net

    def get_rec(self, emb_size):
        if self.bidirectional_rnn:
            scale = 2
        else:
            scale = 1

        net = nn.Sequential(
            #nn.utils.weight_norm(nn.Linear(self.hidden_dim, self.dim)),
            nn.Linear(self.hidden_dim, self.dim),
            nn.Tanh(),
            nn.GRU(self.dim, emb_size, batch_first=True, bidirectional=self.bidirectional_rnn, num_layers=self.num_layers), #,
            SelectItem(0)
            )
        return net



    def forward(self, input):
        #x = self.main(input)

        x = self.preprocess(input)
        #x = x.view(x.size(0), -1, 1, 1)
        x = self.main(x)
        x = x.view(input.size(0), -1)

        output_dict = {}

        for col in self.input_dict['discrete']:
            output_dict[col] = self.outputs['discrete'][col](x)
        for col in self.input_dict['continuous']:
            output_dict[col] = self.outputs['continuous'][col](x)
        for col, emb_size in self.input_dict['text'].items():
            #n_tokens = self.input_dict['text'][col]
            #print(x.size())
            xrep = x.view(input.size(0), 1, -1).repeat(1, self.maxlens[col], 1)
            #print('xrep_'+col, xrep.size())
            output_dict[col] = self.outputs['text'][col](xrep)[:, :, -emb_size:] #[0]
            #print(col, output_dict[col].size())
        for col in self.input_dict['onehot'].keys():
            output_dict[col] = self.outputs['onehot'][col](x)

        return output_dict




#https://github.com/neale/Adversarial-Autoencoder/blob/master/generators.py
class Encoder(nn.Module):
    #can't turn dropout off completely because otherwise the loss -> NaN....
    #batchnorm does not seem to help things...
    def __init__(self, input_dict, dim=24, recurrent_hidden_size=8, sigmoidout=False):
        super(Encoder, self).__init__()
        self._name = 'encoder'
        self.sigmoidout = sigmoidout
        self.input_dict = input_dict
        self.dim = dim
        if self.sigmoidout:
            self.outputdim = 1
        else:
            self.outputdim = dim
        self.hidden_dim = 8*self.dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.dropout = 0.015625#0.03125 #0.0625 # 0.125 #
        self.bidirectional_rnn = True
        self.num_layers = 2
        self.continuous_input_dim = 32
        self.input_size = calc_input_size(self.input_dict, self.recurrent_hidden_size, self.num_layers, self.bidirectional_rnn, self.continuous_input_dim)



        #self.rec = {inp: nn.GRU(emb_size, self.recurrent_hidden_size, bidirectional=False) for inp, emb_size in self.input_dict['text'].items()}
        #for inp, layer in self.rec.items():
        #    self.add_module(inp+'_rec', layer)

        self.input_layers = {}
        for datatype in self.input_dict.keys():
            if datatype == 'text':
                for inp, emb_size in self.input_dict[datatype].items():
                    self.input_layers[inp] = self.get_rec(emb_size) #
                    self.add_module(inp + '_input', self.input_layers[inp])
            elif datatype == 'onehot':
                for inp, emb_size in self.input_dict[datatype].items():
                    self.input_layers[inp] = self.get_onehot(emb_size) #PlaceHolder()
                    self.add_module(inp + '_input', self.input_layers[inp])
            else:
                for inp in self.input_dict[datatype]:
                    self.input_layers[inp] = self.get_lin() #PlaceHolder()
                    self.add_module(inp + '_input', self.input_layers[inp])


        self.main = nn.Sequential(
                #nn.Dropout(p=self.dropout),
                #nn.utils.weight_norm(nn.Linear(self.input_size, int(self.hidden_dim/2)), dim=-1),
                nn.Linear(self.input_size, int(self.hidden_dim / 2)),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                #nn.utils.weight_norm(nn.Linear(int(self.hidden_dim/2), self.hidden_dim), dim=-1),
                nn.Linear(int(self.hidden_dim / 2), self.hidden_dim),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                #nn.Dropout(p=self.dropout),
                #nn.utils.weight_norm(nn.Linear(self.hidden_dim, self.dim), dim=-1),
                )
        '''
        self.main = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.utils.weight_norm(nn.Conv2d(self.input_size, self.hidden_dim, 1, dilation=1,  stride=1)),
                #nn.Conv2d(self.input_size, self.hidden_dim, 1, dilation=1, stride=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.utils.weight_norm(nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, dilation=1,  stride=1)),
                #nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, dilation=1, stride=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                )
        '''

        self.out = nn.Sequential(
                #nn.utils.weight_norm(nn.Linear(self.hidden_dim, self.outputdim)),
                nn.Linear(self.hidden_dim, self.outputdim)
                #nn.Linear(self.hidden_dim, self.dim),
                )



    def get_rec(self, emb_size):
        if self.bidirectional_rnn:
            bidir = 2
        else:
            bidir = 1
        net = nn.Sequential(
            nn.GRU(emb_size, self.recurrent_hidden_size, batch_first=True, bidirectional=self.bidirectional_rnn,
                   num_layers=self.num_layers), #nn.Dropout(p=self.dropout)
            SelectItem(1),
            SwapAxes((1, 0, 2)),
            Flatten(),
            #nn.utils.weight_norm(nn.Linear(self.recurrent_hidden_size*bidir*self.num_layers, self.continuous_input_dim)),
            nn.Linear(self.recurrent_hidden_size * bidir * self.num_layers, self.continuous_input_dim),
            #nn.Dropout(p=self.dropout),
            nn.ReLU(True),
            #DropAll(self.dropout)
        )
        return net

    def get_lin(self):

        net = nn.Sequential(
            PlaceHolder(),
            #MakeMissing(self.dropout),
            #nn.utils.weight_norm(nn.Linear(1, self.continuous_input_dim)),
            nn.Linear(1, self.continuous_input_dim),
            #nn.Dropout(p=self.dropout),
            nn.ReLU(),
            #nn.utils.weight_norm(nn.Linear(self.continuous_input_dim, self.continuous_input_dim)),
            nn.Linear(self.continuous_input_dim, self.continuous_input_dim),
            #nn.Dropout(p=self.dropout),
            nn.ReLU(),
            #DropAll(self.dropout)
        )
        return net

    def get_onehot(self, emb_size):

        net = nn.Sequential(
            PlaceHolder(),
            #nn.utils.weight_norm(nn.Linear(emb_size, self.continuous_input_dim)),
            nn.Linear(emb_size, self.continuous_input_dim),
            nn.ReLU(),
            #nn.utils.weight_norm(nn.Linear(self.continuous_input_dim, self.continuous_input_dim)),
            nn.Linear(self.continuous_input_dim, self.continuous_input_dim),
            #nn.Dropout(p=self.dropout),
            nn.ReLU(),
            #DropAll(self.dropout)
        )
        return net


    def forward(self, inputs):
        preprocessed_inputs = []

        for inp in self.input_dict['discrete']:
            #preprocessed_inputs.append(inputs[inp])
            preprocessed_inputs.append(self.input_layers[inp](inputs[inp]))
        for inp in self.input_dict['continuous']:
            #preprocessed_inputs.append(inputs[inp])
            t = self.input_layers[inp](inputs[inp])
            preprocessed_inputs.append(t)
        for inp in self.input_dict['text'].keys():
            #GRU returns output and hn - we just want hn
            #print('inp '+inp, inputs[inp].size())
            #t = self.input_layers[inp](inputs[inp])[1].permute(1, 0, 2)
            t = self.input_layers[inp](inputs[inp])
            #print(t.size())
            #t = torch.squeeze(t)

            preprocessed_inputs.append(t.reshape(inputs[inp].size(0), -1))


        for inp in self.input_dict['onehot'].keys():
            #preprocessed_inputs.append(inputs[inp])
            preprocessed_inputs.append(self.input_layers[inp](inputs[inp]))

        #for p in preprocessed_inputs:
        #    print(p.size(), p.dtype)

        x = torch.cat(preprocessed_inputs, 1)

        #x = x.view(x.size(0), -1, 1, 1)
        x = self.main(x)

        #x = self.main(x)
        x = x.view(x.size(0), -1)

        x = self.out(x)

        '''
        if self.sigmoidout:
            x = torch.sigmoid(x)
            #print('sigmoid')
        else:
            x = torch.tanh(x)
            #print('tanh')
        '''

        #print('enc out ', self.sigmoidout, x.size())
        return x



class VariationalEncoder(nn.Module):
    #can't turn dropout off completely because otherwise the loss -> NaN....
    #batchnorm does not seem to help things...
    def __init__(self, dim=24):
        super(VariationalEncoder, self).__init__()
        self._name = 'mnistE'
        self.shape = (1, 28, 28)
        self.dim = dim
        self.dropout = 0.03125
        convblock = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.utils.weight_norm(nn.Conv2d(1, 1*self.dim, 3, dilation=1,  stride=2, padding=1)),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.utils.weight_norm(nn.Conv2d(1*self.dim, 2*self.dim, 3, dilation=1,  stride=2, padding=1)),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.utils.weight_norm(nn.Conv2d(2*self.dim, 4*self.dim, 3, dilation=1,  stride=2, padding=1)),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                )
        self.main = convblock

        #self.get_mu = nn.Linear(4*4*4*self.dim, self.dim)
        self.get_mu = nn.utils.weight_norm(nn.Linear(4 * 4 * 4 * self.dim, self.dim))
        self.get_logvar = nn.utils.weight_norm(nn.Linear(4 * 4 * 4 * self.dim, self.dim))
        #self.get_logvar = nn.Linear(4*4*4*self.dim, self.dim)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            #std = 0.5*torch.ones_like(mu)
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        mu = self.get_mu(out)
        logvar = self.get_logvar(out)
        z = self.reparameterize(mu, logvar)
        return z.view(z.size(0), -1), logvar