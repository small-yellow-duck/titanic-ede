import importlib as imp

import pandas as pd
from collections import Counter
import itertools

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset

import torch.nn as nn

import contrastive
import net



use_cuda = True

to_predict = ['Survived']
onehot_cols = ['Pclass', 'Embarked', 'Sex', 'Survived']  # Survived
discrete_cols = [] #['Parch']
continuous_cols = ['Fare', 'Age'] #
text_cols = []#['Ticket', 'Cabin', 'Name'] # 'Ticket', 'Cabin',   'Cabin']

margin=1.0
missing_val = -1.0


'''
add a '' entry to charcounts to account for test-time chars that aren't in the training set
'''


#rawdata, charcounts, maxlens, unique_onehotvals = fetch_and_preprocess()
def fetch_and_preprocess():
    data = pd.read_csv('train.csv')
    charcounts = Counter('')
    maxlens = {}
    for c in text_cols:
        data[c] = data[c].apply(lambda x : str(x).lower() if (str(x) !='nan') else '')
        charcounts += data[c].apply(Counter).sum()

        maxlens[c] = data[c].apply(lambda x: len(x)).max()

    unique_onehotvals = {}
    for c in onehot_cols:
        unique_onehotvals[c] = data[c].unique()

    #data.drop('PassengerId', 1, inplace=True)
    maxlens['Name'] = 20
    maxlens['Ticket'] = 7
    maxlens['Cabin'] = 3

    return data, charcounts, maxlens, unique_onehotvals



class Dataseq(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, charcounts, input_dict, unique_onehotvals, maxlens):
        """
        Args:
            data: pandas dataframe
        """
        self.data = data #assignment is done with a shallow copy in python, so data is not duplicated
        self.charcounts = charcounts
        self.charindex = {k: i for i, k in enumerate(charcounts.keys(), 1)}
        self.charindexreverse = {i: k for i, k in enumerate(charcounts.keys(), 1)}
        self.input_dict = input_dict
        self.unique_onehotvals = unique_onehotvals
        self.maxlens = maxlens
        self.onehotindex = {}
        self.onehotindexreverse = {}
        for k in input_dict['onehot']:
            self.onehotindex[k] = {v: i for i, v in enumerate(unique_onehotvals[k])}
            self.onehotindexreverse[k] = {i: v for i, v in enumerate(unique_onehotvals[k])}
        self.cols = [col for dtype in self.input_dict.keys() for col in self.input_dict[dtype]]

        self.scalings = {}
        for k in input_dict['continuous']:
            self.scalings[k] = {'min': self.data[k].min() - 0.05*(self.data[k].max()-self.data[k].min()), 'max': self.data[k].max() + 0.05*(self.data[k].max()-self.data[k].min())}
            self.scalings[k]['mean'] = self.data[k].mean()
            self.scalings[k]['std'] = self.data[k].std()

    def __len__(self):
        return self.data.shape[0]

    def __getrawitem__(self, idx):
        #sample = {col: self.data.loc[idx, col].values for dtype in self.input_dict.keys() for col in self.input_dict[dtype]}
        sample = self.data.loc[idx, self.cols].to_dict()
        return sample

    def __getitem__(self, idx):
        sample = self.__getrawitem__(idx)

        #encode onehot variables
        for k in self.input_dict['onehot'].keys():
            sample[k] = self.onehotindex[k][sample[k]]

        #encode text variables
        for k in self.input_dict['text'].keys():
            t = np.array([self.charindex[c] for c in sample[k]], dtype=int)[0:self.maxlens[k]]
            #print(sample[k], t)
            sample[k] = np.zeros(self.maxlens[k], dtype=int)
            sample[k][0:t.shape[0]] = t

        #scale continous variables
        for k in self.input_dict['continuous']:
            sample[k] = (sample[k] - self.scalings[k]['min']) / (self.scalings[k]['max'] - self.scalings[k]['min'])
            #sample[k] = (sample[k] - self.scalings[k]['mean']) / self.scalings[k]['std'] / 2.5

            if np.isnan(sample[k]):
                sample[k] = missing_val #-np.random.rand() #
        return sample



def discretize(X2d, embeddings, maxlens):
    T2 = {col: X2d[col] for col in X2d.keys()}
    mb_size = X2d[list(X2d.keys())[0]].size(0)

    for col, embedding in embeddings.items():
        n_tokens = embedding.weight.size(0)
        embedding_dim = embedding.weight.size(1)


        adotb = torch.matmul(X2d[col], embedding.weight.permute(1, 0))
        # adota = torch.bmm(X2d.view(-1, 1, embedding_dim), X2d.view(-1, embedding_dim, 1))

        if col in maxlens.keys():
            adota = torch.matmul(X2d[col].view(-1, maxlens[col], 1, embedding_dim),
                                 X2d[col].view(-1, maxlens[col], embedding_dim, 1))
            adota = adota.view(-1, maxlens[col], 1).repeat(1, 1, n_tokens)
        else:
            adota = torch.matmul(X2d[col].view(-1, 1, embedding_dim), X2d[col].view(-1, embedding_dim, 1))
            adota = adota.view(-1, 1).repeat(1, n_tokens)

        bdotb = torch.bmm(embedding.weight.unsqueeze(-1).permute(0, 2, 1), embedding.weight.unsqueeze(-1)).permute(1, 2,
                                                                                                                   0)
        if col in maxlens.keys():
            bdotb = bdotb.repeat(mb_size, maxlens[col], 1)
        else:
            bdotb = bdotb.reshape(1, n_tokens).repeat(mb_size, 1)

        dist = adota - 2 * adotb + bdotb

        T2[col] = torch.min(dist, dim=len(dist.size()) - 1)[1]

    #for col in continuous_cols:
    #    T2[col] = -1.0*torch.lt(T2[col], torch.zeros_like(T2[col])).float() + T2[col]*torch.gt(T2[col], torch.zeros_like(T2[col])).float()

    return T2



def are_equal(x0):
    equ = None
    mb_size = x0[list(x0.keys())[0]].size(0)
    for col in x0.keys():
        if len(x0[col].size()) == 1:
            #t = (torch.eq(x0[col], x1[col])).float() + (torch.lt(x0[col], torch.zeros_like(x0[col]))).float()*(torch.lt(x1[col], torch.zeros_like(x1[col]))).float()
            t0 = x0[col].float()*(torch.gt(x0[col].float(), torch.zeros_like(x0[col].float()))).float() - (torch.lt(x0[col].float(), torch.zeros_like(x0[col].float()))).float()
            t0 = t0.view(x0[col].size()+(1,))
            t1 = t0.permute(0, 1).repeat(1, mb_size, 1)
            t2 = t0.permute(1, 0).repeat(mb_size, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)
        else: #len(x0[col].size()) == 2:
            t0 = x0[col].view(x0[col].size()+(1,))
            t1 = t0.permute(0, 2, 1).repeat(1, mb_size, 1)
            t2 = t0.permute(2, 0, 1).repeat(mb_size, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)



        if equ is None:
            equ = torch.floor(t)
        else:
            equ *= torch.floor(t)
    return equ


def are_equal2(x0):
    equ = None
    mb_size = x0[list(x0.keys())[0]].size(0)
    for col in x0.keys():
        if len(x0[col].size()) == 1:
            #t = (torch.eq(x0[col], x1[col])).float() + (torch.lt(x0[col], torch.zeros_like(x0[col]))).float()*(torch.lt(x1[col], torch.zeros_like(x1[col]))).float()
            t0 = x0[col]*(torch.gt(x0[col], torch.zeros_like(x0[col]))).float() - (torch.lt(x0[col], torch.zeros_like(x0[col]))).float()
            t0 = t0.view(x0[col].size()+(1,))
            t1 = t0.permute(0, 1).repeat(1, mb_size, 1)
            t2 = t0.permute(1, 0).repeat(mb_size, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)
        elif len(x0[col].size()) == 2:
            t0 = x0[col].view(x0[col].size()+(1,))
            t1 = t0.permute(0, 2, 1).repeat(1, mb_size, 1)
            t2 = t0.permute(2, 0, 1).repeat(mb_size, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)

        else: # len(x0[col].size()) == 3:
            t0 = x0[col].view(x0[col].size()+(1,))
            t1 = t0.permute(0, 3, 1, 2).repeat(1, mb_size, 1, 1)
            t2 = t0.permute(3, 0, 1, 2).repeat(mb_size, 1, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)

        if equ is None:
            equ = torch.floor(t)
        else:
            equ *= torch.floor(t)
    return equ



#T, X, X2, mu, mu2, mu2d, mu_tm, logvar_tm, logvar2d = calc_losses(T, embeddings, enc, dec)
def calc_mus(T, embeddings, reverse_embeddings, enc, dec, mode='train'):
    mb_size = T[list(T.keys())[0]].size(0)
    n_targetvals = embeddings[to_predict[0]].weight.size(0)

    if use_cuda:
        T = {col: Variable(tt).cuda() for col, tt in T.items()}
    else:
        T = {col: Variable(tt) for col, tt in T.items()}

    X = {}
    for col, tt in T.items():
        if col in embeddings.keys():
            X[col] = embeddings[col](tt)
        else:
            X[col] = tt.float()

    # mu, logvar = enc(X)
    # mu = enc.reparameterize(mu, logvar)

    # mu_tm, logvar_tm = enc({col: X[col] if not col in to_predict else 0.0*X[col] for col in X.keys() })
    # encode with the incorrect
    # mu_false, logvar_false = enc({col: X[col] if not col in to_predict else embeddings[col](1-T[col]) for col in X.keys()})



    # make the first element of mu_tm contain the encoding of the input where the target is zeroes (ie encoded as missing)
    # mu_tm = [enc({col: X[col] if not col in to_predict else 0.0 * X[col] for col in X.keys()})[0]]

    mu, logvar = enc({col: X[col] if not col in to_predict else 0.0 * X[col] for col in X.keys()})
    if mode == 'train':
        mu = enc.reparameterize(mu, logvar)
    mu_tm = torch.zeros_like(mu).unsqueeze(1).repeat((1, 1+n_targetvals, 1))
    mu_tm[:, 0, :] = mu

    latent_dim = mu.size(1)


    # encodings for all the possible target embedding values
    for i in range(n_targetvals):
        m, lgv = enc({col: X[col] if not col in to_predict else embeddings[col](i * torch.ones_like(T[col])) for col in
                      X.keys()})
        if mode == 'train':
            m = enc.reparameterize(m, lgv)

        #use = torch.eq(i * torch.ones_like(T[to_predict[0]]), T[to_predict[0]]).float().view(-1, 1).repeat(1, latent_dim)
        #use[0:int(1*mb_size/2)] = 0.0
        #use[0:int(1 * mb_size / 1)] = 0.0
        #mu = mu + m * use - use * mu
        #logvar = logvar + lgv * use - use * logvar

        mu_tm[:, i+1, :] = m


    X2 = dec(mu)

    mu2, logvar2 = enc(X2)
    if mode == 'train':
       mu2 = enc.reparameterize(mu2, logvar2)
    mu2 = mu2.view(mb_size, -1)

    T2 = {}
    X2d = {col: (1.0 * tt).detach() for col, tt in X2.items()}

    #for col, embedding in embeddings.items():
    #    T2[col] = reverse_embeddings[col](X2[col])
    #    X2d[col] = 0.8*embeddings[col](T2[col].detach()) + 0.2*X2d[col]

    mu2d, logvar2d = enc(X2d)

    if mode == 'train':
        mu2d = enc.reparameterize(mu2d, logvar2d)



    mu2d = mu2d.view(mb_size, -1)

    mu = mu.view(mb_size, -1)

    return T, X, X2, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d


#enc_loss, enc_loss0, enc_loss1, enc_loss2, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logloss)
def calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logloss, lookfordups=True):
    mb_size = mu.size(0)
    latent_dim = mu.size(1)
    n_targetvals = mu_tm.size(1) - 1 #len(mu_tm) - 1

    '''
    diff1 = 0.5*torch.pow(mu - mu2, 2)/(torch.exp(2*logvar) + torch.exp(2*logvar2))
    enc_loss1 = 0.5 * logloss(torch.mean(diff1, 1), torch.ones(mb_size).cuda())
    '''
    adotb = torch.matmul(mu, mu2.permute(1, 0))  # batch_size x batch_size
    adota = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
    bdotb = torch.matmul(mu2.view(-1, 1, latent_dim), mu2.view(-1, latent_dim, 1))
    diffsquares = (adota.view(-1, 1).repeat(1, mb_size) + bdotb.view(1, -1).repeat(mb_size, 1) - 2 * adotb) / latent_dim
    tt = np.triu_indices(mb_size, k=1)
    diffsquares = diffsquares[tt]

    adotb2 = torch.matmul(mu, mu.permute(1, 0))  # batch_size x batch_size
    adota2 = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
    diffsquares2 = (adota2.view(-1, 1).repeat(1, mb_size) + adota2.view(1, -1).repeat(mb_size,
                                                                                      1) - 2 * adotb2) / latent_dim
    diffsquares2 = diffsquares2[tt]

    # are_same = are_equal({col: x[::2] for col, x in T.items()}, {col: x[1::2] for col, x in T.items()})
    #are_same = are_equal2(X)
    if lookfordups:
        are_same = are_equal(T)
        are_same = are_same[tt]
    else:
        are_same = torch.zeros_like(diffsquares)

    # print('shapes', are_same.size())
    # print('fraction same', torch.mean(are_same))
    enc_loss0 = 0.125 * logloss(diffsquares, are_same, weights=1.0 - are_same)
    enc_loss0 += 0.125 * logloss(diffsquares2, are_same, weights=1.0 - are_same)

    # enc_loss0 = logloss(torch.mean(torch.pow(mu[::2]-mu[1::2], 2), 1), are_same, weights=1-are_same)
    enc_loss1 = 0.5 * logloss(torch.mean(torch.pow(mu - mu2, 2), 1), torch.ones(mb_size).cuda())
    #enc_loss1 = 0.5 * logloss(torch.mean(torch.pow(mu_tm[:, 0, :] - mu2, 2), 1), torch.ones(mb_size).cuda())
    enc_loss2 = 0.25 * logloss(torch.mean(torch.pow(mu - mu2d, 2), 1), torch.zeros(mb_size).cuda())
    # enc_loss3 = 0.5 * logloss(torch.mean(torch.pow(mu - mu_tm, 2), 1), torch.ones(mb_size).cuda())
    # enc_loss3 += 0.5 * logloss(torch.mean(torch.pow(mu - mu_false, 2), 1), torch.zeros(mb_size).cuda())

    enc_loss3 = 0.0
    for i in range(n_targetvals):
        if use_cuda:
            target = torch.eq(i * torch.ones_like(T[to_predict[0]]), T[to_predict[0]]).float().cuda()
        else:
            target = torch.eq(i * torch.ones_like(T[to_predict[0]]), T[to_predict[0]]).float()
        factor = 0.5 * (1.0 / (n_targetvals - 1) * (1 - target) + target)
        enc_loss3 += torch.mean(
            factor * logloss(torch.mean(torch.pow(mu_tm[:, 0, :] - mu_tm[:, i + 1, :], 2), 1), target, do_batch_mean=False))

    enc_loss = enc_loss0 + enc_loss1 + enc_loss2 + 8.0*enc_loss3
    # enc_loss += 0.0025*torch.mean(torch.pow(mu, 2))
    # enc_loss += 0.0025 * torch.mean(torch.pow(mu2, 2))
    # enc_loss += 0.0025 * torch.mean(torch.exp(logvar) - logvar)
    #enc_loss += 1.0/8*0.25 * torch.mean(torch.pow(mu, 2))
    #enc_loss += 1.0/8*0.25 * torch.mean(torch.pow(mu2, 2))
    #enc_loss += 1.0/8*0.5 * torch.mean(torch.exp(logvar) - logvar)
    enc_loss += 1.0/64.0 * torch.mean(torch.pow(mu, 2))
    enc_loss += 1.0/64.0 * torch.mean(torch.pow(mu2, 2))
    enc_loss += 1.0/64.0 * torch.mean(torch.exp(logvar) - logvar)
    enc_loss += 1.0/256.0 * torch.mean(torch.exp(logvar2) - logvar2)
    enc_loss += 1.0/256.0 * torch.mean(torch.exp(logvar2d) - logvar2d)
    #for col, emb in embeddings.items():
    #    enc_loss += torch.mean(torch.pow(emb.weight, 2))

    #enc_loss += 0.025 * torch.mean(torch.exp(logvar2) - logvar2)
    # for k in onehot_cols:
    #    enc_loss += 0.0025*torch.mean(torch.pow(onehot_embedding_weights[k], 2))


    return enc_loss, enc_loss0, enc_loss1, enc_loss2, enc_loss3


def do_train(rawdata, charcounts, maxlens, unique_onehotvals):
    n_batches = 2000
    mb_size = 128
    lr = 2.0e-4
    momentum = 0.5
    cnt = 0
    latent_dim = 32 #24#
    recurrent_hidden_size = 24

    epoch_len = 8
    max_veclen = 0.0
    patience = 12 * epoch_len
    patience_duration = 0

    # mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    input_dict = {}
    input_dict['discrete'] = discrete_cols
    input_dict['continuous'] = continuous_cols

    input_dict['onehot'] = {}
    for k in onehot_cols:
        dim = int(np.ceil(np.log(len(unique_onehotvals[k])) / np.log(2.0)))
        input_dict['onehot'][k] = dim

    if len(charcounts) > 0:
        text_dim = int(np.ceil(np.log(len(charcounts)) / np.log(2.0)))
        input_dict['text'] = {t: text_dim for t in text_cols}
    else:
        text_dim = 0
        input_dict['text'] = {}

    data = Dataseq(rawdata, charcounts, input_dict, unique_onehotvals, maxlens)
    data_idx = np.arange(data.__len__())
    np.random.shuffle(data_idx)
    n_folds = 6
    fold_size = 1.0 * data.__len__() / n_folds
    folds = [data_idx[int(i * fold_size):int((i + 1) * fold_size)] for i in range(6)]

    fold_groups = {}
    fold_groups[0] = {'train': [0, 1, 2, 4], 'es': [3], 'val': [5]}
    fold_groups[1] = {'train': [0, 2, 3, 5], 'es': [1], 'val': [4]}
    fold_groups[2] = {'train': [1, 3, 4, 5], 'es': [2], 'val': [0]}
    fold_groups[3] = {'train': [0, 2, 3, 4], 'es': [5], 'val': [1]}
    fold_groups[4] = {'train': [0, 1, 3, 5], 'es': [4], 'val': [2]}
    fold_groups[5] = {'train': [1, 2, 4, 5], 'es': [0], 'val': [3]}

    for fold in range(1):

        train_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['train']])))
        es_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['es']])))
        val_idx = np.array(folds[fold_groups[fold]['val'][0]])

        train = Subset(data, train_idx)
        es = Subset(data, es_idx)
        val = Subset(data, val_idx)

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_iter = torch.utils.data.DataLoader(train, batch_size=int(mb_size/1), shuffle=True, **kwargs)
        train_iter_unshuffled = torch.utils.data.DataLoader(train, batch_size=mb_size, shuffle=False, **kwargs)
        es_iter = torch.utils.data.DataLoader(es, batch_size=mb_size, shuffle=True, **kwargs)
        val_iter = torch.utils.data.DataLoader(val, batch_size=mb_size, shuffle=True, **kwargs)

        embeddings = {}
        reverse_embeddings = {}
        onehot_embedding_weights = {}
        onehot_embedding_spread = {}
        for k in onehot_cols:
            dim = input_dict['onehot'][k]
            onehot_embedding_weights[k] = net.get_embedding_weight(len(unique_onehotvals[k]), dim, use_cuda=use_cuda)
            embeddings[k] = nn.Embedding(len(unique_onehotvals[k]), dim, _weight=onehot_embedding_weights[k])
            reverse_embeddings[k] = net.EmbeddingToIndex(len(unique_onehotvals[k]), dim, _weight=onehot_embedding_weights[k])

        if text_dim > 0:
            text_embedding_weights = net.get_embedding_weight(len(charcounts) + 1, text_dim, use_cuda=use_cuda)
            text_embedding = nn.Embedding(len(charcounts) + 1, text_dim, _weight=text_embedding_weights)
            text_embeddingtoindex = net.EmbeddingToIndex(len(charcounts) + 1, text_dim, _weight=text_embedding_weights)
            for k in text_cols:
                embeddings[k] = text_embedding
                reverse_embeddings[k] = text_embeddingtoindex

        enc = net.Encoder(input_dict, dim=latent_dim, recurrent_hidden_size=recurrent_hidden_size)
        dec = net.Decoder(input_dict, maxlens, dim=latent_dim, recurrent_hidden_size=recurrent_hidden_size)

        if use_cuda:
            embeddings = {k: embeddings[k].cuda() for k in embeddings.keys()}
            enc.cuda()
            dec.cuda()


        #print(enc.parameters)
        #print(dec.parameters)


        #contrastivec = contrastive.ContrastiveLoss(margin=margin)
        logloss = contrastive.GaussianOverlap()


        #solver = optim.RMSprop([p for em in embeddings.values() for p in em.parameters()] +  [p for p in enc.parameters()] + [p for p in dec.parameters()], lr=lr)
        #solver = optim.Adam(
        #    [p for em in embeddings.values() for p in em.parameters()] + [p for p in enc.parameters()] + [p for p in
        #                                                                                                  dec.parameters()],
        #    lr=lr)

        solver = optim.RMSprop(
            [p for em in embeddings.values() for p in em.parameters()] + [p for p in enc.parameters()] + [p for p in
                                                                                                          dec.parameters()],
            lr=lr, momentum=momentum)

        Tsample = next(es_iter.__iter__())
        if use_cuda:
            Tsample = {col: Variable(tt).cuda() for col, tt in Tsample.items()}
        else:
            Tsample = {col: Variable(tt) for col, tt in Tsample.items()}

        print({col: tt[0] for col, tt in Tsample.items()})

        print('starting training')
        loss = 0.0
        loss0 = 0.0
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0

        logger_df = pd.DataFrame(columns=['iter', 'train_loss', 'train_veclen', 'es_veclen', 'Survived_correct', 'Survived_false'])

        for it in range(n_batches):
            # X = Variable(torch.tensor(np.array([[1,2,4], [4,1,9]]))).cuda()
            T = next(iter(train_iter))
            #for col, val in T.items():
            #    T[col] = torch.cat((val, val, val, val), 0)

            T, X, X2, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d = calc_mus(T, embeddings, reverse_embeddings, enc, dec)
            enc_loss, enc_loss0, enc_loss1, enc_loss2, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logloss)

            enc_loss.backward()
            solver.step()

            enc.zero_grad()
            dec.zero_grad()
            for col in embeddings.keys():
                embeddings[col].zero_grad()

            loss += enc_loss.data.cpu().numpy()
            loss0 += enc_loss0.data.cpu().numpy()
            loss1 += enc_loss1.data.cpu().numpy()
            loss2 += enc_loss2.data.cpu().numpy()
            loss3 += enc_loss3.data.cpu().numpy()
            veclen = torch.mean(torch.pow(mu, 2))
            if it % epoch_len == 0:
                print(it, loss/epoch_len, loss0/epoch_len, loss1/epoch_len, loss2/epoch_len, loss3/epoch_len, veclen.data.cpu().numpy()) #enc_loss.data.cpu().numpy(),


                if use_cuda:
                    mu = torch.zeros(len(train), mu.size(1)).cuda()
                    logvar = torch.zeros(len(train), mu.size(1)).cuda()
                    mu2 = torch.zeros(len(train), mu.size(1)).cuda()
                    mu2d = torch.zeros(len(train), mu.size(1)).cuda()
                    mu_tm = torch.zeros((len(train),) + mu_tm.size()[1:]).cuda()
                    logvar2 = torch.zeros(len(train), mu.size(1)).cuda()
                    logvar2d = torch.zeros(len(train), mu.size(1)).cuda()
                else:
                    mu = torch.zeros(len(train), mu.size(1))
                    logvar = torch.zeros(len(train), mu.size(1))
                    mu2 = torch.zeros(len(train), mu.size(1))
                    mu2d = torch.zeros(len(train), mu.size(1))
                    mu_tm = torch.zeros((len(train),) + mu_tm.size()[1:])
                    logvar2 = torch.zeros(len(train), mu.size(1))
                    logvar2d = torch.zeros(len(train), mu.size(1))

                s = 0
                for T0 in train_iter_unshuffled:
                    e = s + T0[to_predict[0]].size(0)
                    if s == 0:
                        T = {col : torch.zeros((len(train),) + val.size()[1:], dtype=val.dtype) for col, val in T0.items()}

                    T0, blah, bblah, mu[s:e], logvar[s:e], mu2[s:e], mu2d[s:e], mu_tm[s:e], logvar2[s:e], logvar2d[s:e] = calc_mus(T0, embeddings, reverse_embeddings,  enc, dec, mode='val')
                    for col, val in T0.items():
                        T[col][s:e] = T0[col]

                    s = e

                enc_loss, enc_loss0, enc_loss1, enc_loss3, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logloss, lookfordups=False)
                vl = torch.mean(torch.pow(mu, 2))

                print(f'train enc loss {enc_loss}')
                print(f'train veclen {vl}')
                print(f'mean train logvar {torch.mean(logvar)}')
                logger_df.loc[int(it/epoch_len), ['iter', 'train_loss', 'train_veclen']] = [it, enc_loss.data.cpu().numpy(), vl.data.cpu().numpy()]

                if use_cuda:
                    mu = torch.zeros(len(es), mu.size(1)).cuda()
                    logvar = torch.zeros(len(es), mu.size(1)).cuda()
                    mu2 = torch.zeros(len(es), mu.size(1)).cuda()
                    mu2d = torch.zeros(len(es), mu.size(1)).cuda()
                else:
                    mu = torch.zeros(len(es), mu.size(1))
                    logvar = torch.zeros(len(es), mu.size(1))
                    mu2 = torch.zeros(len(es), mu.size(1))
                    mu2d = torch.zeros(len(es), mu.size(1))

                s = 0
                targets = {}
                for T0 in es_iter:
                    e = s + T0[to_predict[0]].size(0)
                    if s == 0:
                        T = {col : torch.zeros((len(es),) + val.size()[1:], dtype=val.dtype) for col, val in T0.items()}
                        correct = {col: np.zeros((len(es),) + val.size()[1:]) for col, val in T0.items()}
                        actual = {col: np.zeros((len(es),) + val.size()[1:]) for col, val in T0.items()}

                    Xsample = {}
                    for col, tt in T0.items():
                        if use_cuda:
                            tt = Variable(tt).cuda()
                        else:
                            tt = Variable(tt)

                        if col in embeddings.keys():
                            Xsample[col] = embeddings[col](tt)
                        else:
                            Xsample[col] = tt.float()



                    for col in to_predict:
                        targets[col] = tt
                        Xsample[col] = 0.0 * Xsample[col]


                    mu[s:e], logvar[s:e] = enc(Xsample)

                    X2sample = dec(mu[s:e])
                    T2sample = discretize(X2sample, embeddings, maxlens)

                    mu2[s:e], _ = enc(X2sample)


                    T2 = {}
                    X2dsample = {col: (1.0 * tt).detach() for col, tt in X2sample.items()}
                    for col in continuous_cols:
                        if col in to_predict:
                            correct[col][s:e] = np.abs(X2sample[col].data.cpu().numpy().reshape(-1) - targets[
                                col].data.cpu().numpy().reshape(-1))
                            actual[col][s:e] = targets[col].data.cpu().numpy().reshape(-1)
                        else:
                            correct[col][s:e] = np.abs(X2sample[col].data.cpu().numpy().reshape(-1) - T0[
                                col].data.cpu().numpy().reshape(-1))
                            actual[col][s:e] = T0[col].data.cpu().numpy().reshape(-1)


                    for col, embedding in embeddings.items():
                        # T2[col] = reverse_embeddings[col](X2sample[col])
                        X2dsample[col] = embeddings[col](T2sample[col].detach())

                        if col in to_predict:
                            correct[col][s:e] = np.abs(T2sample[col].data.cpu().numpy() == targets[col].data.cpu().numpy())
                            actual[col][s:e] = targets[col].data.cpu().numpy().reshape(-1)
                        else:
                            correct[col][s:e] = np.abs(T2sample[col].data.cpu().numpy() == T0[col].data.cpu().numpy())
                            actual[col][s:e] = T0[col].data.cpu().numpy().reshape(-1)

                    mu2d[s:e], _ = enc(X2dsample)

                    s = e

                #enc_loss, enc_loss0, enc_loss1, enc_loss3, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logloss, lookfordups=False)
                #print(f'es enc loss {enc_loss}')
                vl = torch.mean(torch.pow(mu, 2))

                print(f'es veclen {vl}')
                print(f'mean es logvar {torch.mean(logvar)}')
                logger_df.loc[int(it/epoch_len), ['es_veclen', 'Survived_correct', 'Survived_false']] = vl.data.cpu().numpy(), np.mean(correct['Survived']), np.mean(actual['Survived']==0)


                for col in continuous_cols:
                    #print(np.abs(T0[col].data.cpu().numpy().reshape(-1) - T2sample[col].data.cpu().numpy().reshape(-1)))
                    print(f'% {col} mae: {np.mean(correct[col])}')

                for col in onehot_cols:
                    print(f'% {col} correct: {np.mean(correct[col])} {np.mean(actual[col]==0)}')



                '''
                for col in continuous_cols:
                    mae = np.mean(np.abs(X[col].data.cpu().numpy() - X2[col].data.cpu().numpy()))
                    mse = np.mean(np.square(X[col].data.cpu().numpy() - X2[col].data.cpu().numpy()))
                    print(f'train mae, mse {col} {mae} {mse}')
                    mae = np.mean(np.abs(Xsample[col].data.cpu().numpy() - X2sample[col].data.cpu().numpy()))
                    mse = np.mean(np.square(Xsample[col].data.cpu().numpy() - X2sample[col].data.cpu().numpy()))
                    print(f'val mae, mse {col} {mae} {mse}')

                print({col: tt[0:2].data.cpu().numpy() for col, tt in T2sample.items()})

                if 'Survived' in onehot_cols:
                    print('% survived correct: ', np.mean(T2sample['Survived'].data.cpu().numpy()==Tsample['Survived'].data.cpu().numpy()), np.mean(Tsample['Survived'].data.cpu().numpy()==np.ones_like(Tsample['Survived'].data.cpu().numpy())))

                if 'Sex' in onehot_cols:
                    print('% sex correct: ', np.mean(T2sample['Sex'].data.cpu().numpy()==Tsample['Sex'].data.cpu().numpy()), np.mean(Tsample['Sex'].data.cpu().numpy()==np.ones_like(Tsample['Sex'].data.cpu().numpy())))

                if 'Embarked' in onehot_cols:
                    print('% Embarked correct: ', np.mean(T2sample['Embarked'].data.cpu().numpy()==Tsample['Embarked'].data.cpu().numpy()) )
                    print(onehot_embedding_weights['Embarked'])

                if 'Pclass' in onehot_cols:
                    print('% Pclass correct: ',
                          np.mean(T2sample['Pclass'].data.cpu().numpy() == Tsample['Pclass'].data.cpu().numpy()))

                if 'Cabin' in text_cols:
                    print(embeddings['Cabin'].weight[data.charindex['1']])

                
                if 'Pclass' in onehot_cols:
                    diff = torch.mean(torch.pow(embeddings['Pclass'].weight - reverse_embeddings['Pclass'].weight, 2)).data.cpu().numpy()
                    print(f'diff plcass emb and reverse_emb: {diff}')
                    print(embeddings['Pclass'].weight.data.cpu().numpy())
                '''





                loss = 0.0
                loss0 = 0.0
                loss1 = 0.0
                loss2 = 0.0
                loss3 = 0.0
                #print(T2.data.cpu()[0, 0:30].numpy())

        logger_df.to_csv('logger_'+str(fold)+'.csv', index=False)