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
#import netr as net
import net

MODEL_PATH = 'model.model'
use_cuda = True

to_predict = ['Survived']
onehot_cols = ['Pclass', 'Embarked', 'Sex', 'Survived', 'Surname']  # Survived
discrete_cols = []  # ['Parch']
continuous_cols = ['Fare', 'Age']  #
text_cols = []  # ['Ticket', 'Cabin', 'Name'] # 'Ticket', 'Cabin',   'Cabin']

margin = 1.0
missing_val = -1.0

'''
add a '' entry to charcounts to account for test-time chars that aren't in the training set
'''


# rawdata, charcounts, maxlens, unique_onehotvals = fetch_and_preprocess()
def fetch_and_preprocess():
    data = pd.read_csv('train.csv')
    data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0])
    charcounts = Counter('')
    maxlens = {}
    for c in text_cols:
        data[c] = data[c].apply(lambda x: str(x).lower() if (str(x) != 'nan') else '')
        charcounts += data[c].apply(Counter).sum()

        maxlens[c] = data[c].apply(lambda x: len(x)).max()

    unique_onehotvals = {}
    for c in onehot_cols:
        unique_onehotvals[c] = data[c].unique()

    # data.drop('PassengerId', 1, inplace=True)
    maxlens['Name'] = 20
    maxlens['Ticket'] = 7
    maxlens['Cabin'] = 3

    return data, charcounts, maxlens, unique_onehotvals


class Dataseq(Dataset):
    """Titanic dataset."""

    def __init__(self, data, charcounts, input_dict, unique_onehotvals, maxlens):
        """
        Args:
            data: pandas dataframe
        """
        self.data = data  # assignment is done with a shallow copy in python, so data is not duplicated
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
            self.scalings[k] = {'min': self.data[k].min() - 0.05 * (self.data[k].max() - self.data[k].min()),
                                'max': self.data[k].max() + 0.05 * (self.data[k].max() - self.data[k].min())}
            self.scalings[k]['mean'] = self.data[k].mean()
            self.scalings[k]['std'] = self.data[k].std()

    def __len__(self):
        return self.data.shape[0]

    def __getrawitem__(self, idx):
        # sample = {col: self.data.loc[idx, col].values for dtype in self.input_dict.keys() for col in self.input_dict[dtype]}
        sample = self.data.loc[idx, self.cols].to_dict()
        return sample

    def __getitem__(self, idx):
        sample = self.__getrawitem__(idx)

        # encode onehot variables
        for k in self.input_dict['onehot'].keys():
            sample[k] = self.onehotindex[k][sample[k]]

        # encode text variables
        for k in self.input_dict['text'].keys():
            t = np.array([self.charindex[c] for c in sample[k]], dtype=int)[0:self.maxlens[k]]
            # print(sample[k], t)
            sample[k] = np.zeros(self.maxlens[k], dtype=int)
            sample[k][0:t.shape[0]] = t

        # scale continous variables
        for k in self.input_dict['continuous']:
            sample[k] = (sample[k] - self.scalings[k]['min']) / (self.scalings[k]['max'] - self.scalings[k]['min'])
            # sample[k] = (sample[k] - self.scalings[k]['mean']) / self.scalings[k]['std'] / 2.5

            if np.isnan(sample[k]):
                sample[k] = missing_val  # -np.random.rand() #
        return sample


def discretize(X2d, embeddings, maxlens):
    T2 = {col: X2d[col] for col in X2d.keys()}
    mb_size = X2d[list(X2d.keys())[0]].size(0)

    for col, embedding in embeddings.items():
        n_tokens = embedding.weight.size(0)
        embedding_dim = embedding.weight.size(1)

        adotb = torch.matmul(X2d[col], embedding.weight.permute(1, 0))

        if col in maxlens.keys(): #if col is text data
            adota = torch.matmul(X2d[col].view(-1, maxlens[col], 1, embedding_dim),
                                 X2d[col].view(-1, maxlens[col], embedding_dim, 1))
            adota = adota.view(-1, maxlens[col], 1).repeat(1, 1, n_tokens)
        else: #if col is not text data
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

    # for col in continuous_cols:
    #    T2[col] = -1.0*torch.lt(T2[col], torch.zeros_like(T2[col])).float() + T2[col]*torch.gt(T2[col], torch.zeros_like(T2[col])).float()

    return T2


def are_equal(x0):
    equ = None
    mb_size = x0[list(x0.keys())[0]].size(0)
    for col in x0.keys():
        if not (col in to_predict):
            if len(x0[col].size()) == 1:
                # consider the situation where missing values have been encoded with missing_val = -1
                # t = (torch.eq(x0[col], x1[col])).float() + (torch.lt(x0[col], torch.zeros_like(x0[col]))).float()*(torch.lt(x1[col], torch.zeros_like(x1[col]))).float()
                t0 = x0[col].float() * (torch.gt(x0[col].float(), torch.zeros_like(x0[col].float()))).float() - (
                    torch.lt(x0[col].float(), torch.zeros_like(x0[col].float()))).float()
                t0 = t0.view(x0[col].size() + (1,))
                t1 = t0.permute(0, 1).repeat(1, mb_size, 1)
                t2 = t0.permute(1, 0).repeat(mb_size, 1, 1)
                t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)
            else:  # len(x0[col].size()) == 2:
                t0 = x0[col].view(x0[col].size() + (1,))
                t1 = t0.permute(0, 2, 1).repeat(1, mb_size, 1)
                t2 = t0.permute(2, 0, 1).repeat(mb_size, 1, 1)
                t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)

        if equ is None:
            equ = torch.floor(t)
        else:
            equ *= torch.floor(t)
    return equ




# T, X, X2, mu, mu2, mu2d, mu_tm, logvar_tm, logvar2d, logvar_tm = calc_losses(T, embeddings, enc, dec)
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

    mu, logvar = enc({col: X[col] if not col in to_predict else 0.0 * X[col] for col in X.keys()})
    #t = int(mu.size(0) / 2)
    if mode == 'train':
        mu = enc.reparameterize(mu, logvar)

    mu_tm = torch.zeros_like(mu).unsqueeze(1).repeat((1, 1 + n_targetvals, 1))
    logvar_tm = torch.zeros_like(mu).unsqueeze(1).repeat((1, 1 + n_targetvals, 1))
    mu_tm[:, 0, :] = mu
    logvar_tm[:, 0, :] = logvar

    # encodings for all the possible target embedding values
    for i in range(n_targetvals):
        m, lgv = enc({col: X[col] if not col in to_predict else embeddings[col](i * torch.ones_like(T[col])) for col in
                      X.keys()})
        if mode == 'train':
            #m, lgv = enc(dec(enc.reparameterize(m, lgv)))
            #m, lgv = enc(dec(m))
            m = enc.reparameterize(m, lgv)
            #None

        if mode == 'train':
            use = torch.eq(i * torch.ones_like(T[to_predict[0]]), T[to_predict[0]]).float().view(-1, 1).repeat(1, mu.size(1))
            use[0:int(1*mb_size/2)] = 0.0
            mu = mu + m * use - use * mu
            logvar = logvar + lgv * use - use * logvar

        mu_tm[:, i+1, :] = m
        logvar_tm[:, i + 1, :] = lgv



    X2 = dec(mu)

    '''
    if mode == 'train':
        #mu_tm[:, 0, :] = enc.reparameterize(mu, logvar)
        X2 = dec(mu_tm[:, 0, :])
    else:
        X2 = dec(mu)
    '''



    T2 = {}
    X2d = {col: (1.0 * tt).detach() for col, tt in X2.items()}

    '''
    for col, embedding in embeddings.items():
        if not (col in to_predict):
            None
            T2[col] = reverse_embeddings[col](X2[col])
            X2d[col] = embeddings[col](T2[col].detach()) #+ 0.5*X2d[col]
        else:
            #None
            T2[col] = reverse_embeddings[col](X2[col])
            X2d[col] = embeddings[col](T2[col].detach()) #+ 0.5*X2d[col]
            X2d[col] = torch.cat((0.0*X2d[col][0:int(1*mb_size/2)], X2d[col][int(1*mb_size/2):]), 0)
    '''



    mu2, logvar2 = enc(X2)
    if mode == 'train':
        mu2 = enc.reparameterize(mu2, logvar2)
    mu2 = mu2.view(mb_size, -1)

    mu2d, logvar2d = enc(X2d)
    if mode == 'train':
        mu2d = enc.reparameterize(mu2d, logvar2d)


    mu2d = mu2d.view(mb_size, -1)

    mu = mu.view(mb_size, -1)

    return T, X, X2, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logvar_tm


# enc_loss, enc_loss0, enc_loss1, enc_loss2, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logvar_tm, logloss)
def calc_losses(T, embeddings, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logvar_tm, logloss, lookfordups=True):
    mb_size = mu.size(0)
    latent_dim = mu.size(1)
    n_targetvals = mu_tm.size(1) - 1  # len(mu_tm) - 1

    adotb = torch.matmul(mu, mu2.permute(1, 0))  # batch_size x batch_size
    adota = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
    bdotb = torch.matmul(mu2.view(-1, 1, latent_dim), mu2.view(-1, latent_dim, 1))
    diffsquares = (adota.view(-1, 1).repeat(1, mb_size) + bdotb.view(1, -1).repeat(mb_size, 1) - 2 * adotb) / latent_dim
    tt = np.triu_indices(mb_size, k=1)
    diffsquares = diffsquares[tt]

    adotb2 = torch.matmul(mu, mu.permute(1, 0))  # batch_size x batch_size
    adota2 = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
    bdotb2 = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))
    diffsquares2 = (adota2.view(-1, 1).repeat(1, mb_size) + bdotb2.view(1, -1).repeat(mb_size,
                                                                                      1) - 2 * adotb2) / latent_dim
    diffsquares2 = diffsquares2[tt]

    if lookfordups:
        are_same = are_equal(T)
        are_same = are_same[tt]
    else:
        are_same = torch.zeros_like(diffsquares)

    # print('shapes', are_same.size())
    # print('fraction same', torch.mean(are_same))
    enc_loss0 = 0.25 * logloss(diffsquares, are_same, weights=1.0 - are_same)
    #enc_loss0 += 0.125 * logloss(diffsquares2, are_same, weights=1.0 - are_same)

    # enc_loss0 = logloss(torch.mean(torch.pow(mu[::2]-mu[1::2], 2), 1), are_same, weights=1-are_same)
    enc_loss1 = 0.5 * logloss(torch.mean(torch.pow(mu - mu2, 2), 1), torch.ones(mb_size).cuda())
    # enc_loss1 = 0.5 * logloss(torch.mean(torch.pow(mu_tm[:, 0, :] - mu2, 2), 1), torch.ones(mb_size).cuda())
    enc_loss2 = 0.25 * logloss(torch.mean(torch.pow(mu - mu2d, 2), 1), torch.zeros(mb_size).cuda())
    # enc_loss3 = 0.5 * logloss(torch.mean(torch.pow(mu - mu_tm, 2), 1), torch.ones(mb_size).cuda())
    # enc_loss3 += 0.5 * logloss(torch.mean(torch.pow(mu - mu_false, 2), 1), torch.zeros(mb_size).cuda())

    enc_loss3 = 0.0


    for i in range(n_targetvals):
        if use_cuda:
            target = torch.eq(i * torch.ones_like(T[to_predict[0]]), T[to_predict[0]]).float().cuda()
        else:
            target = torch.eq(i * torch.ones_like(T[to_predict[0]]), T[to_predict[0]]).float()

        factor = 0.5 * ((1 - target) / (n_targetvals - 1) + target)
        enc_loss3 += torch.mean(
            factor * logloss(torch.mean(torch.pow(mu_tm[:, 0] - mu_tm[:, i + 1], 2), 1), target,
                             do_batch_mean=False))

    '''
    #bdot is the square of the difference between the latent vectors for each of the possible (non-missing)
    #values of the target variable
    bdot = torch.bmm(mu_tm[:, 1:, :], mu_tm[:, 1:, :].permute(0, 2, 1))  # batch_size x n_target_vals x n_target_vals
    diag = torch.diagonal(bdot, offset=0, dim1=1, dim2=2)
    bdot = -2.0*bdot
    bdot += diag.view(-1, n_targetvals, 1).repeat(1, 1, n_targetvals)
    bdot += diag.view(-1, 1, n_targetvals).repeat(1, n_targetvals, 1)
    tt = np.triu_indices(n_targetvals, k=1)
    uptri_size = len(tt[0])

    #calculate the upper-triangular indices batchwise
    tt = (np.arange(mb_size).repeat(uptri_size ),) + (np.tile(tt[0], mb_size),) + (np.tile(tt[1], mb_size),)
    bdot = bdot[tt]
    #bdot = bdot[tt].view(uptri_size, mb_size).permute(1,0)
    #print(bdot.size())
    ltemp = 0.25*torch.mean(logloss(bdot/latent_dim, torch.zeros(uptri_size*mb_size).cuda()))
    #print(ltemp)
    enc_loss3 += ltemp
    '''


    enc_loss = 1.0*(enc_loss0 + enc_loss1 + enc_loss2) + 2.0 * enc_loss3

    enc_loss += 2.0 / 64.0 * torch.mean(torch.pow(mu, 2))
    enc_loss += 2.0 / 64.0 * torch.mean(torch.pow(mu2, 2))
    enc_loss += 4.0 / 256.0 * torch.mean(torch.exp(logvar) - logvar) #1.0/64
    enc_loss += 4.0 / 256.0 * torch.mean(torch.exp(logvar2) - logvar2)
    enc_loss += 4.0 / 256.0 * torch.mean(torch.exp(logvar2d) - logvar2d)
    enc_loss += 4.0 / 256.0 * torch.mean(torch.exp(logvar_tm[:, 1:]) - logvar_tm[:, 1:])

    for col, emb in embeddings.items():
        enc_loss += 4.0/64.0 * torch.sum(torch.mean(torch.pow(emb.weight, 2), 1)) / np.sqrt(emb.weight.size(0))

    return enc_loss, enc_loss0, enc_loss1, enc_loss2, enc_loss3

#mu, mu2, target_arr = do_train(rawdata, charcounts, maxlens, unique_onehotvals)
def do_train(rawdata, charcounts, maxlens, unique_onehotvals):
    n_batches = 2800
    mb_size = 128
    lr = 2.0e-4
    momentum = 0.5
    cnt = 0
    latent_dim = 32  # 24#
    recurrent_hidden_size = 24

    epoch_len = 8
    max_veclen = 0.0
    patience = 12 * epoch_len
    patience_duration = 0

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
    #data_idx = np.arange(data.__len__())
    data_idx = np.arange(rawdata.shape[0])
    np.random.shuffle(data_idx)
    n_folds = 6
    fold_size = 1.0 * rawdata.shape[0] / n_folds #data.__len__() / n_folds
    folds = [data_idx[int(i * fold_size):int((i + 1) * fold_size)] for i in range(6)]

    fold_groups = {}
    fold_groups[0] = {'train': [0, 1, 2, 3], 'val': [4]}
    fold_groups[1] = {'train': [1, 2, 3, 4], 'val': [0]}
    fold_groups[2] = {'train': [0, 2, 3, 4], 'val': [1]}
    fold_groups[3] = {'train': [0, 1, 3, 4], 'val': [2]}
    fold_groups[4] = {'train': [0, 1, 2, 4], 'val': [3]}

    for fold in range(1):

        train_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['train']])))
        val_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['val']])))

        #data = Dataseq(rawdata, charcounts, input_dict, unique_onehotvals, maxlens)
        train = Subset(data, train_idx)
        val = Subset(data, val_idx)

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_iter = torch.utils.data.DataLoader(train, batch_size=int(mb_size / 1), shuffle=True, **kwargs)
        train_iter_unshuffled = torch.utils.data.DataLoader(train, batch_size=mb_size, shuffle=False, **kwargs)
        val_iter = torch.utils.data.DataLoader(val, batch_size=mb_size, shuffle=False, **kwargs)

        embeddings = {}
        reverse_embeddings = {}
        onehot_embedding_weights = {}
        for k in onehot_cols:
            dim = input_dict['onehot'][k]
            onehot_embedding_weights[k] = net.get_embedding_weight(len(unique_onehotvals[k]), dim, use_cuda=use_cuda)
            embeddings[k] = nn.Embedding(len(unique_onehotvals[k]), dim, _weight=onehot_embedding_weights[k])
            reverse_embeddings[k] = net.EmbeddingToIndex(len(unique_onehotvals[k]), dim,
                                                         _weight=onehot_embedding_weights[k])

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

        logloss = contrastive.GaussianOverlap()

        solver = optim.RMSprop(
            [p for em in embeddings.values() for p in em.parameters()] + [p for p in enc.parameters()] + [p for p in
                                                                                                          dec.parameters()],
            lr=lr, momentum=momentum)

        print('starting training')
        loss = 0.0
        loss0 = 0.0
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0

        logger_df = pd.DataFrame(
            columns=['iter', 'train_loss', 'train_veclen', 'val_veclen', 'val_loss', 'val_acc']+[t+'_correct' for t in to_predict]+[t+'_false' for t in to_predict])


        for it in range(n_batches):
            T = next(iter(train_iter))
            # for col, value in T.items():
            #    T[col] = torch.cat((value, value, value, value), 0)

            T, X, X2, mu, logvar, mu2, mu2d, mu_tm, logvar2, logvar2d, logvar_tm = calc_mus(T, embeddings, reverse_embeddings, enc,
                                                                                 dec)
            enc_loss, enc_loss0, enc_loss1, enc_loss2, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d,
                                                                               mu_tm, logvar2, logvar2d, logvar_tm, logloss)

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


            if (it+1) % epoch_len == 0:
                print(it, loss / epoch_len, loss0 / epoch_len, loss1 / epoch_len, loss2 / epoch_len, loss3 / epoch_len,
                      veclen.data.cpu().numpy())  # enc_loss.data.cpu().numpy(),

                n_targetvals = embeddings[to_predict[0]].weight.size(0)
                if use_cuda:
                    mu = torch.zeros(len(train), mu.size(1)).cuda()
                    logvar = torch.zeros(len(train), mu.size(1)).cuda()
                    mu2 = torch.zeros(len(train), mu.size(1)).cuda()
                    mu2d = torch.zeros(len(train), mu.size(1)).cuda()
                    mu_tm = torch.zeros((len(train),) + mu_tm.size()[1:]).cuda()
                    logvar2 = torch.zeros(len(train), mu.size(1)).cuda()
                    logvar2d = torch.zeros(len(train), mu.size(1)).cuda()
                    logvar_tm = torch.zeros(len(train), 1+n_targetvals, mu.size(1)).cuda()
                    train_loss = torch.zeros(len(train)).cuda()
                else:
                    mu = torch.zeros(len(train), mu.size(1))
                    logvar = torch.zeros(len(train), mu.size(1))
                    mu2 = torch.zeros(len(train), mu.size(1))
                    mu2d = torch.zeros(len(train), mu.size(1))
                    mu_tm = torch.zeros((len(train),) + mu_tm.size()[1:])
                    logvar2 = torch.zeros(len(train), mu.size(1))
                    logvar2d = torch.zeros(len(train), mu.size(1))
                    logvar_tm = torch.zeros(len(train), 1 + n_targetvals, mu.size(1))
                    train_loss = torch.zeros(len(train))

                s = 0
                for T0 in train_iter_unshuffled:
                    e = s + T0[to_predict[0]].size(0)
                    if s == 0:
                        T = {col: torch.zeros((len(train),) + value.size()[1:], dtype=value.dtype) for col, value in
                             T0.items()}

                    T0, Xsample, _, mu[s:e], logvar[s:e], mu2[s:e], mu2d[s:e], mu_tm[s:e], logvar2[s:e], logvar2d[
                                                                                                          s:e], logvar_tm[s:e] = calc_mus(
                        T0, embeddings, reverse_embeddings, enc, dec, mode='val')
                    for col, value in T0.items():
                        T[col][s:e] = T0[col]

                    n_targetvals = embeddings[to_predict[0]].weight.size(0)
                    mu_tm[s:e, 0, :] = 1.0*mu[s:e]
                    p = torch.zeros((e-s), n_targetvals).cuda()

                    # encodings for all the possible target embedding values
                    for i in range(n_targetvals):
                        if use_cuda:
                            t = {col: Xsample[col] if not col in to_predict else embeddings[col](i * torch.ones_like(T0[col]).cuda()) for
                                 col in Xsample.keys()}
                            mu_tm[s:e, i + 1, :], _ = enc(t)
                        else:
                            mu_tm[s:e, i + 1, :], _ = enc(
                                {col: Xsample[col] if not col in to_predict else embeddings[col](i * torch.ones_like(T0[col])) for
                                 col in Xsample.keys()})
                        diffsquares = torch.sqrt(torch.mean(torch.pow(mu_tm[s:e, 0, :] - mu_tm[s:e, i + 1, :], 2), 1))
                        p[:, i] = 1.0-torch.abs(torch.erf(diffsquares / 2.0))



                    labels = T0[to_predict[0]]
                    target = torch.zeros(e-s, n_targetvals)
                    target[torch.arange(e-s), labels] = 1
                    target = target.cuda()

                    #print(target[0:5])
                    #print(p[0:5])
                    p = p / torch.sum(p, 1).view(-1, 1).repeat(1, n_targetvals)

                    train_loss[s:e] += -torch.mean(target * torch.log(torch.clamp(p, 1e-8, 1.0)) + (1 - target) * torch.log(torch.clamp(1 - p, 1e-8, 1.0)), 1)

                    s = e

                enc_loss, enc_loss0, enc_loss1, enc_loss3, enc_loss3 = calc_losses(T, embeddings, mu, logvar, mu2, mu2d,
                                                                                   mu_tm, logvar2, logvar2d, logvar_tm, logloss,
                                                                                   lookfordups=False)
                vl = torch.mean(torch.pow(mu, 2))

                print(f'train enc loss {enc_loss}')
                print(f'train veclen {vl}')
                print(f'mean train logvar {torch.mean(logvar)}')
                print(f'mean train_loss {torch.mean(train_loss)}')
                logger_df.loc[int(it / epoch_len), ['iter', 'train_loss', 'train_veclen', 'train_loss']] = [it,
                                                                                              enc_loss.data.cpu().numpy(),
                                                                                              vl.data.cpu().numpy(),
                                                                                              torch.mean(train_loss).data.cpu().numpy()]

                if use_cuda:
                    mu = torch.zeros(len(val), mu.size(1)).cuda()
                    logvar = torch.zeros(len(val), mu.size(1)).cuda()
                    mu2 = torch.zeros(len(val), mu.size(1)).cuda()
                    mu2d = torch.zeros(len(val), mu.size(1)).cuda()
                    n_targetvals = embeddings[to_predict[0]].weight.size(0)
                    mu_tm = torch.zeros(len(val), 1 + n_targetvals, mu.size(1)).cuda()
                    val_loss = torch.zeros(len(val)).cuda()
                    val_accuracy = torch.zeros(len(val)).cuda()
                    target_arr = torch.zeros(len(val)).cuda()
                else:
                    mu = torch.zeros(len(val), mu.size(1))
                    logvar = torch.zeros(len(val), mu.size(1))
                    mu2 = torch.zeros(len(val), mu.size(1))
                    mu2d = torch.zeros(len(val), mu.size(1))
                    n_targetvals = embeddings[to_predict[0]].weight.size(0)
                    mu_tm = torch.zeros(len(val), 1 + n_targetvals, mu.size(1))
                    val_loss = torch.zeros(len(val))
                    val_accuracy = torch.zeros(len(val))
                    target_arr = torch.zeros(len(val)).cuda()

                s = 0
                targets = {}
                for T0 in val_iter:
                    e = s + T0[to_predict[0]].size(0)
                    target_arr[s:e] = T0[to_predict[0]]
                    #print(s, e)

                    if s == 0:
                        correct = {col: np.zeros((len(val),) + v.size()[1:]) for col, v in T0.items()}
                        actual = {col: np.zeros((len(val),) + v.size()[1:]) for col, v in T0.items()}

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

                        if col in to_predict:
                            targets[col] = tt
                            Xsample[col] = 0.0 * Xsample[col]


                    mu[s:e], logvar[s:e] = enc(Xsample)

                    X2sample = dec(mu[s:e])
                    T2sample = discretize(X2sample, embeddings, maxlens)

                    mu2[s:e], _ = enc(X2sample)

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
                            correct[col][s:e] = np.abs(
                                T2sample[col].data.cpu().numpy() == targets[col].data.cpu().numpy())
                            actual[col][s:e] = targets[col].data.cpu().numpy().reshape(-1)
                        else:
                            correct[col][s:e] = np.abs(T2sample[col].data.cpu().numpy() == T0[col].data.cpu().numpy())
                            actual[col][s:e] = T0[col].data.cpu().numpy().reshape(-1)

                    mu2d[s:e], _ = enc(X2dsample)



                    '''
                    calculate target predictions for validation data
                    '''

                    n_targetvals = embeddings[to_predict[0]].weight.size(0)
                    mu_tm[s:e, 0, :] = 1.0*mu[s:e]
                    if use_cuda:
                        p = torch.zeros((e-s), n_targetvals).cuda()
                    else:
                        p = torch.zeros((e - s), n_targetvals)

                    # generate encodings for all the possible target embedding values
                    for i in range(n_targetvals):
                        if use_cuda:
                            t = {col: Xsample[col] if not col in to_predict else embeddings[col](i * torch.ones_like(T0[col]).cuda()) for
                                 col in Xsample.keys()}
                            mu_tm[s:e, i + 1, :], _ = enc(t)
                        else:
                            mu_tm[s:e, i + 1, :], _ = enc(
                                {col: Xsample[col] if not col in to_predict else embeddings[col](i * torch.ones_like(T0[col])) for
                                 col in Xsample.keys()})
                        diffsquares = torch.sqrt(torch.mean(torch.pow(mu_tm[s:e, 0, :] - mu_tm[s:e, i + 1, :], 2), 1))
                        p[:, i] = 1.0-torch.abs(torch.erf(diffsquares / 2.0))

                        #print(mu_tm[s:s+5, i + 1, 0:5])
                        print(diffsquares[0:5])


                    labels = T0[to_predict[0]]
                    target = torch.zeros(e-s, n_targetvals)
                    target[torch.arange(e-s), labels] = 1
                    if use_cuda:
                        target = target.cuda()
                        labels = labels.cuda()

                    p = p / torch.sum(p, 1).view(-1, 1).repeat(1, n_targetvals)
                    val_accuracy[s:e] = torch.eq(labels, torch.max(p, 1)[1]).float()

                    val_loss[s:e] += -torch.mean(target * torch.log(torch.clamp(p, 1e-8, 1.0)) + (1 - target) * torch.log(torch.clamp(1 - p, 1e-8, 1.0)), 1)

                    s = e

                vl = torch.mean(torch.pow(mu, 2))

                print(f'val veclen {vl}')
                print(f'mean es logvar {torch.mean(logvar)}')
                print(f'mean val_loss {torch.mean(val_loss)}')
                print(f'mean val_accuracy {torch.mean(val_accuracy)}')

                logger_df.loc[int(it / epoch_len), ['val_veclen', 'val_loss', 'val_acc']] = vl.data.cpu().numpy(), torch.mean(val_loss).data.cpu().numpy(), torch.mean(val_accuracy).data.cpu().numpy()
                for target_col in to_predict:
                    logger_df.loc[int(it / epoch_len), [target_col+'_correct',
                                                        target_col+'_false']] = np.mean(
                        correct[target_col]), np.mean(actual[target_col] == 0)

                for col in continuous_cols:
                    # print(np.abs(T0[col].data.cpu().numpy().reshape(-1) - T2sample[col].data.cpu().numpy().reshape(-1)))
                    print(f'% {col} mae: {np.mean(correct[col])}')

                for col in onehot_cols:
                    print(f'% {col} correct: {np.mean(correct[col])} {np.mean(actual[col]==0)}')


                loss = 0.0
                loss0 = 0.0
                loss1 = 0.0
                loss2 = 0.0
                loss3 = 0.0
                # print(T2.data.cpu()[0, 0:30].numpy())

        logger_df.to_csv('logger_' + str(fold) + '.csv', index=False)
        #torch.save({'enc': enc, 'dec': dec, 'emb': embeddings}, MODEL_PATH)
        return mu, mu2, target_arr

#logger_df =  pd.read_csv('logger_0.csv')
def plot_stuff(logger_df):
    plt.plot(logger_df.iter, logger_df.train_loss, '.', label='train loss')
    plt.plot(logger_df.iter, logger_df.train_veclen, '.', label='train vector length')
    plt.plot(logger_df.iter, logger_df.val_loss, '.', label='val loss')
    plt.plot(logger_df.iter, logger_df.val_acc, '.', label='val acc')
    plt.plot(logger_df.iter, logger_df.Survived_correct, '.', label='survived correct')
    plt.xlabel('iterations')
    plt.legend()


def mu_overlaps(mu, target_arr):
    n = mu.size(0)
    ld = mu.size(1)
    bdot = torch.matmul(mu, mu.permute(1,0))
    diag = torch.diagonal(bdot, offset=0, dim1=0, dim2=1)
    bdot = -2.0*bdot
    bdot += diag.view(n, 1).repeat(1, n)
    bdot += diag.view(1, n).repeat(n,  1)
    bdot /= ld
    tt = np.triu_indices(n, k=1)

    targets_same = torch.eq(target_arr.view(n, 1).repeat(1, n), target_arr.view(1, n).repeat(n, 1))
    targets_diff = 1.0 - targets_same
    targets_0 = targets_same * torch.eq(target_arr.view(n, 1).repeat(1, n), torch.zeros(n, n).cuda())
    targets_1 = targets_same * torch.eq(target_arr.view(n, 1).repeat(1, n), torch.ones(n, n).cuda())

    dists_0 = bdot[tt][(targets_0 * targets_same)[tt]].data.cpu().numpy()
    dists_1 = bdot[tt][(targets_1 * targets_same)[tt]].data.cpu().numpy()
    dists_diff = bdot[tt][(targets_diff)[tt]].data.cpu().numpy()

    delta = 0.04*np.median(dists_diff)
    bins = 100
    counts_diff = np.histogram(dists_diff, bins=delta*np.arange(bins))[0]
    counts_0 = np.histogram(dists_0, bins=delta * np.arange(bins))[0]
    counts_1 = np.histogram(dists_1, bins=delta * np.arange(bins))[0]

    q = target_arr.mean().data.cpu().numpy()
    #plt.plot(delta * np.arange(bins-1), counts_diff/q/(1-q)/2.0, 'k.-', label='targets different')
    #plt.plot(delta * np.arange(bins-1), counts_0/(1-q)**2, 'r.-', label='target=died')
    #plt.plot(delta * np.arange(bins-1), counts_1/q**2, 'g.-', label='target=survived')
    plt.plot(delta * np.arange(bins-1), counts_diff/len(dists_diff), 'k.-', label='targets different')
    plt.plot(delta * np.arange(bins-1), counts_0/len(dists_0), 'r.-', label='target=died')
    plt.plot(delta * np.arange(bins-1), counts_1/len(dists_1), 'g.-', label='target=survived')
    plt.legend()
    plt.xlabel('distance between embeddings')
    plt.ylabel('scaled hist(distance between embeddings)')



