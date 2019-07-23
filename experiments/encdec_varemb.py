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
continuous_cols = ['Fare', 'Age']
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
    maxlens['Name'] = 7
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
                sample[k] = -np.random.rand() #missing_val #
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


def are_equal(x0, x1):
    equ = None
    for col in x0.keys():
        if len(x0[col].size()) == 1:
            t = (torch.eq(x0[col], x1[col])).float() + (torch.lt(x0[col], torch.zeros_like(x0[col]))).float()*(torch.lt(x1[col], torch.zeros_like(x1[col]))).float()
        if len(x0[col].size()) == 2:
            t = torch.mean((torch.eq(x0[col], x1[col])).float(), -1)
        if len(x0[col].size()) == 3:
            t = torch.mean(torch.mean((torch.eq(x0[col], x1[col])).float(), -1), -1)
        if equ is None:
            equ = torch.floor(t)
        else:
            equ *= torch.floor(t)
    return equ.reshape(-1)

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
        if len(x0[col].size()) == 2:
            t0 = x0[col].view(x0[col].size()+(1,))
            t1 = t0.permute(0, 2, 1).repeat(1, mb_size, 1)
            t2 = t0.permute(2, 0, 1).repeat(mb_size, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)

        if len(x0[col].size()) == 3:
            t0 = x0[col].view(x0[col].size()+(1,))
            t1 = t0.permute(0, 3, 1, 2).repeat(1, mb_size, 1, 1)
            t2 = t0.permute(3, 0, 1, 2).repeat(mb_size, 1, 1, 1)
            t = torch.mean(torch.eq(t1, t2).float().view(mb_size, mb_size, -1), -1)
        if equ is None:
            equ = torch.floor(t)
        else:
            equ *= torch.floor(t)
    return equ


def do_train(rawdata, charcounts, maxlens, unique_onehotvals):
    mb_size = 128
    lr = 2.0e-4
    cnt = 0
    latent_dim = 64
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
        train_iter = torch.utils.data.DataLoader(train, batch_size=mb_size, shuffle=True, **kwargs)
        es_iter = torch.utils.data.DataLoader(es, batch_size=mb_size, shuffle=True, **kwargs)
        val_iter = torch.utils.data.DataLoader(val, batch_size=mb_size, shuffle=True, **kwargs)

        embeddings = {}
        reverse_embeddings = {}
        onehot_embedding_weights = {}
        onehot_embedding_spread = {}
        for k in onehot_cols:
            dim = input_dict['onehot'][k]
            onehot_embedding_weights[k] = net.get_embedding_weight(len(unique_onehotvals[k]), dim, use_cuda=use_cuda)
            onehot_embedding_spread[k] = net.get_embedding_weight(len(unique_onehotvals[k]), dim, use_cuda=use_cuda)
            #embeddings[k] = nn.Embedding(len(unique_onehotvals[k]), dim, _weight=onehot_embedding_weights[k])
            embeddings[k] = net.VariationalEmbedding(len(unique_onehotvals[k]), dim, _weight=onehot_embedding_weights[k], _spread=onehot_embedding_spread[k])
            reverse_embeddings[k] = net.EmbeddingToIndex(len(unique_onehotvals[k]), dim, _weight=onehot_embedding_weights[k])

        if text_dim > 0:
            text_embedding_weights = net.get_embedding_weight(len(charcounts) + 1, text_dim, use_cuda=use_cuda)
            # text_embedding = nn.Embedding(len(charcounts) + 1, text_dim, _weight=text_embedding_weights)
            text_embedding_spread = net.get_embedding_weight(len(charcounts) + 1, text_dim, use_cuda=use_cuda)
            text_embedding = net.VariationalEmbedding(len(charcounts) + 1, text_dim, _weight=text_embedding_weights, _spread=text_embedding_spread)
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
        solver = optim.Adam(
            [p for em in embeddings.values() for p in em.parameters()] + [p for p in enc.parameters()] + [p for p in
                                                                                                          dec.parameters()],
            lr=lr)

        Tsample = next(es_iter.__iter__())
        if use_cuda:
            Tsample = {col: Variable(tt[0:128]).cuda() for col, tt in Tsample.items()}
        else:
            Tsample = {col: Variable(tt[0:128]) for col, tt in Tsample.items()}

        print({col: tt[0] for col, tt in Tsample.items()})

        print('starting training')
        loss = 0.0
        loss0 = 0.0
        loss1 = 0.0
        loss2 = 0.0

        for it in range(1000000):
            # X = Variable(torch.tensor(np.array([[1,2,4], [4,1,9]]))).cuda()
            batch_idx, T = next(enumerate(train_iter))
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

            mu = enc(X)
            X2 = dec(mu)

            T2 = {}
            X2d = {col: (1.0 * tt).detach() for col, tt in X2.items()}


            for col, embedding in embeddings.items():
                T2[col] = reverse_embeddings[col](X2[col])
                X2[col] = embeddings[col](T2[col])
                X2d[col] = embeddings[col](T2[col].detach())






            '''
            X2d = {col: (1.0*tt).detach() for col, tt in X2.items()}
            T2 = discretize(X2d, embeddings, maxlens)
            for col, embedding in embeddings.items():
                X2d[col] = embeddings[col](T2[col].detach())
            '''

            '''
            T2 = discretize(X2, embeddings, maxlens)
            X2d = {col: (1.0*tt).detach() for col, tt in X2.items()}

            for col, embedding in embeddings.items():
                X2[col] = embeddings[col](T2[col]) #+0.05 X2[col]
                X2d[col] = embeddings[col](T2[col].detach())
            '''


            mu2 = enc(X2)
            mu2 = mu2.view(mb_size, -1)

            mu2d = enc(X2d)

            mu2d = mu2d.view(mb_size, -1)


            mu = mu.view(mb_size, -1)


            adotb = torch.matmul(mu, mu.permute(1, 0))  # batch_size x batch_size
            adota = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
            diffsquares = (adota.view(-1, 1).repeat(1, mb_size) + adota.view(1, -1).repeat(mb_size, 1) - 2 * adotb) / latent_dim
            tt = np.triu_indices(mb_size, k=1)
            diffsquares = diffsquares[tt]

            adotb = torch.matmul(mu, mu2.permute(1, 0))  # batch_size x batch_size
            adota = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
            bdotb = torch.matmul(mu2.view(-1, 1, latent_dim), mu2.view(-1, latent_dim, 1))  # batch_size x 1 x 1
            diffsquares2 = (adota.view(-1, 1).repeat(1, mb_size) + bdotb.view(1, -1).repeat(mb_size, 1) - 2 * adotb) / latent_dim
            tt = np.triu_indices(mb_size, k=1)
            diffsquares2 = diffsquares2[tt]

            #are_same = are_equal({col: x[::2] for col, x in T.items()}, {col: x[1::2] for col, x in T.items()})
            are_same = are_equal2(X)
            are_same = are_same[tt]

            #print('shapes', are_same.size())

            #print('fraction same', torch.mean(are_same))
            enc_loss0 = 0.5*logloss(diffsquares, are_same, weights=1.0-are_same)
            enc_loss0 += 0.5*logloss(diffsquares2, are_same, weights=1.0 - are_same)
            #enc_loss0 = logloss(torch.mean(torch.pow(mu[::2]-mu[1::2], 2), 1), are_same, weights=1-are_same)
            enc_loss1 = 2.0*logloss(torch.mean(torch.pow(mu-mu2,2), 1), torch.ones(mb_size).cuda())
            enc_loss2 = 0.0*logloss(torch.mean(torch.pow(mu-mu2d, 2), 1), torch.zeros(mb_size).cuda())

            enc_loss = enc_loss0 + enc_loss1 + enc_loss2
            enc_loss += 0.025*torch.mean(torch.pow(mu, 2))
            enc_loss += 0.025 * torch.mean(torch.pow(mu2, 2))
            for k in onehot_cols:
                enc_loss += 0.025*torch.mean(torch.pow(onehot_embedding_weights[k], 2))
                enc_loss += 0.025 * torch.mean(torch.exp(onehot_embedding_spread[k]) - onehot_embedding_spread[k] )





            '''
            adotb = torch.matmul(mu, mu.permute(1, 0))  # batch_size x batch_size
            adota = torch.matmul(mu.view(-1, 1, latent_dim), mu.view(-1, latent_dim, 1))  # batch_size x 1 x 1
            diffsquares = (adota.view(-1, 1).repeat(1, mb_size) + adota.view(1, -1).repeat(mb_size, 1) - 2 * adotb) / latent_dim

            # did I fuck up something here? diffsquares can apparently be less than 0....
            mdist = torch.sqrt(torch.clamp(torch.triu(diffsquares, diagonal=1),  min=0.0))
            mdist = torch.clamp(margin - mdist, min=0.0)
            number_of_pairs = mb_size * (mb_size - 1) / 2

            enc_loss = 0.5 * torch.sum(torch.triu(torch.pow(mdist, 2), diagonal=1)) / number_of_pairs

            target = torch.ones(mu.size(0), 1)
            if use_cuda:
                target.cuda()
            enc_loss += contrastivec(mu, mu2, target.cuda())

            target = torch.zeros(mu.size(0), 1)
            if use_cuda:
                target.cuda()
            enc_loss += 2.0 * contrastivec(mu, mu2d, target.cuda())
            '''


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
            veclen = torch.mean(torch.pow(mu, 2))
            if it % epoch_len == 0:
                print(it, loss/epoch_len, loss0/epoch_len, loss1/epoch_len, loss2/epoch_len, veclen.data.cpu().numpy()) #enc_loss.data.cpu().numpy(),

                Xsample = {}
                for col, tt in Tsample.items():
                    if col in embeddings.keys():
                        Xsample[col] = embeddings[col](tt)
                    else:
                        Xsample[col] = tt.float()

                mu = enc(Xsample)
                X2sample = dec(mu)
                X2sampled = {col: tt.detach() for col, tt in X2sample.items()}
                T2sample = discretize(X2sample, embeddings, maxlens)

                mu2 = enc(X2sample)
                mu2d = enc(X2sampled)


                if 'Fare' in continuous_cols and 'Age' in continuous_cols:
                    print([np.mean(np.abs(Xsample[col].data.cpu().numpy()-X2sample[col].data.cpu().numpy())) for col in ['Fare', 'Age']])

                print({col: tt[0:2].data.cpu().numpy() for col, tt in T2sample.items()})

                if 'Survived' in onehot_cols:
                    print('% survived correct: ', np.mean(T2sample['Survived'].data.cpu().numpy()==Tsample['Survived'].data.cpu().numpy()), np.mean(Tsample['Survived'].data.cpu().numpy()==np.ones_like(Tsample['Survived'].data.cpu().numpy())))

                if 'Sex' in onehot_cols:
                    print('% sex correct: ', np.mean(T2sample['Sex'].data.cpu().numpy()==Tsample['Sex'].data.cpu().numpy()), np.mean(Tsample['Sex'].data.cpu().numpy()==np.ones_like(Tsample['Sex'].data.cpu().numpy())))

                if 'Embarked' in onehot_cols:
                    print('% Embarked correct: ', np.mean(T2sample['Embarked'].data.cpu().numpy()==Tsample['Embarked'].data.cpu().numpy()) )
                    print(embeddings['Embarked'].weight[0])

                if 'Pclass' in onehot_cols:
                    print('% Pclass correct: ',
                          np.mean(T2sample['Pclass'].data.cpu().numpy() == Tsample['Pclass'].data.cpu().numpy()))
                    print(embeddings['Pclass'].weight[0])

                if 'Cabin' in text_cols:
                    print(embeddings['Cabin'].weight[data.charindex['1']])



                are_same = are_equal({col: x[::2] for col, x in Tsample.items()}, {col: x[1::2] for col, x in Tsample.items()})
                # print('f same ', torch.mean(torch.mean(are_same, 1)))
                # enc_loss = contrastivec(mu2[::2], mu2[1::2], torch.zeros(int(mb_size / 2)).cuda())
                #es_loss = contrastivec(mu[::2], mu[1::2], are_same)
                # enc_loss += 0.25*contrastivec(mu2[::2], mu2[1::2], are_same)
                # enc_loss += 0.5 * contrastivec(mu[::2], mu2[1::2], are_same)
                es_loss = 1.0 * logloss(torch.mean(torch.pow(mu-mu2, 2), 1), torch.ones(mu.size(0)).cuda())
                #es_loss += 2.0 * contrastivec(mu, mu2d, torch.zeros(mu.size(0)).cuda())

                #print('mean mu ', torch.mean(torch.pow(mu, 2)))
                print('es loss ', es_loss)

                loss = 0.0
                loss0 = 0.0
                loss1 = 0.0
                loss2 = 0.0
                #print(T2.data.cpu()[0, 0:30].numpy())