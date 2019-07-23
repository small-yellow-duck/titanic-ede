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
text_cols = ['Name','Cabin'] # 'Ticket', 'Cabin',   'Cabin']

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
    maxlens['Name'] = 3
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
                sample[k] = -np.random.rand() #missing_val
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
            t = (torch.eq(x0[col], x1[col])).float()
        if len(x0[col].size()) == 2:
            t = torch.mean((torch.eq(x0[col], x1[col])).float(), -1)
        if len(x0[col].size()) == 3:
            t = torch.mean(torch.mean((torch.eq(x0[col], x1[col])).float(), -1), -1)
        if equ is None:
            equ = torch.floor(t)
        else:
            equ *= torch.floor(t)
    return equ.reshape(-1, 1)

def train(rawdata, charcounts, maxlens, unique_onehotvals):
    mb_size = 256
    lr = 1.0e-4
    cnt = 0
    latent_dim = 16
    recurrent_hidden_size = 8

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
        for k in onehot_cols:
            dim = input_dict['onehot'][k]
            embeddings[k] = nn.Embedding(len(unique_onehotvals[k]), dim, max_norm=1.0)

        if text_dim > 0:
            text_embedding = nn.Embedding(len(charcounts)+1, text_dim, max_norm=1.0)
            for k in text_cols:
                embeddings[k] = text_embedding

        enc = net.Encoder(input_dict, dim=latent_dim, recurrent_hidden_size=recurrent_hidden_size, sigmoidout=False)
        dec = net.Decoder(input_dict, maxlens, dim=latent_dim, recurrent_hidden_size=recurrent_hidden_size)
        disc = net.Encoder(input_dict, dim=latent_dim, recurrent_hidden_size=recurrent_hidden_size, sigmoidout=False)

        if use_cuda:
            #embeddings_ed = {k: embeddings[k].cuda() for k in embeddings.keys()}
            #embeddings_c = {k: embeddings[k].cuda() for k in embeddings.keys()}
            embeddings = {k: embeddings[k].cuda() for k in embeddings.keys()}
            enc.cuda()
            dec.cuda()
            disc.cuda()

        print(enc.parameters)
        #print(dec.parameters)
        #print(disc.parameters)

        contrastivec = contrastive.ContrastiveLoss(margin=margin)
        logloss = nn.BCELoss()


        #solver = optim.RMSprop([p for em in embeddings.values() for p in em.parameters()] +  [p for p in enc.parameters()] + [p for p in dec.parameters()], lr=lr)
        solver_encdec = optim.Adam(
            [p for em in embeddings.values() for p in em.parameters()] + [p for p in enc.parameters()] + [p for p in
                                                                                                          dec.parameters()],
            lr=lr)

        solver_disc = optim.Adam(
            [p for p in disc.parameters()],
            lr=lr)


        Tsample = next(es_iter.__iter__())
        if use_cuda:
            Tsample = {col: Variable(tt[0:128]).cuda() for col, tt in Tsample.items()}
        else:
            Tsample = {col: Variable(tt[0:128]) for col, tt in Tsample.items()}

        print({col: tt[0] for col, tt in Tsample.items()})

        print('starting training')
        total_loss = 0.0
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


            T2 = discretize(X2, embeddings, maxlens)
            for col, embedding in embeddings.items():
                X2[col] = embeddings[col](T2[col])


            mu2 = enc(X2)
            pred = disc(X2)

            '''
            if use_cuda:
                target = torch.ones(mb_size).cuda()
            else:
                target = torch.ones(mb_size)

            pred = disc(X2)
            print('pred mean: ', pred.mean())
            encdec_loss = logloss(pred.reshape(-1, 1), target.reshape(-1, 1))
            '''

            target = torch.ones(int(mb_size/2))
            if use_cuda:
                target = target.cuda()
            encdec_loss = 4.0*contrastivec(pred[0::2], pred[1::2], target)


            are_same = are_equal({col: x[::2] for col, x in T.items()}, {col: x[1::2] for col, x in T.items()})
            #print('f same ', torch.mean(torch.mean(are_same, 1)))
            #enc_loss = contrastivec(mu2[::2], mu2[1::2], torch.zeros(int(mb_size / 2)).cuda())
            encdec_loss += 0.5*contrastivec(mu[::2], mu2[1::2], are_same)
            encdec_loss += 0.25*contrastivec(mu[::2], mu[1::2], are_same)
            encdec_loss += 0.25 * contrastivec(mu2[::2], mu2[1::2], are_same)
            encdec_loss += 1.0*contrastivec(mu, mu2, torch.ones(mb_size).cuda())


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


            encdec_loss.backward()
            solver_encdec.step()

            solver_encdec.zero_grad()

            total_loss += encdec_loss.data.cpu().numpy()


            #train the discriminator
            X = {}
            for col, tt in T.items():
                if col in embeddings.keys():
                    X[col] = embeddings[col](tt)
                else:
                    X[col] = tt.float()

            mu = enc(X)
            #X2d = dec(mu)


            X2 = dec(mu)
            T2 = discretize(X2, embeddings, maxlens)
            X2d = {col: tt.detach() for col, tt in T2.items()}

            for col, embedding in embeddings.items():
                X2d[col] = embeddings[col](T2[col].detach())


            p = disc(X)
            p2 = disc(X2d)


            #target = torch.cat((torch.zeros(mb_size), torch.ones(mb_size)), 0)
            #if use_cuda:
            #    target = target.cuda()
            #
            #p0 = disc(X2d)
            #p1 = disc(X)
            #print('p0 mean: ', p0.mean(), p1.mean())
            #pred = torch.cat((p0, p1), 0)
            #disc_loss = logloss(pred.reshape(-1, 1), target.reshape(-1, 1))


            target = torch.zeros(int(mb_size/2))
            if use_cuda:
                target = target.cuda()
            disc_loss = 0.5*contrastivec(p[0::2], p2[1::2], target)
            disc_loss += 0.25*contrastivec(p2[0::2], p2[1::2], 1-target)
            disc_loss += 0.25 * contrastivec(p[0::2], p[1::2], 1 - target)

            disc_loss.backward()
            solver_disc.step()

            solver_disc.zero_grad()

            total_loss += disc_loss.data.cpu().numpy()


            veclen = torch.mean(torch.pow(mu, 2))
            if it % epoch_len == 0:
                print(it, total_loss/epoch_len, veclen.data.cpu().numpy()) #enc_loss.data.cpu().numpy(),

                Xsample = {}
                for col, tt in Tsample.items():
                    if col in embeddings.keys():
                        Xsample[col] = embeddings[col](tt)
                    else:
                        Xsample[col] = tt.float()

                mu = enc(Xsample)
                X2sample = dec(mu)
                T2sample = discretize(X2sample, embeddings, maxlens)


                if 'Fare' in continuous_cols and 'Age' in continuous_cols:
                    print([np.mean(np.abs(Xsample[col].data.cpu().numpy()-X2sample[col].data.cpu().numpy())) for col in ['Fare', 'Age']])

                print({col: tt[0:2].data.cpu().numpy() for col, tt in T2sample.items()})

                if 'Survived' in onehot_cols:
                    print('% survived correct: ', np.mean(T2sample['Survived'].data.cpu().numpy()==Tsample['Survived'].data.cpu().numpy()), np.mean(Tsample['Survived'].data.cpu().numpy()==np.ones_like(Tsample['Survived'].data.cpu().numpy())))

                if 'Cabin' in text_cols:
                    print(embeddings['Cabin'].weight[data.charindex['1']])
                loss = 0.0
                #print(T2.data.cpu()[0, 0:30].numpy())