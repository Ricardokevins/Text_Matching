import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import torch
import time
import get_data
USE_CUDA = torch.cuda.is_available()
class BiLSTM_Match(nn.Module):

    def __init__(self, pre_train,embedding_dim, hidden_dim, vocab_size, tagset_size,batch_size,str_len):
        super(BiLSTM_Match, self).__init__()

        self.hidden_dim = hidden_dim
        self.str_len=str_len
        self.batch_size=batch_size

        #initrange = 0.5 / embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = pre_train.in_embed
        #self.word_embeddings.weight.data.uniform_(-initrange, initrange)

        self.lstma = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)
        self.lstmb = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)

        # self.dropout1 = nn.Dropout(0.5)
        self.densea= nn.Linear(2*str_len*hidden_dim, hidden_dim)
        self.denseb = nn.Linear(2*str_len * hidden_dim, hidden_dim)

        self.op_prelu=nn.PReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.hidden2taga = nn.Linear(hidden_dim, tagset_size)
        self.hidden2tagb = nn.Linear(hidden_dim, tagset_size)

        self.op_tanh=nn.Tanh()
        self.hiddena = self.init_hidden()
        self.hiddenb = self.init_hidden()

    def init_hidden(self):
        if USE_CUDA:
            return (torch.zeros(2, self.batch_size, self.hidden_dim).cuda(),torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(2, self.batch_size, self.hidden_dim),torch.zeros(2, self.batch_size, self.hidden_dim))

    def forward(self, sentencea , sentenceb , statea,stateb , train_flag):
        embedsa = self.word_embeddings(sentencea)
        embedsb = self.word_embeddings(sentenceb)
        #print(embeds.shape)
        self.hiddena=statea
        self.hiddenb = stateb

        lstm_outa, self.hiddena = self.lstma(embedsa.view(self.str_len, len(sentencea), -1), self.hiddena)
        lstm_outb, self.hiddenb = self.lstmb(embedsb.view(self.str_len, len(sentenceb), -1), self.hiddenb)

        tag_spacea = self.densea(lstm_outa.view(self.batch_size,-1))
        tag_spaceb = self.denseb(lstm_outb.view(self.batch_size, -1))

        tag_spacea = self.op_prelu(tag_spacea)
        tag_spaceb = self.op_prelu(tag_spaceb)

        if train_flag:
            tag_spacea = self.dropout2(tag_spacea)
            tag_spaceb = self.dropout2(tag_spaceb)

        tag_spacea=self.hidden2taga(tag_spacea)
        tag_spaceb = self.hidden2tagb(tag_spaceb)

        tag_spacea=self.op_tanh(tag_spacea)
        tag_spaceb = self.op_tanh(tag_spaceb)

        #similarity = torch.cosine_similarity(tag_spacea, tag_spaceb, dim=1)
        similarity=(F.pairwise_distance(tag_spacea, tag_spaceb, p=2)) # pytorch求欧氏距离
        similarity=similarity.view(-1,1)
        similarity=similarity.float()
        return similarity ,self.hiddena,self.hiddenb
        #tag_spacea=self.op_tanh(tag_spacea)
        #tag_spaceb = self.op_tanh(tag_spaceb)
        """
        tag_spacea = self.op_prelu(tag_spacea)
        tag_spaceb = self.op_prelu(tag_spaceb)
        similarity = torch.cosine_similarity(tag_spacea, tag_spaceb, dim=1)
        similarity+=1
        similarity = similarity.float()
        similarity=similarity/2
        #similarity=(F.pairwise_distance(tag_spacea, tag_spaceb, p=2)) # pytorch求欧氏距离
        similarity=similarity.view(-1,1)
        # Fisrt try

        #similarity=-similarity
        #similarity=torch.exp(similarity, out=None)  # y_i=e^(x_i)
        similarity=similarity.float()
        return similarity ,self.hiddena,self.hiddenb
        """