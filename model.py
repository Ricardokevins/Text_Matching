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

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,batch_size,str_len):
        super(BiLSTM_Match, self).__init__()

        self.hidden_dim = hidden_dim
        self.str_len=str_len
        self.batch_size=batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
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



class LSTM_Match(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,batch_size,str_len):
        super(LSTM_Match, self).__init__()

        self.hidden_dim = hidden_dim
        self.str_len=str_len
        self.batch_size=batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstma = nn.LSTM(embedding_dim, hidden_dim)
        self.lstmb = nn.LSTM(embedding_dim, hidden_dim)

        #self.dropout1 = nn.Dropout(0.5)
        self.densea= nn.Linear(str_len*hidden_dim, hidden_dim)
        self.denseb = nn.Linear(str_len * hidden_dim, hidden_dim)

        self.op_prelu=nn.PReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.hidden2taga = nn.Linear(hidden_dim, tagset_size)
        self.hidden2tagb = nn.Linear(hidden_dim, tagset_size)

        self.op_tanh=nn.Tanh()
        self.hiddena = self.init_hidden()
        self.hiddenb = self.init_hidden()

    def init_hidden(self):
        if USE_CUDA:
            return (torch.zeros(1, self.batch_size, self.hidden_dim).cuda(),torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, self.batch_size, self.hidden_dim),torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sentencea , sentenceb , statea,stateb , train_flag):
        embedsa = self.word_embeddings(sentencea)
        embedsb = self.word_embeddings(sentenceb)
        #print(embeds.shape)
        self.hiddena = statea
        self.hiddenb = stateb

        lstm_outa, self.hiddena = self.lstma(embedsa.view(self.str_len, len(sentencea), -1), self.hiddena)
        lstm_outb, self.hiddenb = self.lstmb(embedsb.view(self.str_len, len(sentenceb), -1), self.hiddenb)

        if train_flag:
            lstm_outa=self.dropout1(lstm_outa)
            lstm_outb = self.dropout1(lstm_outb)

        tag_spacea = self.densea(lstm_outa.view(self.batch_size,-1))
        tag_spaceb = self.denseb(lstm_outb.view(self.batch_size, -1))

        tag_spacea = self.op_prelu(tag_spacea)
        tag_spaceb = self.op_prelu(tag_spaceb)
        if train_flag:
            tag_spacea = self.dropout2(tag_spacea)
            tag_spaceb = self.dropout2(tag_spaceb)

        tag_spacea = self.hidden2taga(tag_spacea)
        tag_spaceb = self.hidden2tagb(tag_spaceb)

        tag_spacea = self.op_tanh(tag_spacea)
        tag_spaceb = self.op_tanh(tag_spaceb)


        #similarity = torch.cosine_similarity(tag_spacea, tag_spaceb, dim=1)
        similarity=(F.pairwise_distance(tag_spacea, tag_spaceb, p=2)) # pytorch求欧氏距离
        similarity=similarity.view(-1,1)
        similarity=similarity.float()
        #print(tag_spacea)
        #print(tag_spaceb)

        #print(similarity)
        #print("t", similarity.shape)
        #tag_scores = F.log_softmax(tag_space,dim=1)
        #exit()
        return similarity ,self.hiddena,self.hiddenb

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]

        return: loss, [batch_size]
        '''

        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


