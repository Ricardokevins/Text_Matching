import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter
import jieba
from collections import Counter
import numpy as np
import random
import math
import os
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import time



USE_CUDA = torch.cuda.is_available()
#USE_CUDA=0
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

if USE_CUDA:
    print("train on GPU")
# 设定一些超参数

K = 100  # number of negative samples
C = 3  # nearby words threshold
NUM_EPOCHS = 100  # The number of epochs of training
MAX_VOCAB_SIZE = 51158  # the vocabulary size
BATCH_SIZE = 128  # the batch size
#LEARNING_RATE = 0.2  # the initial learning rate
LEARNING_RATE = 0.08#0.05 too small
EMBEDDING_SIZE = 400

cur_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# tokenize函数，把一篇文本转化成一个个单词

def word_tokenize(text):
    return text.split()

base_path= "data/corpus.txt"

"""
with open(base_path, "r",encoding="UTF-8") as fin:
    text = fin.read()
"""
file = open(base_path)
#830606
index=0
text=""
for line in file:
    if index>=10000:
        break
    text+=line
    text+=" "
    index+=1

text = [w for w in word_tokenize(text)]

vocab={}
for i in range(51158):
    vocab[str(i)]=int(i)

vocab1 = dict(Counter(text).most_common(MAX_VOCAB_SIZE))
print(len(vocab))
print(len(vocab1))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
word_counts=np.array([vocab1[i] if i in vocab1 else 0 for i in vocab.keys()], dtype=np.float32)
print(word_counts)
print(len(word_counts))

#word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
VOCAB_SIZE = len(idx_to_word)

print(VOCAB_SIZE)

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words

dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


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

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)




optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

iter=0
epoch=0

if os.path.exists('Model/checkpoint.pth2.tar'):
    checkpoint = torch.load('Model/checkpoint.pth2.tar')
    model.load_state_dict(checkpoint['state_dict'])  # 模型参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 优化参数
    iter = checkpoint['iter']
    epoch=checkpoint['epoch']
    print("loading successfully!!!")

if USE_CUDA:
    model = model.cuda()

set_flag=False
if set_flag:
    iter=0
    epoch=0

train_new_set=False
if train_new_set:
    iter=0
    epoch=0

for e in range(epoch,NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader,start=iter):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            result1="epoch: {}, iter: {}, loss: {}".format(e, i, loss.item())
            print(result1)
        if i % 1000 == 0:
            state = {'iter': i + 1,  # 保存的当前轮数
                     'epoch':e,
                     'state_dict': model.state_dict(),  # 训练好的参数
                     'optimizer': optimizer.state_dict(),  # 优化器参数,为了后续的resume
                     'embedding': model.input_embeddings()
                     }
            # 保存模型到checkpoint.pth.tar
            torch.save(state, 'Model/checkpoint.pth2.tar')
            print("save success")
    iter=0





