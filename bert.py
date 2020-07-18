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
import argparse
import get_data

import os
#main.py
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
#from transformers import BertPreTrainedModel

embedding_dim=400
hidden_dim=256
vocab_size=51158
target=1
Batchsize=128
stringlen=25
Epoch=20
lr=0.01
from transformers import BertModel


USE_CUDA = torch.cuda.is_available()

texta,textb,labels,evala,evalb,evallabels=get_data.train_data(stringlen)
resulta,resultb=get_data.result_data(stringlen)

if USE_CUDA:
    texta = texta.cuda()
    textb= textb.cuda()
    labels= labels.cuda()
    evala= evala.cuda()
    evalb= evalb.cuda()
    evallabels= evallabels.cuda()
    resulta=resulta.cuda()
    resultb=resulta.cuda()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.add_tokens([str(i) for i in range(0,51157)])

len_token=len(tokenizer)

print(len_token)

def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=2,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=30,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.001,help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=2,help='分类类数')
    parser.add_argument('--data_path',type=str,default='./data',help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt,len_token):
    model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=opt.num_labels)
    #model = bert_.BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=opt.num_labels)
    #model = bert_LSTM.Net()
    model.resize_token_embeddings(len_token)
    return model

def train(net, texta,textb,labels,evala,evalb,evallabels,resulta,resultb,num_epochs, learning_rate,  batch_size):
    net.train()
    #loss_fct = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(texta,textb,labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    pre=0.5
    acc_list=[]
    index_list=[]
    pre_list=[]
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        net.train()
        for XA, XB , y in train_iter:
            iter += 1
            XA = XA.long()
            XB = XB.long()
            y=y.long()
            y=y.view(-1)
            #print(y)
            if XA.size(0)!= batch_size:
                break
            optimizer.zero_grad()

            outputs= net(XA,XB, labels=y)
            loss, logits = outputs[0], outputs[1]
            _, predicted = torch.max(logits.data, 1)
            loss.backward()
            optimizer.step()

            total += XA.size(0)
            #print("predict",predicted)
            #print("label",y)
            correct += predicted.data.eq(y.data).cpu().sum()
            s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))

            if iter % 200 ==0:
                print(s)
            print(s)
                #print(net.state_dict()["word_embeddings.weight"])
        s = ((1.0 * correct) / total)
        print("epoch: ",epoch, " ",correct,"/" , total, "TrainAcc:", s)
        acc_list.append(s)
        index_list.append(epoch)


    return acc_list,index_list,pre_list

opt = get_train_args()
model=get_model(opt,len_token)
model=model.cuda()
a,b,c=train(model,texta,textb,labels,evala,evalb,evallabels,resulta,resultb,Epoch,lr,2)
