import numpy as np
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
USE_CUDA = torch.cuda.is_available()
import get_data

from model import BiLSTM_Match
from model import LSTM_Match
from esmi import ESIM
import matplotlib.pyplot as plt

import pandas as pd

np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)


embedding_dim=400
hidden_dim=256
vocab_size=51158
target=1
Batchsize=128
stringlen=25
Epoch=20
lr=0.001

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

def out_put(net,resulta,resultb):
    net.eval()
    dataset = torch.utils.data.TensorDataset(resulta, resultb)
    train_iter = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
    filename = 'SheShuaiJie_NJU_predict.txt'
    with open(filename, 'w') as file_object:
        print("open success")
        with torch.no_grad():
            for XA, XB in train_iter:
                XA = XA.long()
                XB = XB.long()
                output = net(XA, XB)
                _, predicted = torch.max(output.data, -1)
                predicted = predicted.reshape(-1)
                result = predicted.cpu().numpy().tolist()

                for i in result:
                    file_object.write(str(i)+"\n")
    return

def eval(net,eval_dataa,eval_datab, eval_label,resulta,resultb,batch_size,pre,model_path):
    net.eval()
    dataset = torch.utils.data.TensorDataset(eval_dataa,eval_datab, eval_label)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    total=0
    correct=0

    with torch.no_grad():
        for XA, XB , y in train_iter:
            XA = XA.long()
            XB = XB.long()
            if XA.size(0)!= batch_size:
                break


            output=net(XA,XB)
            _, predicted = torch.max(output.data, -1)
            y = y.reshape(-1)
            total += XA.size(0)
            predicted = predicted.reshape(-1)
            y = y.reshape(-1)
            correct += predicted.data.eq(y.data).cpu().sum()
        s = correct.item()/total
        print(correct.item(),"/" , total, "TestAcc: ", s)
    if s > pre:
        print("Flush and save model")
        torch.save(net.state_dict(), model_path)
        out_put(net,resulta,resultb)
    return s

def train(net, texta,textb,labels,evala,evalb,evallabels,resulta,resultb,num_epochs, learning_rate,  batch_size,model_path):
    net.train()

    loss_fct = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
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
            #y=y.float()
            y=y.long()
            if XA.size(0)!= batch_size:
                break

            optimizer.zero_grad()

            output=net(XA,XB)
            _, predicted = torch.max(output.data, -1)
            y=y.reshape(-1)
            #predicted=predicted.float()
            #print(output.size())
            #print(y.size())

            loss = loss_fct(output, y)
            #exit()
            #print(predicted.data)
            #print(output)
            loss.backward()
            optimizer.step()
            total += XA.size(0)
            predicted=predicted.reshape(-1)
            y=y.reshape(-1)
            correct += predicted.data.eq(y.data).cpu().sum()
            if iter % 200 ==0:
                s = correct.item()/total
                print(correct,"/" , total, "TrainAcc:", s)
                #print(net.state_dict()["word_embeddings.weight"])
        s = correct.item()/total
        print("epoch: ",epoch, " ",correct,"/" , total, "TrainAcc:", s)
    return



USE_Bi=True



model=ESIM(256,400,256)


if USE_CUDA:
    model=model.cuda()

model_path="./Model/ESIM.pkt"

model=model.cuda()
print(model)

train(model,texta,textb,labels,evala,evalb,evallabels,resulta,resultb,Epoch,lr,Batchsize,model_path)


