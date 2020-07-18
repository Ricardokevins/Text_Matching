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
"""
embedding_dim=400
hidden_dim=256
vocab_size=51158
target=1
Batchsize=8
stringlen=25
Epoch=20
lr=0.1
"""
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
    resultb=resultb.cuda()


def out_put(net,resulta,resultb,batchsize):
    net.eval()
    dataset = torch.utils.data.TensorDataset(resulta, resultb)
    train_iter = torch.utils.data.DataLoader(dataset, batchsize, shuffle=False)
    statea = None
    stateb = None
    filename = 'SheShuaiJie_NJU_predict.txt'
    with open(filename, 'w') as file_object:
        print("open success")
        with torch.no_grad():
            for XA, XB in train_iter:
                XA = XA.long()
                XB = XB.long()
                if XA.size(0)!= batchsize:
                    break
                if statea is not None:
                    if isinstance(statea, tuple):  # LSTM, state:(h, c)
                        statea = (statea[0].detach(), statea[1].detach())
                    else:
                        statea = statea.detach()
                if stateb is not None:
                    if isinstance(stateb, tuple):  # LSTM, state:(h, c)
                        stateb = (stateb[0].detach(), stateb[1].detach())
                    else:
                        stateb = stateb.detach()
                (output, statea, stateb) = net(XA, XB, statea, stateb, False)
                result = output.detach().cpu().numpy().tolist()
                result = [1 if i[0] > 0.5 else 0 for i in result]
                for i in result:
                    file_object.write(str(i)+"\n")
    return

def eval(net,eval_dataa,eval_datab, eval_label,resulta,resultb,batch_size,pre,model_path):
    net.eval()
    dataset = torch.utils.data.TensorDataset(eval_dataa,eval_datab, eval_label)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    total=0
    correct=0
    statea = None
    stateb = None
    with torch.no_grad():
        for XA, XB , y in train_iter:
            XA = XA.long()
            XB = XB.long()
            if XA.size(0)!= batch_size:
                break

            if statea is not None:
                if isinstance(statea, tuple):  # LSTM, state:(h, c)
                    statea = (statea[0].detach(), statea[1].detach())
                else:
                    statea = statea.detach()
            if stateb is not None:
                if isinstance(stateb, tuple):  # LSTM, state:(h, c)
                    stateb = (stateb[0].detach(), stateb[1].detach())
                else:
                    stateb = stateb.detach()

            (output, statea, stateb) = net(XA, XB, statea, stateb, False)
            total += XA.size(0)
            result = output.detach().cpu().numpy().tolist()
            result = [1 if i[0] > 0.5 else 0 for i in result]
            answer = y.cpu().numpy().tolist()
            for i in range(len(answer)):
                # print(answer[i][0]," ",result[i])
                if answer[i][0] == result[i]:
                    correct += 1

        s = (((1.0 * correct) / total))
        print(correct,"/" , total, "TestAcc: ", s)
    if s > pre:
        print("Flush and save model")
        torch.save(net.state_dict(), model_path)
        out_put(net,resulta,resultb,batch_size)
    return s

def train(net, texta,textb,labels,evala,evalb,evallabels,resulta,resultb,num_epochs, learning_rate,  batch_size,model_path):
    net.train()
    statea = None
    stateb = None
    #loss_fct = nn.CrossEntropyLoss()
    loss_fct = torch.nn.MSELoss()
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
            y=y.float()
            if XA.size(0)!= batch_size:
                break
            if statea is not None:
                if isinstance (statea, tuple): # LSTM, state:(h, c)
                    statea = (statea[0].detach(), statea[1].detach())
                else:
                    statea = statea.detach()
            if stateb is not None:
                if isinstance (stateb, tuple): # LSTM, state:(h, c)
                    stateb = (stateb[0].detach(), stateb[1].detach())
                else:
                    stateb = stateb.detach()
            optimizer.zero_grad()

            (output, statea,stateb) = net(XA,XB, statea,stateb,True)
            loss = loss_fct(output, y)

            #print(output)
            loss.backward()
            optimizer.step()
            total += XA.size(0)
            result = output.detach().cpu().numpy().tolist()

            result=[ 1 if i[0]>0.5 else 0 for i in result]
            answer=y.cpu().numpy().tolist()
            for i in range(len(answer)):
                #print(answer[i][0]," ",result[i])
                if answer[i][0]==result[i]:
                    correct += 1
            if iter % 200 ==0:
                s = (((1.0 * correct) / total))
                print(correct,"/" , total, "TrainAcc:", s)

                #print(net.state_dict()["word_embeddings.weight"])
        s = ((1.0 * correct) / total)
        print("epoch: ",epoch, " ",correct,"/" , total, "TrainAcc:", s)
        temp=eval(net,evala,evalb,evallabels,resulta,resultb,batch_size,pre,model_path)
        if temp>pre:
            pre=temp
        acc_list.append(s)
        index_list.append(epoch)
        pre_list.append(temp)

    return acc_list,index_list,pre_list

def Draw(acc_list,index_list,pre_list):
    tempa = np.array(acc_list)
    tempb = np.array(index_list)
    tempc = np.array(pre_list)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 1)
    df1 = pd.DataFrame(tempa, index=tempb, columns=['Train'])
    df2 = pd.DataFrame(tempc, index=tempb, columns=['Test'])
    df1.plot(ax=ax1, kind='line', rot=360, grid='on')
    ax1.set_xticks(range(len(index_list)))
    ax1.set_xticklabels(range(len(index_list)))
    df2.plot(ax=ax2, kind='line', rot=360, grid='on')
    ax2.set_xticks(range(Epoch))
    ax2.set_xticklabels(range(Epoch))
    plt.show()

USE_Bi=True



if USE_Bi:
    print("Using BiLSTM")
    model = BiLSTM_Match(embedding_dim, hidden_dim, vocab_size, target, Batchsize, stringlen)
    model_path = "./Model/BiLSTMmodel.pth"
else:
    print("Using LSTM")
    model = LSTM_Match(embedding_dim, hidden_dim, vocab_size,target,Batchsize,stringlen)
    model_path = "./Model/LSTMmodel.pth"




if USE_CUDA:
    model=model.cuda()



model=model.cuda()
print(model)
exit()
a,b,c=train(model,texta,textb,labels,evala,evalb,evallabels,resulta,resultb,Epoch,lr,Batchsize,model_path)
Draw(a,b,c)

