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


from model import EmbeddingModel
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
lr=0.01
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
    resultb=resulta.cuda()


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
"""
USE_Bi=True
"""
w2v = EmbeddingModel(vocab_size, embedding_dim)
checkpoint = torch.load('Model/checkpoint.pth2.tar')
w2v.load_state_dict(checkpoint['state_dict'])  # 模型参数

print(w2v.state_dict()["in_embed.weight"])
"""
if USE_Bi:
    print("Using BiLSTM")
    model = BiLSTM_Match(w2v,embedding_dim, hidden_dim, vocab_size, target, Batchsize, stringlen)
    model_path = "./Model/BiLSTMmodel.pth"
else:
    print("Using LSTM")
    model = LSTM_Match(embedding_dim, hidden_dim, vocab_size,target,Batchsize,stringlen)
    model_path = "./Model/LSTMmodel.pth"

print(w2v.in_embed==model.word_embeddings)


print(model)
if USE_CUDA:
    model=model.cuda()
"""

"""


"""
class BiLSTM_Match(nn.Module):

    def __init__(self, pre_train,embedding_dim, hidden_dim, vocab_size, tagset_size,batch_size,str_len):
        super(BiLSTM_Match, self).__init__()

        self.hidden_dim = hidden_dim
        self.str_len=str_len
        self.batch_size=batch_size

        self.word_embeddings = pre_train.in_embed
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

model=BiLSTM_Match(w2v,embedding_dim, hidden_dim, vocab_size, target, Batchsize, stringlen)

model_path = "./Model/BiW2vmodel.pth"



model=model.cuda()
print(model)
a,b,c=train(model,texta,textb,labels,evala,evalb,evallabels,resulta,resultb,Epoch,lr,Batchsize,model_path)
Draw(a,b,c)

