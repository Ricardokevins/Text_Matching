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
"""
embedding_dim=400
hidden_dim=256
vocab_size=51158
target=1
Batchsize=64
stringlen=25
Epoch=20
lr=0.1
"""
embedding_dim=400
hidden_dim=512
vocab_size=51158
target=1
Batchsize=16
stringlen=25
Epoch=20
lr=0.1


USE_Bi=True
if USE_Bi:
    print("Using BiLSTM")
    model = BiLSTM_Match(embedding_dim, hidden_dim, vocab_size, target, Batchsize, stringlen)
    model_path = "./Model/BiLSTMmodel.pth"
else:
    print("Using LSTM")
    model = LSTM_Match(embedding_dim, hidden_dim, vocab_size,target,Batchsize,stringlen)
    model_path = "./Model/LSTMmodel.pth"

model.load_state_dict(torch.load(model_path))
model=model.cuda()
print(model)

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

def eval(net,eval_dataa,eval_datab, eval_label,batch_size):
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
    return s

eval(model,evala,evalb,evallabels,Batchsize)