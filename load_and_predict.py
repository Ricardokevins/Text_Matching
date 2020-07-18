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
Batchsize=1
stringlen=25
#stringlen=10
Epoch=20
#lr=0.005
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
resulta,resultb=get_data.result_data(stringlen)
if USE_CUDA:
    resulta=resulta.cuda()
    resultb=resultb.cuda()

def out_put(net,resulta,resultb,batchsize):
    net.eval()
    dataset = torch.utils.data.TensorDataset(resulta, resultb)
    train_iter = torch.utils.data.DataLoader(dataset, batchsize, shuffle=False)
    statea = None
    stateb = None
    filename = 'SheShuaiJie_NJU_predict.txt'
    index=0
    with open(filename, 'w+') as file_object:
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
                    index+=1
        print(index)
        while index<12500:
            file_object.write(str(i) + "\n")
            index += 1
    return

out_put(model,resulta,resultb,Batchsize)