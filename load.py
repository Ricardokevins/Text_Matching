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



embedding_dim=400
hidden_dim=512
vocab_size=51158
target=1
Batchsize=16
stringlen=25
Epoch=20
lr=0.1

from esmi import ESIM

model=ESIM(256,400,256)


if USE_CUDA:
    model=model.cuda()

model_path="./Model/ESIM.pkt"



model.load_state_dict(torch.load(model_path))

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

def out_put(net,resulta,resultb,batchsize):
    net.eval()
    dataset = torch.utils.data.TensorDataset(resulta, resultb)
    train_iter = torch.utils.data.DataLoader(dataset, batchsize, shuffle=False)
    filename = 'SheShuaiJie_NJU_predict.txt'
    index=0
    with open(filename, 'w+') as file_object:
        with torch.no_grad():
            for XA, XB in train_iter:
                XA = XA.long()
                XB = XB.long()

                if XA.size(0) != batchsize:
                    break
                output = net(XA, XB)
                _, predicted = torch.max(output.data, -1)

                predicted = predicted.reshape(-1)
                result = predicted.cpu().numpy().tolist()

                for i in result:
                    file_object.write(str(i) + "\n")
                    index+=1
                    print(index)
        print(index)
        while index<12500:
            file_object.write(str(i) + "\n")
            index += 1
    return

#

def eval(net,eval_dataa,eval_datab, eval_label,batch_size):
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
            output=net(XB,XA)
            _, predicted = torch.max(output.data, -1)
            y = y.reshape(-1)
            total += XA.size(0)
            print(output)
            print(predicted)
            predicted = predicted.reshape(-1)
            y = y.reshape(-1)
            correct += predicted.data.eq(y.data).cpu().sum()
        s = correct.item()/total
        print(correct.item(),"/" , total, "TestAcc: ", s)
    return s

eval(model,evala,evalb,evallabels,Batchsize)
out_put(model,resulta,resultb,1)

