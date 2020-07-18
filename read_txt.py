data_path="./data/"
train_path="train.txt"

result_path="test.txt"

max_len=0
min_len=1000
hh=[]

def str2int(a):
    temp=a.split(" ")
    for i in range(len(temp)):
        temp[i] = int(temp[i])
    return temp



def get_train():
    f = open(data_path + train_path, "r")
    lines = f.readlines()
    texta = []
    textb = []
    labels = []
    for i in lines:
        a = i.split("\t")
        texta.append(str2int(a[0]))
        textb.append(str2int(a[1]))
        labels.append(str2int(a[2]))
    return texta,textb,labels

def get_result():
    f = open(data_path + result_path, "r")
    lines = f.readlines()
    texta = []
    textb = []
    for i in lines:
        a = i.split("\t")
        texta.append(str2int(a[0]))
        textb.append(str2int(a[1]))
    return texta,textb

texta,textb,labels=get_train()

resulta,resultb=get_result()

dict=[]
min=10000000
max=-1
one=0
zero=0
for i in labels:
    if i[0] == 1:
        one+=1
    else:
        zero+=1

for i in texta:
    if len(i) > max_len:
        max_len = len(i)
    if len(i) < min_len:
        min_len = len(i)
    for j in i:

        if j>max:
            max=j
        if j<min:
            min=j
        dict.append(j)

for i in textb:
    if len(i) > max_len:
        max_len = len(i)
    if len(i) < min_len:
        min_len = len(i)
    for j in i:
        if j>max:
            max=j
        if j<min:
            min=j
        dict.append(j)

for i in resulta:
    if len(i) > max_len:
        max_len = len(i)
    if len(i) < min_len:
        min_len = len(i)
    for j in i:

        if j>max:
            max=j
        if j<min:
            min=j
        dict.append(j)

for i in resultb:
    if len(i) > max_len:
        max_len = len(i)
    if len(i) < min_len:
        min_len = len(i)
    for j in i:
        if j>max:
            max=j
        if j<min:
            min=j
        dict.append(j)
dict=set(dict)
"""
print(len(dict))
print(max)
print(min)
print(max_len)
print(min_len)
14205
51157
0
32
1

"""

