import torch
import read_txt




def train_data(str_len):

    texta, textb, labels = read_txt.get_train()
    maxa = -1
    maxb = -1
    total_lena = 0
    total_numa = 0
    total_lenb = 0
    total_numb = 0
    for i in range(len(texta)):
        total_lena += len(texta[i])
        total_numa += 1
        if(len(texta[i])>maxa):
            maxa=len(texta[i])
        if len(texta[i])>str_len:
            texta[i]=texta[i][0:str_len-1]
        while len(texta[i])<str_len:
            texta[i].append(0)

    for i in range(len(textb)):
        total_lenb += len(textb[i])
        total_numb += 1
        if (len(textb[i]) > maxb):
            maxb = len(textb[i])
        if len(textb[i])>str_len:
            textb[i]=textb[i][0:str_len-1]
        while len(textb[i])<str_len:
            textb[i].append(0)
    print(maxa,maxb)
    print(total_lena/total_numa)
    print(total_lenb/total_numb)
    print((total_lenb+total_lena)/(total_numb+total_numa))
    evala = texta[218000:]
    evalb = textb[218000:]
    evallabels = labels[218000:]

    texta = texta[0:218000]
    textb = textb[0:218000]
    labels = labels[0:218000]

    texta = torch.tensor(texta)
    textb = torch.tensor(textb)
    labels = torch.tensor(labels)

    evala = torch.tensor(evala)
    evalb = torch.tensor(evalb)
    evallabels = torch.tensor(evallabels)
    return texta,textb,labels,evala,evalb,evallabels


def result_data(str_len):
    max_lena = -1
    max_lenb = -1
    total_lena = 0
    total_numa = 0
    total_lenb = 0
    total_numb = 0
    texta, textb = read_txt.get_result()
    for i in range(len(texta)):
        total_lena+=len(texta[i])
        total_numa+=1
        if len(texta[i])>max_lena:
            max_lena=len(texta[i])
        if len(texta[i]) > str_len:
            texta[i] = texta[i][0:str_len-1]
        while len(texta[i]) < str_len:
            texta[i].append(0)

    for i in range(len(textb)):
        total_lenb += len(textb[i])
        total_numb += 1
        if len(textb[i])>max_lenb:
            max_lenb=len(textb[i])
        if len(textb[i]) > str_len:
            textb[i] = textb[i][0:str_len-1]
        while len(textb[i]) < str_len:
            textb[i].append(0)

    print(max_lena, max_lenb)
    print(total_lena / total_numa)
    print(total_lenb / total_numb)
    print((total_lenb + total_lena) / (total_numb + total_numa))
    texta = torch.tensor(texta)
    textb = torch.tensor(textb)

    return texta, textb

