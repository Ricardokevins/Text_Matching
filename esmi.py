from torch import nn
import torch
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self, hidden_size,embeds_dim,linear_size):
        super(ESIM, self).__init__()
        #self.args = args
        self.dropout = 0.5
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        num_word = 51158
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        #self.embeds = pre.in_embed
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, 2),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        print("x1.size()",x1.size())
        print("x2.size()",x2.size())
        attention = torch.matmul(x1, x2.transpose(1, 2))
        print("attention.size()",attention.size())
        #print(mask1)
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        print("mask1.size()",mask1.size())

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        temp=torch.cat([sub, mul], -1)
        print("temp",temp.size())
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, input1,input2):
        # batch_size * seq_len
        sent1, sent2 = input1, input2
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        #print(sent1)(data)
        #print(mask1)(True or false by eq 0)
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)
        print("q1 combination",q1_combined.size())
        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
        print("q1 compose", q1_compose.size())
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)
        print("q1 compose", q1_rep.size())
        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity
