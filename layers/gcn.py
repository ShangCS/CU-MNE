import torch
import torch.nn as nn
from torch.nn import functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, isBias=True):
        super(GCN, self).__init__()

        self.w = nn.Linear(in_ft, out_ft, bias=False)
        self.w1 = nn.Linear(in_ft, out_ft, bias=False)
        self.dropout = 0.5
        self.act = nn.ReLU()#ReLU Tanh

        if isBias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feature, adj, sparse=False):
        value_1 = self.w1(feature)
        if sparse:
            value_2 = self.w(torch.unsqueeze(torch.spmm(adj, torch.squeeze(feature, 0)), 0))
        else:
            value_2 = self.w(torch.bmm(adj, feature))
        value = value_1 + value_2
        if self.isBias:
            value += self.bias
        value = F.dropout(value, self.dropout, training=self.training)

        return self.act(value)
