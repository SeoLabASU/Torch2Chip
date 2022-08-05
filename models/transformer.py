"""
transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from methods import RCFSQ, QLinear, MulShift

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., msabit:int=4, mlpbit:int=8):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, wbit=msabit, abit=msabit)
        self.la2 = nn.LayerNorm(feats)
        
        # quantizer
        self.aq = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        xq = self.aq(x)
        out = self.msa(self.la1(xq)) + xq

        outq = self.aq(out)
        out = self.mlp(self.la2(outq)) + outq
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., wbit=4, abit=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        if wbit < 32:
            self.aq = RCFSQ(nbit=abit, train_flag=True, alpha=8.0)
            self.qq = RCFSQ(nbit=abit, train_flag=True, alpha=2.0)
            self.kq = RCFSQ(nbit=abit, train_flag=True, alpha=2.0)
            self.vq = RCFSQ(nbit=abit, train_flag=True, alpha=2.0)

            self.q = QLinear(feats, feats, wbit=wbit, abit=32)
            self.k = QLinear(feats, feats, wbit=wbit, abit=32)
            self.v = QLinear(feats, feats, wbit=wbit, abit=32)

            self.o = QLinear(feats, feats, wbit=wbit)
        else:

            self.aq = nn.Identity()
            self.qq = nn.Identity()
            self.kq = nn.Identity()
            self.vq = nn.Identity()

            self.q = nn.Linear(feats, feats)
            self.k = nn.Linear(feats, feats)
            self.v = nn.Linear(feats, feats)

            self.o = nn.Linear(feats, feats)
    
        # dropout
        self.dropout = nn.Dropout(dropout)

        # dequantizer
        self.deq = MulShift()
        self.deq.scale.data = torch.tensor(1 / self.sqrt_d)

        self.vdeq = MulShift()


    def forward(self, x):
        b, n, f = x.size()
        xq = self.aq(x)
        q = self.q(xq).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(xq).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(xq).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        # low precision attn
        q = self.qq(q)
        k = self.kq(k)
        v = self.vq(v)
        
        # sparsification
        score = torch.einsum("bhif, bhjf->bhij", q, k)
        score = self.deq(score)

        score = F.softmax(score, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        attn = self.vdeq(attn)

        o = self.dropout(self.o(attn.flatten(2)))
        return o