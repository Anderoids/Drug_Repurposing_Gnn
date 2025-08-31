import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv  # force import, no try/except



class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, rels, tails):
        h = self.dropout(self.entity_emb(heads))
        r = self.dropout(self.rel_emb(rels))
        t = self.dropout(self.entity_emb(tails))
        score = (h * r * t).sum(-1)   # DistMult scoring
        return score


class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, 2*dim)  # real+imag
        self.rel_emb = nn.Embedding(num_relations, 2*dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, heads, rels, tails):
        h = self.dropout(self.entity_emb(heads))
        r = self.dropout(self.rel_emb(rels))
        t = self.dropout(self.entity_emb(tails))
        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = r.chunk(2, dim=-1)
        t_re, t_im = t.chunk(2, dim=-1)
        score = (h_re*r_re*t_re + h_im*r_re*t_im + h_re*r_im*t_im - h_im*r_im*t_re).sum(-1)
        return score


class SimpleGraphSAGE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.W = nn.Linear(2*dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, heads, rels, tails):
        h = self.dropout(self.entity_emb(heads))
        r = self.dropout(self.rel_emb(rels))
        t = self.dropout(self.entity_emb(tails))
        concat = torch.cat([h * r, t], dim=-1)
        out = self.W(concat)
        score = (out * t).sum(-1)
        return score
