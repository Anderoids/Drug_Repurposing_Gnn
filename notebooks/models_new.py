import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv



# Baseline Models

class DistMult(nn.Module):
    """DistMult bilinear model for link prediction"""
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
        score = (h * r * t).sum(-1)
        return score


class ComplEx(nn.Module):
    """ComplEx complex-valued embedding model"""
    def __init__(self, num_entities, num_relations, dim=200, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, 2 * dim)  # real+imag
        self.rel_emb = nn.Embedding(num_relations, 2 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, heads, rels, tails):
        h = self.dropout(self.entity_emb(heads))
        r = self.dropout(self.rel_emb(rels))
        t = self.dropout(self.entity_emb(tails))

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = r.chunk(2, dim=-1)
        t_re, t_im = t.chunk(2, dim=-1)

        score = (
            h_re * r_re * t_re +
            h_im * r_re * t_im +
            h_re * r_im * t_im -
            h_im * r_im * t_re
        ).sum(-1)
        return score


class SimpleGraphSAGE(nn.Module):
    """Toy GNN baseline with relation embeddings"""
    def __init__(self, num_entities, num_relations, dim=200, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.W = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, heads, rels, tails):
        h = self.dropout(self.entity_emb(heads))
        r = self.dropout(self.rel_emb(rels))
        t = self.dropout(self.entity_emb(tails))
        concat = torch.cat([h * r, t], dim=-1)
        out = self.W(concat)
        score = (out * t).sum(-1)
        return score

# Novel Proposed Model

class RGCNEncoder(nn.Module):
    """Multi-layer RGCN encoder for heterogeneous graphs"""
    def __init__(self, num_entities, num_relations, dim=200, num_layers=2, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.convs = nn.ModuleList([
            RGCNConv(dim, dim, num_relations, num_bases=30)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, edge_type):
        x = self.entity_emb.weight
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x, edge_index, edge_type)))
        return x


class PathRegularizedDecoder(nn.Module):
    """Decoder combining embeddings with path-support evidence"""
    def __init__(self, hidden_dim, alpha=0.5):
        super().__init__()
        self.rel_mat = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.rel_mat)
        self.alpha = alpha

    def forward(self, drug_emb, disease_emb, path_support=None):
        score = (drug_emb @ self.rel_mat @ disease_emb.T).diag()
        if path_support is not None:
            score = score + self.alpha * path_support
        return score


class ProposedRGCNModel(nn.Module):
    """
    Full proposed model:
    - RGCN encoder
    - Path-regularized bilinear decoder
    """
    def __init__(self, num_entities, num_relations, dim=200, num_layers=2, dropout=0.2, alpha=0.5):
        super().__init__()
        self.encoder = RGCNEncoder(num_entities, num_relations, dim, num_layers, dropout)
        self.decoder = PathRegularizedDecoder(dim, alpha)

    def forward(self, triples, edge_index, edge_type, path_support=None):
        # Encode entities
        x = self.encoder(edge_index, edge_type)

        heads, rels, tails = triples
        drug_emb = x[heads]
        disease_emb = x[tails]

        # Compute score with path evidence
        score = self.decoder(drug_emb, disease_emb, path_support)
        return score
    
# Calibration Wrapper


class ModelWithTemperature(nn.Module):
    """
    Wraps a model to add temperature scaling
    for probability calibration.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature
