import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np

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



# RGCN Encoder

class RGCNEncoder(nn.Module):
    """RGCN Encoder for heterogeneous knowledge graphs."""
    def __init__(self, num_entities, num_relations, hidden_dim=200, num_layers=2, dropout=0.2, num_bases=30):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, edge_type):
        x = self.entity_emb.weight
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
            x = self.dropout(x)
        return x

# Path-Regularized Bilinear Decoder

class PathRegularizedDecoder(nn.Module):
    """Decoder combining entity embeddings with path-support scores."""
    def __init__(self, hidden_dim, alpha=0.5):
        super().__init__()
        self.rel_mat = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.rel_mat)
        self.alpha = alpha

    def forward(self, head_emb, tail_emb, path_support=None):
        # Bilinear score
        score = torch.sum(head_emb @ self.rel_mat * tail_emb, dim=-1)
        if path_support is not None:
            score = score + self.alpha * path_support
        return score


# Full Proposed Model

class ProposedRGCNModel(nn.Module):
    """
    Novel Knowledge Graph Link Prediction Model:
    - RGCN encoder for entity embeddings
    - Path-regularized bilinear decoder
    """
    def __init__(self, num_entities, num_relations, hidden_dim=200, num_layers=2, dropout=0.2, alpha=0.5):
        super().__init__()
        self.encoder = RGCNEncoder(num_entities, num_relations, hidden_dim, num_layers, dropout)
        self.decoder = PathRegularizedDecoder(hidden_dim, alpha)

    def forward(self, triples, edge_index, edge_type, path_support=None):
        """
        triples: tuple of (heads, relations, tails)
        """
        heads, rels, tails = triples
        x = self.encoder(edge_index, edge_type)

        # Extract embeddings for heads and tails
        head_emb = x[heads]
        tail_emb = x[tails]

        # Compute scores
        score = self.decoder(head_emb, tail_emb, path_support)
        return score

# Optional Temperature Calibration

class ModelWithTemperature(nn.Module):
    """Wraps a model to add temperature scaling for probability calibration."""
    def __init__(self, model, init_temp=1.5):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature


def validate_and_map_triples(triples, ent2id, rel2id):
    """
    Validate triples and map them to indices.
    Any entity or relation not in ent2id/rel2id will be removed.
    
    Args:
        triples: List of (head, rel, tail) tuples with original names/ids
        ent2id: dict mapping entity -> index
        rel2id: dict mapping relation -> index
        
    Returns:
        Tensor of shape [3, N] with (heads, rels, tails) indices
    """
    heads, rels, tails = [], [], []
    skipped = 0

    for h, r, t in triples:
        if h not in ent2id or t not in ent2id or r not in rel2id:
            skipped += 1
            continue
        heads.append(ent2id[h])
        tails.append(ent2id[t])
        rels.append(rel2id[r])

    if skipped > 0:
        print(f"Skipped {skipped} triples due to missing entities or relations.")

    # Convert to torch tensors
    heads = torch.tensor(heads, dtype=torch.long)
    tails = torch.tensor(tails, dtype=torch.long)
    rels = torch.tensor(rels, dtype=torch.long)

    return heads, rels, tails

