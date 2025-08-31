import random
from collections import defaultdict
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Iterable

def load_vocab(path_entities_txt: str, path_relations_txt: str):
    ent2id = {}
    id2ent = {}
    with open(path_entities_txt, 'r') as f:
        for line in f:
            idx, ent = line.strip().split("\t")
            ent2id[ent] = int(idx)
            id2ent[int(idx)] = ent
    rel2id = {}
    id2rel = {}
    with open(path_relations_txt, 'r') as f:
        for line in f:
            idx, rel = line.strip().split("\t")
            rel2id[rel] = int(idx)
            id2rel[int(idx)] = rel
    return ent2id, id2ent, rel2id, id2rel

def load_triples_csv(path_csv: str):
    """Expect columns head,relation,tail (csv saved by earlier pipeline)"""
    df = pd.read_csv(path_csv)
    return df

def build_edge_list_from_df(df: pd.DataFrame, ent2id: Dict[str,int], rel2id: Dict[str,int]):
    src = []
    dst = []
    rels = []
    for _, r in df.iterrows():
        h, rel, t = r['head'], r['relation'], r['tail']
        if h in ent2id and t in ent2id and rel in rel2id:
            src.append(ent2id[h]); dst.append(ent2id[t]); rels.append(rel2id[rel])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type  = torch.tensor(rels, dtype=torch.long)
    return edge_index, edge_type

class KGDataset(torch.utils.data.Dataset):
    def __init__(self, triples: List[Tuple[int,int,int]]):
        self.samples = triples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def negative_sample_for_batch(batch_pos: List[Tuple[int,int,int]], num_entities: int, neg_per_pos: int = 5):
    negs = []
    for (h,r,t) in batch_pos:
        for _ in range(neg_per_pos):
            if random.random() < 0.5:
                h_neg = random.randrange(num_entities)
                negs.append((h_neg, r, t))
            else:
                t_neg = random.randrange(num_entities)
                negs.append((h, r, t_neg))
    return negs

def collate_for_loader(batch_pos, num_entities, neg_per_pos=5):
    # batch_pos: list of (h,r,t)
    pos = batch_pos
    neg = negative_sample_for_batch(pos, num_entities, neg_per_pos)
    all_triples = pos + neg
    heads = torch.tensor([h for h,_,_ in all_triples], dtype=torch.long)
    rels  = torch.tensor([r for _,r,_ in all_triples], dtype=torch.long)
    tails = torch.tensor([t for _,_,t in all_triples], dtype=torch.long)
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))], dim=0)
    return heads, rels, tails, labels

# Path-support precompute

def compute_2hop_drug_gene_disease_support(train_df: pd.DataFrame, ent2id: Dict[str,int]):
    """
    Compute count of Drug -> Gene -> Disease 2-hop paths for each (drug,disease) pair
    using only train_df. Returns dict keyed by (drug_id, disease_id) -> count.
    """
    # Build adjacency maps by relation semantics (we look for drug->gene and gene->disease edges)
    drug_to_genes = defaultdict(set)
    gene_to_diseases = defaultdict(set)
    for _, row in train_df.iterrows():
        h, rel, t = row['head'], row['relation'], row['tail']
        # heuristics: relation strings that indicate target / assoc
        rel_l = rel.lower()
        if ('target' in rel_l or 'bind' in rel_l or 'interact' in rel_l) and ('drug' in h.lower() or 'drugbank' in h.lower() or 'chem' in h.lower()):
            # drug -> gene
            if h in ent2id and t in ent2id:
                drug_to_genes[ent2id[h]].add(ent2id[t])
        # gene -> disease edges: association
        if ('disease' in t.lower() or 'doid' in t.lower() or 'disease' in rel_l or 'associate' in rel_l):
            if h in ent2id and t in ent2id:
                gene_to_diseases[ent2id[h]].add(ent2id[t])

    # Now count 2-hop connections
    pair_counts = {}
    for d_id, genes in drug_to_genes.items():
        for g in genes:
            diseases = gene_to_diseases.get(g, [])
            for dis in diseases:
                pair_counts[(d_id, dis)] = pair_counts.get((d_id, dis), 0) + 1

    # Normalize counts to [0,1] by dividing by max_count (or use log-scaling)
    if len(pair_counts) == 0:
        return {}, {}
    max_count = max(pair_counts.values())
    pair_support = {k: v / max_count for k, v in pair_counts.items()}
    return pair_counts, pair_support
