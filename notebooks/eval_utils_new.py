import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))


def compute_classification_metrics(y_true, y_prob):
    """AUROC, AUPRC"""
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float('nan')
    return auroc, auprc


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with n_bins equal-width bins"""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = np.where((y_prob >= bins[i]) & (y_prob < bins[i + 1]))[0]
        if len(idx) == 0: 
            continue
        acc = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece += (len(idx) / len(y_true)) * abs(acc - conf)
    return float(ece)


# Ranking Metrics (MRR, Hits@10)

def approx_mrr_hits(model, pos_triples, num_entities, edge_index=None, edge_type=None, negatives_per_pos=200, device="cpu"):
    """
    Estimate MRR and Hits@10 with random negative sampling.
    Supports both baselines (DistMult/ComplEx/GraphSAGE) and GNN models.
    """
    model.eval()
    mrrs = []
    hits = 0
    with torch.no_grad():
        for (h, r, t) in pos_triples:
            h_t = torch.tensor([h], dtype=torch.long).to(device)
            r_t = torch.tensor([r], dtype=torch.long).to(device)
            t_pos = torch.tensor([t], dtype=torch.long).to(device)

            # sample negatives
            negs = np.random.choice(num_entities, size=negatives_per_pos, replace=False)
            negs_t = torch.tensor(negs, dtype=torch.long).to(device)

            # batch: pos + negs
            heads = torch.cat([h_t, h_t.repeat(negs_t.shape[0])], dim=0)
            rels  = torch.cat([r_t, r_t.repeat(negs_t.shape[0])], dim=0)
            tails = torch.cat([t_pos, negs_t], dim=0)

            # forward pass (handle both API types)
            try:
                logits = model(heads, rels, tails)  # baseline models
            except TypeError:
                logits = model((heads, rels, tails), edge_index.to(device), edge_type.to(device))  # GNN models

            if isinstance(logits, tuple):  # if model returns (mu, logvar, etc.)
                logits = logits[0]

            scores = logits.cpu().numpy()
            pos_score = scores[0]
            neg_scores = scores[1:]

            rank = 1 + int((neg_scores > pos_score).sum())
            mrrs.append(1.0 / rank)
            if rank <= 10:
                hits += 1

    return float(np.mean(mrrs)), float(hits / len(pos_triples))


# MC Dropout

def mc_dropout_predict(model, heads, rels, tails, edge_index=None, edge_type=None, mc_runs=30, device="cpu"):
    """
    Returns mean_probs, epistemic_var from MC Dropout.
    Works with both baseline & GNN models.
    """
    model.train()  # activate dropout
    preds = []
    with torch.no_grad():
        for _ in range(mc_runs):
            try:
                out = model(heads.to(device), rels.to(device), tails.to(device))  # baseline
            except TypeError:
                out = model((heads, rels, tails), edge_index.to(device), edge_type.to(device))  # GNN

            if isinstance(out, tuple):
                out = out[0]
            preds.append(torch.sigmoid(out).cpu().numpy())

    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    var = preds.var(axis=0)
    return mean, var

# Ensemble

def ensemble_predict(models, heads, rels, tails, edge_index=None, edge_type=None, device="cpu"):
    """
    Ensemble prediction across models.
    """
    probs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            try:
                out = m(heads.to(device), rels.to(device), tails.to(device))
            except TypeError:
                out = m((heads, rels, tails), edge_index.to(device), edge_type.to(device))

            if isinstance(out, tuple):
                out = out[0]
            probs.append(torch.sigmoid(out).cpu().numpy())

    probs = np.stack(probs, axis=0)
    mean = probs.mean(axis=0)
    var = probs.var(axis=0)
    return mean, var
