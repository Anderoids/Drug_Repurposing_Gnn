import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def compute_classification_metrics(y_true, y_logits):
    y_prob = sigmoid(y_logits)
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
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        idx = np.where((y_prob >= bins[i]) & (y_prob < bins[i+1]))[0]
        if len(idx) == 0: continue
        acc = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece += (len(idx)/len(y_true)) * abs(acc - conf)
    return float(ece)

# approximate ranking metrics using one positive and sampled negatives per positive
def approx_mrr_hits(model, edge_index, edge_type, pos_triples, num_entities, negatives_per_pos=200, device='cpu'):
    """
    pos_triples: list of (h,r,t) numpy or tensor arrays for positive examples (test set)
    For each positive, sample many random negatives to estimate rank.
    Returns mean reciprocal rank and Hits@10 estimate.
    """
    model.eval()
    mrrs = []
    hits = 0
    with torch.no_grad():
        for (h,r,t) in pos_triples:
            h_t = torch.tensor([h], dtype=torch.long).to(device)
            r_t = torch.tensor([r], dtype=torch.long).to(device)
            t_pos = torch.tensor([t], dtype=torch.long).to(device)
            # sample negatives
            negs = np.random.choice(num_entities, size=negatives_per_pos, replace=False)
            negs_t = torch.tensor(negs, dtype=torch.long).to(device)
            # prepare batch: pos + negs
            heads = torch.cat([h_t.repeat(1), h_t.repeat(negs_t.shape[0])], dim=0)
            rels  = torch.cat([r_t, r_t.repeat(negs_t.shape[0])], dim=0)
            tails = torch.cat([t_pos, negs_t], dim=0)
            # compute scores depending on model type API
            # assume model.forward(edge_index, edge_type, heads, rels, tails) returns logits
            logits = model(edge_index.to(device), edge_type.to(device), heads, rels, tails)
            if isinstance(logits, tuple):
                logits = logits[0]  # handle PathAwareGNN returns (mu, logvar, x)
            scores = logits.cpu().numpy()
            pos_score = scores[0]
            neg_scores = scores[1:]
            rank = 1 + int((neg_scores > pos_score).sum())
            mrrs.append(1.0 / rank)
            if rank <= 10: hits += 1
    return float(np.mean(mrrs)), float(hits / len(pos_triples))

# MC dropout wrapper to get mean & variance predictions
def mc_dropout_predict(model, edge_index, edge_type, heads, rels, tails, mc_runs=30, device='cpu'):
    """
    model: should have dropout layers. We run model in training mode but with torch.no_grad().
    Returns mean_probs, epistemic_var (variance over runs)
    """
    model.train()  # activate dropout
    preds = []
    with torch.no_grad():
        for _ in range(mc_runs):
            out = model(edge_index.to(device), edge_type.to(device), heads.to(device), rels.to(device), tails.to(device))
            if isinstance(out, tuple):
                mu = out[0]
            else:
                mu = out
            preds.append(torch.sigmoid(mu).cpu().numpy())
    preds = np.stack(preds, axis=0)  # (mc_runs, batch)
    mean = preds.mean(axis=0)
    var = preds.var(axis=0)
    return mean, var

# Ensemble predict: average probabilities from saved checkpoints (list of models or states)
def ensemble_predict(models: list, edge_index, edge_type, heads, rels, tails, device='cpu'):
    probs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            out = m(edge_index.to(device), edge_type.to(device), heads.to(device), rels.to(device), tails.to(device))
            if isinstance(out, tuple):
                mu = out[0]
            else:
                mu = out
            probs.append(torch.sigmoid(mu).cpu().numpy())
    probs = np.stack(probs, axis=0)
    mean = probs.mean(axis=0)
    var = probs.var(axis=0)
    return mean, var
