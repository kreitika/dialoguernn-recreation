import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def masked_nll_loss(log_probs, labels, umask):
    """
    log_probs: (seq_len, batch, num_classes)
    labels:    (batch, seq_len)
    umask:     (batch, seq_len)  -- 1 for real utterance, 0 for padding
    """
    seq_len, batch_size, num_classes = log_probs.size()
    log_probs = log_probs.transpose(0, 1).contiguous().view(-1, num_classes)  # (batch*seq_len, C)
    labels = labels.contiguous().view(-1)                                     # (batch*seq_len,)
    umask = umask.contiguous().view(-1)                                       # (batch*seq_len,)

    loss_fn = torch.nn.NLLLoss(reduction="none")
    losses = loss_fn(log_probs, labels)          # (batch*seq_len,)
    losses = losses * umask                      # zero out padded positions
    return losses.sum() / umask.sum()


def masked_accuracy_f1(log_probs, labels, umask):
    """
    Returns (accuracy, weighted_f1) computed only over real (non-padded) utterances.
    """
    seq_len, batch_size, num_classes = log_probs.size()
    preds = log_probs.transpose(0, 1).argmax(dim=2).contiguous().view(-1).cpu().numpy()
    labels_flat = labels.contiguous().view(-1).cpu().numpy()
    mask_flat = umask.contiguous().view(-1).cpu().numpy().astype(bool)

    preds = preds[mask_flat]
    labels_flat = labels_flat[mask_flat]

    acc = accuracy_score(labels_flat, preds)
    f1 = f1_score(labels_flat, preds, average="weighted", zero_division=0)
    return acc, f1