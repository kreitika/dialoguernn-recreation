
import torch
from sklearn.metrics import f1_score, accuracy_score

def masked_nll_loss(log_probs, labels, umask, class_weights=None):
    """
    log_probs: (seq_len, batch, num_classes)
    labels:    (batch, seq_len)
    umask:     (batch, seq_len)  -- 1 for real utterance, 0 for padding
    class_weights: optional (num_classes,) tensor to upweight rare classes
    """
    seq_len, batch_size, num_classes = log_probs.size()
    log_probs = log_probs.transpose(0, 1).contiguous().view(-1, num_classes)
    labels = labels.contiguous().view(-1)
    umask = umask.contiguous().view(-1)

    loss_fn = torch.nn.NLLLoss(weight=class_weights, reduction="none")
    losses = loss_fn(log_probs, labels)
    losses = losses * umask
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