
import torch
from collections import Counter
from torch.utils.data import DataLoader
from dataloader import IEMOCAPDataset
from model import DialogueRNN
from evaluate import masked_nll_loss, masked_accuracy_f1

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)


def run_epoch(model, loader, optimizer=None, class_weights=None):
    """If optimizer is None, runs in eval mode (no backprop)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_acc, total_f1, n_batches = 0, 0, 0, 0

    for text, speakers, labels, umask, vid in loader:
        text, speakers = text.to(DEVICE), speakers.to(DEVICE)
        labels, umask = labels.to(DEVICE), umask.to(DEVICE)

        if training:
            optimizer.zero_grad()

        log_probs = model(text, speakers)          # (seq_len, batch, C)
        loss = masked_nll_loss(log_probs, labels, umask, class_weights)

        if training:
            loss.backward()
            optimizer.step()

        acc, f1 = masked_accuracy_f1(log_probs, labels, umask)

        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches, total_f1 / n_batches


def compute_class_weights(dataset, num_classes=6):
    all_labels = []
    for i in range(len(dataset)):
        _, _, labels, _, _ = dataset[i]
        all_labels.extend(labels.tolist())
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = torch.FloatTensor([total / counts.get(c, 1) for c in range(num_classes)])
    weights = weights / weights.sum() * num_classes  # normalize so avg weight ~1
    return weights


def main():
    train_set = IEMOCAPDataset(train=True)
    test_set = IEMOCAPDataset(train=False)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=train_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=test_set.collate_fn)

    class_weights = compute_class_weights(train_set).to(DEVICE)
    print("Class weights:", class_weights)

    model = DialogueRNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    n_epochs = 60
    best_f1 = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, class_weights)
        test_loss, test_acc, test_f1 = run_epoch(model, test_loader, optimizer=None, class_weights=class_weights)

        print(f"Epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc*100:.2f} f1 {train_f1*100:.2f} "
              f"| test loss {test_loss:.3f} acc {test_acc*100:.2f} f1 {test_f1*100:.2f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), "model_outputs/best_model.pt")

    print(f"\nBest test weighted F1: {best_f1*100:.2f}  (paper reports 59.89 for vanilla DialogueRNN)")


if __name__ == "__main__":
    main()
