
import json
import pickle
import torch
from dataloader import IEMOCAPDataset
from model import DialogueRNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PKL_PATH = "data/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw.pkl"


def get_dialoguernn_predictions():
    test_set = IEMOCAPDataset(train=False)
    model = DialogueRNN().to(DEVICE)
    model.load_state_dict(torch.load("model_outputs/best_model.pt", map_location=DEVICE))
    model.eval()

    predictions = {}
    with torch.no_grad():
        for i in range(len(test_set)):
            text, speakers, labels, umask, vid = test_set[i]
            text = text.unsqueeze(1).to(DEVICE)
            speakers = speakers.unsqueeze(1).to(DEVICE)

            log_probs = model(text, speakers)
            preds = log_probs.argmax(dim=2).squeeze(1).cpu().tolist()
            predictions[vid] = preds

    return predictions


def find_shift_points(speaker_list, true_labels):
    last_label_by_speaker = {}
    shifts = []
    for spk, label in zip(speaker_list, true_labels):
        prev = last_label_by_speaker.get(spk)
        shifts.append(prev is not None and prev != label)
        last_label_by_speaker[spk] = label
    return shifts


def accuracy_at(preds, trues, mask):
    correct = sum(1 for p, t, m in zip(preds, trues, mask) if m and p == t)
    total = sum(mask)
    return correct / total if total > 0 else None


def main():
    with open(PKL_PATH, "rb") as f:
        (videoIDs, videoSpeakers, videoLabels, videoText,
         videoAudio, videoVisual, videoSentence, trainVid, testVid) = pickle.load(f, encoding="latin1")

    with open("model_outputs/gpt_predictions.json") as f:
        gpt_preds = json.load(f)

    rnn_preds = get_dialoguernn_predictions()

    rnn_shift_acc, rnn_noshift_acc = [], []
    gpt_shift_acc, gpt_noshift_acc = [], []

    for vid in list(testVid):
        true_labels = videoLabels[vid]
        speakers = videoSpeakers[vid]
        shifts = find_shift_points(speakers, true_labels)
        noshifts = [not s for s in shifts]

        rnn_p = rnn_preds[vid]
        gpt_p = gpt_preds[vid]

        a = accuracy_at(rnn_p, true_labels, shifts)
        b = accuracy_at(rnn_p, true_labels, noshifts)
        if a is not None: rnn_shift_acc.append(a)
        if b is not None: rnn_noshift_acc.append(b)

        a = accuracy_at(gpt_p, true_labels, shifts)
        b = accuracy_at(gpt_p, true_labels, noshifts)
        if a is not None: gpt_shift_acc.append(a)
        if b is not None: gpt_noshift_acc.append(b)

    def avg(lst):
        return sum(lst) / len(lst) * 100 if lst else 0.0

    print("=== Emotional-shift error analysis (extends paper Section 5.5) ===\n")
    print(f"{'Model':<15} {'Shift Acc':<12} {'No-Shift Acc':<12}")
    print(f"{'DialogueRNN':<15} {avg(rnn_shift_acc):<12.2f} {avg(rnn_noshift_acc):<12.2f}")
    print(f"{'GPT-4o-mini':<15} {avg(gpt_shift_acc):<12.2f} {avg(gpt_noshift_acc):<12.2f}")

    print(f"\nPaper reports (Section 5.5, vanilla DialogueRNN): "
          f"47.5% at shifts, 69.2% at no-shifts.")

    results = {
        "dialoguernn_shift_acc": avg(rnn_shift_acc),
        "dialoguernn_noshift_acc": avg(rnn_noshift_acc),
        "gpt_shift_acc": avg(gpt_shift_acc),
        "gpt_noshift_acc": avg(gpt_noshift_acc),
    }
    with open("model_outputs/error_analysis.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
