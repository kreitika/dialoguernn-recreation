
import json
import pickle
import torch
import gradio as gr
from dataloader import IEMOCAPDataset
from model import DialogueRNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PKL_PATH = "data/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw.pkl"
LABEL_NAMES = {0: "happy", 1: "sad", 2: "neutral", 3: "angry", 4: "excited", 5: "frustrated"}

with open(PKL_PATH, "rb") as f:
    (videoIDs, videoSpeakers, videoLabels, videoText,
     videoAudio, videoVisual, videoSentence, trainVid, testVid) = pickle.load(f, encoding="latin1")

with open("model_outputs/gpt_predictions.json") as f:
    gpt_preds = json.load(f)

test_ids = sorted(list(testVid))

test_set = IEMOCAPDataset(train=False)
model = DialogueRNN().to(DEVICE)
model.load_state_dict(torch.load("model_outputs/best_model.pt", map_location=DEVICE))
model.eval()


def get_rnn_prediction(vid):
    idx = test_set.keys.index(vid)
    text, speakers, labels, umask, _ = test_set[idx]
    text = text.unsqueeze(1).to(DEVICE)
    speakers = speakers.unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        log_probs = model(text, speakers)
    return log_probs.argmax(dim=2).squeeze(1).cpu().tolist()


def show_dialogue(vid):
    speakers = videoSpeakers[vid]
    sentences = videoSentence[vid]
    true_labels = videoLabels[vid]
    rnn_pred = get_rnn_prediction(vid)
    gpt_pred = gpt_preds.get(vid, [2] * len(sentences))  # default neutral if missing

    rows = []
    for spk, sent, true, rnn, gpt in zip(speakers, sentences, true_labels, rnn_pred, gpt_pred):
        rows.append([
            spk,
            sent,
            LABEL_NAMES[true],
            LABEL_NAMES[rnn],
            LABEL_NAMES[gpt],
        ])
    return rows


with gr.Blocks(title="DialogueRNN vs GPT-4o-mini — Emotion Recognition") as demo:
    gr.Markdown("""
    # DialogueRNN vs GPT-4o-mini: Emotion Recognition in Conversations
    Pick an IEMOCAP test dialogue to see ground-truth emotions alongside
    predictions from a from-scratch DialogueRNN reimplementation and a
    zero-shot GPT-4o-mini baseline.

    *(Note: DialogueRNN runs only on the paper's pre-extracted 100-dim
    utterance features, so this demo shows real test dialogues rather
    than free-text input — a known scope limitation, see README.)*
    """)

    dropdown = gr.Dropdown(choices=test_ids, value=test_ids[0], label="Select a test dialogue")
    table = gr.Dataframe(
        headers=["Speaker", "Utterance", "Ground Truth", "DialogueRNN", "GPT-4o-mini"],
        wrap=True,
    )

    dropdown.change(fn=show_dialogue, inputs=dropdown, outputs=table)
    demo.load(fn=show_dialogue, inputs=dropdown, outputs=table)


if __name__ == "__main__":
    demo.launch()
