
import os
import re
import pickle
import json
import time
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

LABEL_MAP = {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5}
VALID_TAGS = "|".join(LABEL_MAP.keys())

PKL_PATH = "data/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw.pkl"


def load_test_dialogues():
    with open(PKL_PATH, "rb") as f:
        (videoIDs, videoSpeakers, videoLabels, videoText,
         videoAudio, videoVisual, videoSentence, trainVid, testVid) = pickle.load(f, encoding="latin1")
    return videoSpeakers, videoLabels, videoSentence, list(testVid)


PROMPT_TEMPLATE = """You are an expert annotator for emotion recognition in conversations.
Label each numbered utterance below with exactly one emotion tag from this set:
hap, sad, neu, ang, exc, fru

Conversation ({n} utterances, numbered):
{conversation}

Respond with EXACTLY {n} lines, nothing else — no intro, no explanation, no markdown.
Each line must be in this exact format:
<number>: <tag>

Example of the expected format for 3 utterances:
1: neu
2: hap
3: fru
"""


def parse_lines(raw, expected_len):
    """
    Parses 'N: tag' lines. Returns a list of length expected_len,
    filling any missing/malformed line with 'neu'.
    """
    results = {}
    for line in raw.splitlines():
        m = re.match(rf'^\s*(\d+)\s*[:.]\s*({VALID_TAGS})\s*$', line.strip())
        if m:
            idx = int(m.group(1))
            tag = m.group(2)
            results[idx] = tag

    return [results.get(i + 1, "neu") for i in range(expected_len)]


def predict_dialogue(speakers, sentences):
    numbered = "\n".join(f"{i+1}. {spk}: {sent}" for i, (spk, sent) in enumerate(zip(speakers, sentences)))
    prompt = PROMPT_TEMPLATE.format(conversation=numbered, n=len(sentences))

    # generous but bounded token budget: ~6 tokens/line is plenty for "12: fru"
    token_budget = min(4096, 20 * len(sentences) + 100)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=token_budget,
    )
    raw = response.choices[0].message.content.strip()

    tags = parse_lines(raw, len(sentences))
    n_missing = sum(1 for t in tags if t == "neu") 
    return [LABEL_MAP[t] for t in tags]


def main():
    videoSpeakers, videoLabels, videoSentence, test_ids = load_test_dialogues()

    all_true, all_pred = [], []

    for i, vid in enumerate(test_ids):
        speakers = videoSpeakers[vid]
        sentences = videoSentence[vid]
        true_labels = videoLabels[vid]

        pred_labels = predict_dialogue(speakers, sentences)

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

        print(f"[{i+1}/{len(test_ids)}] dialogue {vid} done ({len(sentences)} utterances)")
        time.sleep(0.5)

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    print(f"\nGPT-4o-mini baseline — Test Accuracy: {acc*100:.2f}  Weighted F1: {f1*100:.2f}")
    print(f"DialogueRNN (this repro):            Weighted F1: 59.09")
    print(f"DialogueRNN (paper, Table 2):         Weighted F1: 59.89")

    with open("model_outputs/gpt_baseline_results.json", "w") as f:
        json.dump({"accuracy": acc, "weighted_f1": f1}, f, indent=2)


if __name__ == "__main__":
    main()
