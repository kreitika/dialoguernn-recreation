# DialogueRNN Recreation

Reimplementation of *DialogueRNN: An Attentive RNN for Emotion Detection in
Conversations* (Majumder, Poria et al., AAAI 2019) from scratch in PyTorch,
plus a GPT-4o-mini baseline comparison on the same task.

## Status
🚧 In progress — building milestone by milestone.

## Setup
\`\`\`bash
conda create -n dialoguernn python=3.10 -y
conda activate dialoguernn
pip install -r requirements.txt
python download_data.py
\`\`\`

## Data

Uses the official IEMOCAP text features released by the DeCLaRe Lab
(same features used in the original paper's codebase), downloaded
automatically via `download_data.py`. 120 train / 31 test dialogues,
matching Table 1 of the paper exactly.

## Architecture

Three connected GRUs, following the paper's Section 3.3:
- **Global GRU** — encodes shared conversational context
- **Party GRU** (with attention over global states) — tracks each
  speaker's individual emotional state through the conversation
- **Emotion GRU** — decodes the final emotion representation from the
  speaker state, linking across speakers turn to turn

See `model.py` and `attention.py`.

## Results

| Model | Dataset | Best Test Weighted F1 |
|---|---|---|
| DialogueRNN (paper, Table 2) | IEMOCAP | 59.89 |
| DialogueRNN (this repro, run 1 — lr=1e-4, wd=1e-5, no class weights) | IEMOCAP | 56.83 |
| DialogueRNN (this repro, run 2 — lr=1e-4, wd=1e-3, class-weighted loss) | IEMOCAP | 58.16 |

Run 2 closes most of the gap to the paper. Remaining gap (~1.7 F1)
likely attributable to unpublished exact hyperparameters (paper used
grid search; specific values weren't itemized) and/or random seed
variance on a small dataset (120 train dialogues). Overfitting is
still present but less severe than run 1 — best epoch shifted from
~35 to ~22, and the interval between best test F1 and end-of-training
test F1 is smaller.

## Project structure
\`\`\`
dataloader.py       # loads IEMOCAP features, handles variable-length dialogue batching
model.py            # DialogueRNN architecture (3 GRUs)
attention.py         # attention module over global states (Eq. 2-4 in paper)
train.py            # training loop, masked loss, class weighting
evaluate.py          # masked NLL loss + weighted accuracy/F1
download_data.py    # reproducible data download script
check_data.py        # sanity check for dataloader
check_model.py       # sanity check for model forward pass
\`\`\`

## Reference

Majumder, N., Poria, S., Hazarika, D., Mihalcea, R., Gelbukh, A., &
Cambria, E. (2019). DialogueRNN: An Attentive RNN for Emotion Detection
in Conversations. *AAAI*.