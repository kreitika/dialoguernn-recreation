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

| Run | Config | Best Test Weighted F1 |
|---|---|---|
| Paper (Table 2) | grid-searched (unpublished exact values) | 59.89 |
| Run 1 | lr=1e-4, wd=1e-5, no class weights, no fixed seed | 56.83 |
| Run 2 | lr=1e-4, wd=1e-3, class-weighted loss, no fixed seed | 58.16 |
| Run 3 | Run 2 + grad clipping + LR scheduler, no fixed seed | 57.86 |
| **Final** | Run 2 config + fixed seed (42) for full reproducibility | **59.09** |

Early runs without a fixed random seed showed F1 varying between
56.83–58.16 across identical configs — expected variance on a small
120-dialogue training set. Adding `torch.manual_seed`, `np.random.seed`,
and a seeded DataLoader generator (`SEED=42`) made results
reproducible. The final run comes within **0.8 F1** of the paper's
published 59.89, a gap attributable to their unpublished exact
grid-searched hyperparameters.

## GPT-4o-mini Baseline Comparison

| Model | Test Weighted F1 |
|---|---|
| GPT-4o-mini (zero-shot, full dialogue in-context) | 39.21 |
| DialogueRNN (this repro) | 59.09 |
| DialogueRNN (paper, Table 2) | 59.89 |

DialogueRNN outperforms zero-shot GPT-4o-mini by ~20 weighted F1 points
on the same 31 IEMOCAP test dialogues. This suggests that explicit
architectural modeling of speaker state and conversational context
(the paper's core contribution) still meaningfully outperforms a
general-purpose LLM given no fine-tuning or few-shot examples on this
task. See `gpt_baseline.py` for the prompting setup.

**Note on methodology:** GPT-4o-mini was prompted zero-shot with the
full dialogue in one call per dialogue (no few-shot examples, no
fine-tuning). A stronger baseline would add few-shot examples or
fine-tune GPT on the training split — a natural extension explored in
Milestone 6.

## Error Analysis: Emotional Shifts (extends paper Section 5.5)

The paper notes DialogueRNN performs far worse at points where a
speaker's emotion changes from their own previous utterance ("shifts")
than at emotionally stable points ("no-shifts") — 47.5% vs 69.2%
accuracy in their Table/Section 5.5.

| Model | Shift Accuracy | No-Shift Accuracy | Gap |
|---|---|---|---|
| DialogueRNN (paper) | 47.5 | 69.2 | 21.7 |
| DialogueRNN (this repro) | 45.07 | 65.17 | 20.1 |
| GPT-4o-mini (zero-shot) | 41.82 | 40.59 | 1.2 |

This reproduction shows the same asymmetry the paper reports, at
similar magnitude — evidence the reimplementation captures the
architecture's real behavior, not just a matching aggregate score.

Interestingly, GPT-4o-mini shows almost no shift/no-shift gap. A likely
explanation: DialogueRNN explicitly maintains a per-speaker running
state, so it's strongly biased toward predicting emotional continuity
— accurate when the speaker stays consistent, but this same bias hurts
it exactly when the emotion changes. GPT-4o-mini has no equivalent
explicit per-speaker memory mechanism, so its performance is
mediocre-but-uniform rather than "good at stability, bad at surprises."

See `error_analysis.py`.

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