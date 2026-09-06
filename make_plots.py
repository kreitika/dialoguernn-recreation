
import json
import matplotlib.pyplot as plt

# F1 comparison bar chart
models = ["Paper\n(DialogueRNN)", "This repro\n(DialogueRNN)", "GPT-4o-mini\n(zero-shot)"]
f1_scores = [59.89, 59.09, 40.71]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(models, f1_scores, color=["#888888", "#4C72B0", "#DD8452"])
ax.set_ylabel("Weighted F1 (Test Set)")
ax.set_title("IEMOCAP Emotion Recognition: F1 Comparison")
ax.set_ylim(0, 70)
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{score:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("model_outputs/f1_comparison.png", dpi=150)
plt.close()

# Shift vs no-shift accuracy comparison
with open("model_outputs/error_analysis.json") as f:
    err = json.load(f)

labels = ["DialogueRNN", "GPT-4o-mini"]
shift_acc = [err["dialoguernn_shift_acc"], err["gpt_shift_acc"]]
noshift_acc = [err["dialoguernn_noshift_acc"], err["gpt_noshift_acc"]]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([i - width/2 for i in x], shift_acc, width, label="At emotional shift", color="#C44E52")
ax.bar([i + width/2 for i in x], noshift_acc, width, label="No shift (stable)", color="#55A868")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy at Emotional Shift Points\n(extends paper Section 5.5)")
ax.legend()
plt.tight_layout()
plt.savefig("model_outputs/shift_analysis.png", dpi=150)
plt.close()

print("Saved: model_outputs/f1_comparison.png, model_outputs/shift_analysis.png")
