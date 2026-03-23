#!/usr/bin/env python3
"""Quick CLIP pilot: check if CLIPScore correlates with judge error."""

import pandas as pd
import numpy as np
import json
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")

# --- Load our new CoT results so far ---
dfs = []
for i in range(2):
    try:
        dfs.append(pd.read_csv(f'results/v2/features_s2_p{i}.csv'))
    except:
        pass
df = pd.concat(dfs, ignore_index=True)
df = df[df.gt_score >= 1].reset_index(drop=True)
print(f"New CoT samples available: {len(df)}")

# --- Load dataset metadata ---
score_data = []
with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
    for line in f:
        score_data.append(json.loads(line))

meta = {}
for i, item in enumerate(score_data):
    meta[i] = {
        'image_path': f"data/mllm_judge/raw/Dataset/image/{item['image_path']}",
        'question': item['instruction'],
        'answer': item['answer'],
        'original_dataset': item.get('original_dataset', 'unknown'),
        'model_name': item.get('name', 'unknown'),
    }

# --- Load CLIP (use GPU 0, small model) ---
print("Loading CLIP model...")
device = "cuda:0"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# --- Compute CLIPScore for available samples ---
clip_img_ans = []
clip_img_q = []
orig_datasets = []
judge_errors = []
gt_scores = []
parsed_scores = []
sample_ids = []
s2_entropies = []

print("Computing CLIPScores...")
computed = 0
for idx, row in df.iterrows():
    sid = int(row['sample_id'])
    if sid not in meta:
        continue

    m = meta[sid]
    try:
        image = Image.open(m['image_path']).convert("RGB")
    except Exception as e:
        continue

    # Truncate text to 77 tokens (CLIP limit)
    ans_text = m['answer'][:300]
    q_text = m['question'][:300]

    # CLIPScore(image, answer)
    inputs_ans = clip_processor(text=[ans_text], images=image, return_tensors="pt",
                                padding=True, truncation=True).to(device)
    with torch.no_grad():
        out_ans = clip_model(**inputs_ans)
        score_img_ans = out_ans.logits_per_image.item()

    # CLIPScore(image, question)
    inputs_q = clip_processor(text=[q_text], images=image, return_tensors="pt",
                              padding=True, truncation=True).to(device)
    with torch.no_grad():
        out_q = clip_model(**inputs_q)
        score_img_q = out_q.logits_per_image.item()

    # Judge error
    error = abs(row['parsed_score'] - row['gt_score'])

    # S2 entropy (textual uncertainty)
    s2 = np.array([row['s2_lp1'], row['s2_lp2'], row['s2_lp3'], row['s2_lp4'], row['s2_lp5']])
    probs = softmax(s2)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    clip_img_ans.append(score_img_ans)
    clip_img_q.append(score_img_q)
    orig_datasets.append(m['original_dataset'])
    judge_errors.append(error)
    gt_scores.append(row['gt_score'])
    parsed_scores.append(row['parsed_score'])
    sample_ids.append(sid)
    s2_entropies.append(entropy)
    computed += 1

    if computed % 100 == 0:
        print(f"  {computed} samples done...")

clip_img_ans = np.array(clip_img_ans)
clip_img_q = np.array(clip_img_q)
judge_errors = np.array(judge_errors)
s2_entropies = np.array(s2_entropies)
orig_datasets = np.array(orig_datasets)

print(f"\nComputed CLIPScores for {len(clip_img_ans)} samples")

# --- Correlation Analysis ---
print("\n" + "=" * 60)
print("CORRELATION: Feature vs Judge Error |parsed - GT|")
print("=" * 60)
p1, pv1 = pearsonr(clip_img_ans, judge_errors)
s1, sv1 = spearmanr(clip_img_ans, judge_errors)
print(f"CLIPScore(image, answer) vs |error|:   Pearson={p1:.4f} (p={pv1:.4f}), Spearman={s1:.4f}")

p2, pv2 = pearsonr(clip_img_q, judge_errors)
s2r, sv2 = spearmanr(clip_img_q, judge_errors)
print(f"CLIPScore(image, question) vs |error|:  Pearson={p2:.4f} (p={pv2:.4f}), Spearman={s2r:.4f}")

p3, pv3 = pearsonr(s2_entropies, judge_errors)
s3, sv3 = spearmanr(s2_entropies, judge_errors)
print(f"S2 entropy (textual uncert) vs |error|: Pearson={p3:.4f} (p={pv3:.4f}), Spearman={s3:.4f}")

combined = clip_img_ans - clip_img_q
p4, pv4 = pearsonr(combined, judge_errors)
s4, sv4 = spearmanr(combined, judge_errors)
print(f"CLIP(ans) - CLIP(q) vs |error|:         Pearson={p4:.4f} (p={pv4:.4f}), Spearman={s4:.4f}")

# Also check: does CLIPScore correlate with GT score itself?
print("\n" + "=" * 60)
print("CORRELATION: CLIPScore vs GT Score (does visual grounding predict quality?)")
print("=" * 60)
gt = np.array(gt_scores)
p5, pv5 = pearsonr(clip_img_ans, gt)
print(f"CLIPScore(image, answer) vs GT:         Pearson={p5:.4f} (p={pv5:.4f})")
p6, pv6 = pearsonr(clip_img_q, gt)
print(f"CLIPScore(image, question) vs GT:        Pearson={p6:.4f} (p={pv6:.4f})")

# --- Per-Dataset Analysis ---
print("\n" + "=" * 60)
print("PER-DATASET BREAKDOWN")
print("=" * 60)
print(f"{'Dataset':20s} {'N':>5s} {'CLIP_ans':>9s} {'CLIP_q':>9s} {'MeanErr':>8s} {'S2_ent':>7s} {'Acc':>6s}")
print("-" * 75)

unique_ds = sorted(set(orig_datasets))
ds_stats = []
for ds in unique_ds:
    mask = orig_datasets == ds
    n = mask.sum()
    if n < 5:
        continue
    ca = clip_img_ans[mask].mean()
    cq = clip_img_q[mask].mean()
    err = judge_errors[mask].mean()
    ent = s2_entropies[mask].mean()
    acc = (judge_errors[mask] == 0).mean()
    print(f"{ds:20s} {n:5d} {ca:9.2f} {cq:9.2f} {err:8.3f} {ent:7.3f} {acc:6.3f}")
    ds_stats.append((ds, n, ca, cq, err, ent, acc))

# --- Quadrant Analysis ---
print("\n" + "=" * 60)
print("QUADRANT ANALYSIS: Textual Uncertainty vs Visual Grounding")
print("=" * 60)
med_ent = np.median(s2_entropies)
med_clip = np.median(clip_img_ans)
print(f"Median S2 entropy: {med_ent:.4f}")
print(f"Median CLIPScore(image,answer): {med_clip:.2f}")

q1 = (s2_entropies < med_ent) & (clip_img_ans >= med_clip)
q2 = (s2_entropies >= med_ent) & (clip_img_ans >= med_clip)
q3 = (s2_entropies < med_ent) & (clip_img_ans < med_clip)
q4 = (s2_entropies >= med_ent) & (clip_img_ans < med_clip)

for label, mask in [
    ("Q1: TextConf + VisGrounded  (easy)", q1),
    ("Q2: TextUncert + VisGrounded       ", q2),
    ("Q3: TextConf + VisUngrounded (trap)", q3),
    ("Q4: TextUncert + VisUngrounded (hard)", q4),
]:
    n = mask.sum()
    err = judge_errors[mask].mean() if n > 0 else 0
    acc = (judge_errors[mask] == 0).mean() if n > 0 else 0
    print(f"  {label}: N={n:4d}, MeanError={err:.3f}, Accuracy={acc:.3f}")

print("\nKey insight:")
print("  Q1 (easy): Judge confident AND answer grounded → should have smallest intervals")
print("  Q3 (trap): Judge confident BUT answer NOT grounded → judge may be wrong despite confidence")
print("  Q4 (hard): Both uncertain → needs widest intervals")
print("  If Q3 error >> Q1 error, CLIPScore adds info BEYOND what logprobs capture")

# Cleanup
del clip_model
torch.cuda.empty_cache()
