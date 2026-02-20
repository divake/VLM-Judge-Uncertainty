"""Extract 3-signal feature vectors from saved actor + judge inference outputs.

Produces a 13-dim feature vector per sample:
    Signal 1 (5-dim): Actor's generation confidence
        [mean_entropy, max_entropy, min_confidence, std_entropy, n_tokens]
    Signal 2 (5-dim): Judge's score-token logprobs
        [log_P("1"), log_P("2"), log_P("3"), log_P("4"), log_P("5")]
    Signal 3 (3-dim): Judge's self-uncertainty (derived from Signal 2)
        [entropy_of_score_dist, top1_margin, score_probs_std]

Score token extraction adapted from:
    extract_logits_REFERENCE.py (backward scan, FALLBACK_LP)
"""

import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils.io import load_json, load_pt, get_sample_paths

FALLBACK_LP = math.log(1e-5)  # from extract_logits_REFERENCE.py
SCORE_TOKENS = ["1", "2", "3", "4", "5"]


class SignalExtractor:
    """Extract 3-signal feature vectors from saved inference outputs.

    Reads per-sample JSON + PT files from actor and judge output directories,
    computes the 13-dim feature vector, and outputs a CSV.
    """

    def __init__(self, actor_output_dir: str, judge_output_dir: str,
                 dataset=None):
        self.actor_output_dir = actor_output_dir
        self.judge_output_dir = judge_output_dir
        self.dataset = dataset

    def extract_all(self, output_csv: str = None) -> pd.DataFrame:
        """Extract features from all available samples.

        Returns:
            DataFrame with columns:
                s1_mean_ent, s1_max_ent, s1_min_conf, s1_std_ent, s1_ntok,
                s2_lp1, s2_lp2, s2_lp3, s2_lp4, s2_lp5,
                s3_entropy, s3_margin, s3_std,
                gt_score
        """
        # Find all sample IDs that have both actor and judge outputs
        sample_ids = self._find_complete_samples()
        print(f"[SignalExtractor] Found {len(sample_ids)} complete samples")

        rows = []
        skipped = 0

        for sid in tqdm(sample_ids, desc="Extracting signals"):
            try:
                signal_1 = self._extract_signal_1(sid)
                signal_2 = self._extract_signal_2(sid)
                signal_3 = self._compute_signal_3(signal_2)
                gt_score = self._get_gt_score(sid)

                row = {
                    "sample_id": sid,
                    # Signal 1: Actor confidence (5-dim)
                    "s1_mean_ent": signal_1[0],
                    "s1_max_ent": signal_1[1],
                    "s1_min_conf": signal_1[2],
                    "s1_std_ent": signal_1[3],
                    "s1_ntok": signal_1[4],
                    # Signal 2: Judge score logprobs (5-dim)
                    "s2_lp1": signal_2[0],
                    "s2_lp2": signal_2[1],
                    "s2_lp3": signal_2[2],
                    "s2_lp4": signal_2[3],
                    "s2_lp5": signal_2[4],
                    # Signal 3: Judge self-uncertainty (3-dim)
                    "s3_entropy": signal_3[0],
                    "s3_margin": signal_3[1],
                    "s3_std": signal_3[2],
                    # Ground truth
                    "gt_score": gt_score,
                }
                rows.append(row)
            except Exception as e:
                print(f"\n[SignalExtractor] Skipping sample {sid}: {e}")
                skipped += 1

        df = pd.DataFrame(rows)
        print(f"[SignalExtractor] Extracted {len(df)} samples ({skipped} skipped)")

        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"[SignalExtractor] Saved to {output_csv}")

        return df

    def _find_complete_samples(self) -> List[int]:
        """Find sample IDs that have both actor and judge outputs."""
        actor_files = set()
        judge_files = set()

        if os.path.exists(self.actor_output_dir):
            for f in os.listdir(self.actor_output_dir):
                if f.startswith("sample_") and f.endswith(".json"):
                    sid = int(f.replace("sample_", "").replace(".json", ""))
                    actor_files.add(sid)

        if os.path.exists(self.judge_output_dir):
            for f in os.listdir(self.judge_output_dir):
                if f.startswith("sample_") and f.endswith(".json"):
                    sid = int(f.replace("sample_", "").replace(".json", ""))
                    judge_files.add(sid)

        complete = sorted(actor_files & judge_files)
        return complete

    def _extract_signal_1(self, sample_id: int) -> Tuple[float, ...]:
        """Extract Signal 1: Actor's generation confidence (5-dim).

        Computes per-token entropy from full logits or top-k logprobs.
        Returns: (mean_ent, max_ent, min_conf, std_ent, n_tokens)
        """
        pt_path = get_sample_paths(self.actor_output_dir, sample_id)[1]
        json_path = get_sample_paths(self.actor_output_dir, sample_id)[0]

        pt_data = load_pt(pt_path)
        json_data = load_json(json_path)

        # Try full_scores first (more accurate), fall back to top_logprobs
        if "full_scores" in pt_data and pt_data["full_scores"] is not None:
            entropies, max_probs = self._entropy_from_full_scores(pt_data["full_scores"])
        else:
            entropies, max_probs = self._entropy_from_top_logprobs(pt_data["top_logprobs"])

        n_tokens = len(entropies)
        if n_tokens == 0:
            return (0.0, 0.0, 0.0, 0.0, 0)

        entropies = np.array(entropies)
        max_probs = np.array(max_probs)

        return (
            float(np.mean(entropies)),      # mean entropy
            float(np.max(entropies)),        # max entropy
            float(np.min(max_probs)),        # min confidence (max prob at worst token)
            float(np.std(entropies)),        # std of entropy
            n_tokens,                        # number of generated tokens
        )

    @staticmethod
    def _entropy_from_full_scores(full_scores: torch.Tensor):
        """Compute per-token entropy from full logit tensors.

        Args:
            full_scores: (num_steps, vocab_size) raw logits

        Returns:
            (entropies, max_probs) lists
        """
        probs = torch.softmax(full_scores.float(), dim=-1)
        # Entropy: -sum(p * log(p)), with small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        max_prob_per_token = probs.max(dim=-1).values

        return entropy_per_token.tolist(), max_prob_per_token.tolist()

    @staticmethod
    def _entropy_from_top_logprobs(top_logprobs: List[dict]):
        """Approximate per-token entropy from top-k logprobs.

        Less accurate than full scores but works when full_scores not saved.
        """
        entropies = []
        max_probs = []

        for step_dict in top_logprobs:
            if not step_dict:
                continue
            lps = list(step_dict.values())
            probs = np.exp(lps)
            # Normalize (top-k doesn't sum to 1)
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
            max_probs.append(float(np.max(probs)))

        return entropies, max_probs

    def _extract_signal_2(self, sample_id: int) -> Tuple[float, ...]:
        """Extract Signal 2: Judge's score-token logprobs (5-dim).

        Strategy (in priority order):
            1. Find "Score" + ":" + digit pattern in tokens (most reliable)
            2. Find "score" ... digit pattern within a small window
            3. Backward scan for last digit in {1,2,3,4,5} (fallback)

        Returns: (lp_1, lp_2, lp_3, lp_4, lp_5)
        """
        json_path = get_sample_paths(self.judge_output_dir, sample_id)[0]
        pt_path = get_sample_paths(self.judge_output_dir, sample_id)[1]

        json_data = load_json(json_path)
        pt_data = load_pt(pt_path)

        tokens = json_data["tokens"]
        top_logprobs = pt_data["top_logprobs"]

        score_idx = self._find_score_token(tokens)

        if score_idx is None:
            print(f"[Signal2] WARNING: No score token found for sample {sample_id}. "
                  f"Last 10 tokens: {tokens[-10:]}")
            return tuple([FALLBACK_LP] * 5)

        # Extract logprobs at score position
        lp_dict = top_logprobs[score_idx]

        # Try exact match first, then partial match for different tokenizers
        signal_2 = []
        for s in SCORE_TOKENS:
            lp = lp_dict.get(s, None)
            if lp is None:
                # Try common tokenizer variants: "▁1", "Ġ1", " 1"
                for variant in [f"▁{s}", f"Ġ{s}", f" {s}", f"\t{s}"]:
                    if variant in lp_dict:
                        lp = lp_dict[variant]
                        break
            if lp is None:
                lp = FALLBACK_LP
            signal_2.append(lp)

        return tuple(signal_2)

    @staticmethod
    def _compute_signal_3(signal_2: Tuple[float, ...]) -> Tuple[float, float, float]:
        """Compute Signal 3: Judge's self-uncertainty (3-dim).

        Derived from Signal 2 (no additional model call):
            entropy: -sum(p * log(p)) over score distribution
            margin: top1_prob - top2_prob
            std: std of score probabilities
        """
        logprobs = np.array(signal_2)
        probs = np.exp(logprobs)
        # Normalize to valid probability distribution
        probs = probs / probs.sum()

        # Entropy of score distribution
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Margin: difference between top-1 and top-2 probabilities
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]

        # Std of probabilities
        std = float(np.std(probs))

        return (float(entropy), float(margin), std)

    @staticmethod
    def _find_score_token(tokens: List[str]) -> int:
        """Find the score digit token index using multiple strategies.

        Priority:
            1. "Score" + ":" + digit pattern (scanning backward for last match)
            2. "score" + nearby digit within 3-token window
            3. Plain backward scan for last digit (fallback)
        """
        stripped = [t.strip().lower() for t in tokens]

        # Strategy 1: Find "Score:" + digit pattern (backward scan)
        for i in range(len(tokens) - 1, -1, -1):
            if stripped[i] in SCORE_TOKENS:
                # Check if preceded by ":" and "score" within 3 tokens
                for j in range(max(0, i - 3), i):
                    if "score" in stripped[j]:
                        return i

        # Strategy 2: Find "score" keyword then look ahead for digit
        for i in range(len(tokens) - 1, -1, -1):
            if "score" in stripped[i]:
                # Look ahead 1-4 tokens for a score digit
                for j in range(i + 1, min(len(tokens), i + 5)):
                    if stripped[j] in SCORE_TOKENS:
                        return j

        # Strategy 3: Plain backward scan (original approach from LLM repo)
        for i in range(len(tokens) - 1, -1, -1):
            if stripped[i] in SCORE_TOKENS:
                return i

        return None

    def _get_gt_score(self, sample_id: int) -> float:
        """Get ground truth score for a sample."""
        # Try from dataset first
        if self.dataset is not None:
            return self.dataset.get_gt_score(sample_id)

        # Fall back to stored value in judge/actor JSON
        json_path = get_sample_paths(self.judge_output_dir, sample_id)[0]
        json_data = load_json(json_path)
        gt = json_data.get("gt_score", None)
        if gt is not None:
            return float(gt)

        json_path = get_sample_paths(self.actor_output_dir, sample_id)[0]
        json_data = load_json(json_path)
        return float(json_data.get("gt_score", 0))
