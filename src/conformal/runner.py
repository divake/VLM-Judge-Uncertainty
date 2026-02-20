"""Conformal prediction runner supporting multiple methods.

Supported methods:
    - R2CCP: Neural network-based conformal regression
    - CQR: Conformalized Quantile Regression (mapie)
    - OrdinalAPS: Ordinal Adaptive Prediction Sets (classification-based, S2 only)

Adapted from conformal_predictors/ scripts in sibling LLM repo.
"""

import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path for importing conformal_predictors
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from conformal_predictors.interval_processing import range_modification


def merge_intervals(sample_intervals):
    """Merge multiple interval components into a single [low, high].

    From R2CCP_rancom.py: R2CCP may return disjoint intervals per sample.
    We take the envelope (min low, max high).
    """
    if not sample_intervals:
        return (1, 5)
    lows = [low for low, high in sample_intervals]
    highs = [high for low, high in sample_intervals]
    return (min(lows), max(highs))


class ConformalRunner:
    """Run conformal prediction over multiple seeds with multiple methods.

    Follows the pattern from the sibling LLM repo:
        1. Load feature CSV -> X, y
        2. For each seed: split -> fit -> intervals -> metrics
        3. Aggregate: mean +/- std over seeds
    """

    SUPPORTED_METHODS = ["R2CCP", "CQR", "OrdinalAPS"]

    def __init__(self, feature_csv: str, method: str = "R2CCP",
                 alpha: float = 0.10, num_seeds: int = 100,
                 test_size: float = 0.5, output_dir: str = None,
                 score_range: tuple = (1, 5)):
        self.feature_csv = feature_csv
        self.method = method
        self.alpha = alpha
        self.num_seeds = num_seeds
        self.test_size = test_size
        self.output_dir = output_dir
        self.score_range = score_range

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def run(self, feature_cols: list = None) -> dict:
        """Run conformal prediction experiments over multiple seeds."""
        torch.set_float32_matmul_precision('medium')

        df = pd.read_csv(self.feature_csv)
        print(f"[ConformalRunner] Loaded {len(df)} samples, {df.shape[1]} columns")

        if feature_cols is None:
            exclude = {"sample_id", "gt_score"}
            feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols]
        y = df["gt_score"]

        print(f"[ConformalRunner] Method: {self.method}")
        print(f"[ConformalRunner] Features ({len(feature_cols)}): {feature_cols}")
        print(f"[ConformalRunner] Running {self.num_seeds} seeds with alpha={self.alpha}")

        widths = []
        coverages = []

        for seed_idx in tqdm(range(self.num_seeds), desc=f"Conformal seeds ({self.method})"):
            seed = seed_idx + 1
            try:
                w, c = self._run_single_seed(X, y, seed)
                widths.append(w)
                coverages.append(c)
            except Exception as e:
                print(f"\n[ConformalRunner] Seed {seed} failed: {e}")
                continue

        if not widths:
            raise RuntimeError("All seeds failed!")

        summary = {
            "method": self.method,
            "alpha": self.alpha,
            "num_seeds": len(widths),
            "feature_cols": feature_cols,
            "num_samples": len(df),
            "coverage_mean": float(np.mean(coverages)),
            "coverage_std": float(np.std(coverages)),
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)),
            "coverage_all": coverages,
            "width_all": widths,
        }

        print(f"\n[ConformalRunner] Results over {len(widths)} seeds:")
        print(f"  Coverage: {summary['coverage_mean']:.4f} +/- {summary['coverage_std']:.4f}")
        print(f"  Width:    {summary['width_mean']:.4f} +/- {summary['width_std']:.4f}")

        if self.output_dir:
            summary_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved to {summary_path}")

        return summary

    def _run_single_seed(self, X: pd.DataFrame, y: pd.Series,
                         seed: int) -> tuple:
        """Dispatch to method-specific seed runner."""
        if self.method == "R2CCP":
            return self._run_r2ccp(X, y, seed)
        elif self.method == "CQR":
            return self._run_cqr(X, y, seed)
        elif self.method == "OrdinalAPS":
            return self._run_ordinal_aps(X, y, seed)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _run_r2ccp(self, X: pd.DataFrame, y: pd.Series, seed: int) -> tuple:
        """R2CCP: neural network conformal regression."""
        from R2CCP.main import R2CCP

        random.seed(seed)
        np.random.seed(seed)

        X_np = X.to_numpy().astype(np.float32)
        y_np = y.to_numpy().astype(np.float32)

        X_cal, X_test, y_cal, y_test = train_test_split(
            X_np, y_np, test_size=self.test_size, random_state=seed
        )

        model_path = os.path.join(
            self.output_dir or "model_paths", "model_save_destination.pth"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if os.path.exists(model_path):
            os.remove(model_path)

        model = R2CCP({
            'model_path': model_path,
            'max_epochs': 100,
            'alpha': self.alpha,
        })
        model.fit(X_cal, y_cal.flatten())

        intervals = model.get_intervals(X_test)
        intervals = [merge_intervals(si) for si in intervals]

        lows = np.array([iv[0] for iv in intervals])
        highs = np.array([iv[1] for iv in intervals])
        lows, highs = range_modification(lows, highs, *self.score_range)

        self._save_seed_results(seed, lows, highs, y_test)
        width, coverage = self._compute_metrics(lows, highs, y_test)

        del model
        torch.cuda.empty_cache()
        return width, coverage

    def _run_cqr(self, X: pd.DataFrame, y: pd.Series, seed: int) -> tuple:
        """CQR: Conformalized Quantile Regression via mapie."""
        from sklearn.ensemble import GradientBoostingRegressor
        from mapie.quantile_regression import MapieQuantileRegressor

        random.seed(seed)
        np.random.seed(seed)

        X_np = X.to_numpy().astype(np.float32)
        y_np = y.to_numpy().astype(np.float32)

        X_cal, X_test, y_cal, y_test = train_test_split(
            X_np, y_np, test_size=self.test_size, random_state=seed
        )

        gb_reg = GradientBoostingRegressor(loss="quantile", random_state=seed)
        mapie_qr = MapieQuantileRegressor(estimator=gb_reg, alpha=self.alpha)
        mapie_qr.fit(X_cal, y_cal, random_state=seed)

        # Use asymmetric intervals (more informative)
        y_pred, y_pis = mapie_qr.predict(X_test, symmetry=False)
        y_pis = np.clip(y_pis, *self.score_range)
        lows = y_pis[:, 0].ravel()
        highs = y_pis[:, 1].ravel()

        self._save_seed_results(seed, lows, highs, y_test)
        return self._compute_metrics(lows, highs, y_test)

    def _run_ordinal_aps(self, X: pd.DataFrame, y: pd.Series,
                         seed: int) -> tuple:
        """OrdinalAPS: ordinal classification conformal prediction.

        NOTE: Requires X to be the 5-dim Signal 2 (score logprobs).
        Adapted from conformal_predictors/OrdinalAPS_random.py.
        """
        from scipy.interpolate import interp1d

        random.seed(seed)
        np.random.seed(seed)

        X_np = X.to_numpy().astype(np.float32)
        y_np = y.to_numpy().astype(np.float32)

        # OrdinalAPS expects 5-column logit input
        if X_np.shape[1] != 5:
            raise ValueError(
                f"OrdinalAPS requires exactly 5 feature columns (S2 logprobs), "
                f"got {X_np.shape[1]}. Use feature_cols=['s2_lp1',...,'s2_lp5']."
            )

        # Map y from 1-5 to 13-class indices: y*3-3 -> {0,3,6,9,12}
        y_mapped = (y_np * 3 - 3).astype(int)

        # Interpolate 5 logits to 13 classes
        n = X_np.shape[0]
        m = 13
        new_x = np.zeros((n, m))
        orig_idx = np.linspace(3, 15, 5) - 3
        target_idx = np.linspace(3, 15, m) - 3
        for i in range(n):
            f = interp1d(orig_idx, X_np[i, :], kind='linear')
            new_x[i, :] = f(target_idx)

        # Softmax
        exp_x = np.exp(new_x - np.max(new_x, axis=1, keepdims=True))
        new_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        X_cal, X_test, y_cal, y_test = train_test_split(
            new_x, y_mapped, test_size=self.test_size, random_state=seed
        )

        # Calibrate qhat
        qhat = self._ordinal_aps_get_qhat(X_cal, y_cal)

        # Get prediction sets
        pred_sets = self._ordinal_aps_predict(X_test, qhat)

        # Convert prediction sets to intervals
        lows_list = []
        highs_list = []
        for ps in pred_sets:
            indices = np.where(ps)[0]
            if len(indices) > 0:
                lows_list.append((indices.min() + 3) / 3)
                highs_list.append((indices.max() + 3) / 3)
            else:
                lows_list.append(self.score_range[0])
                highs_list.append(self.score_range[1])

        lows = np.array(lows_list)
        highs = np.array(highs_list)
        # Map y_test back to 1-5 scale
        y_test_real = y_test / 3 + 1

        self._save_seed_results(seed, lows, highs, y_test_real)
        return self._compute_metrics(lows, highs, y_test_real)

    def _ordinal_aps_get_qhat(self, cal_scores, cal_labels):
        """Calibrate qhat for OrdinalAPS via grid search."""
        n = cal_scores.shape[0]
        grid_size = 10000
        for q in np.linspace(1e-3, 1 - 1e-3, grid_size)[::-1]:
            sets = self._ordinal_aps_predict(cal_scores, q)
            covered = sets[np.arange(n), cal_labels]
            coverage = covered.mean()
            if coverage <= (np.ceil((n + 1) * (1 - self.alpha)) / n):
                return min(q + 1 / (grid_size - 1), 1.0 - 1e-6)
        return q

    @staticmethod
    def _ordinal_aps_predict(scores, qhat):
        """Build ordinal prediction sets by expanding from argmax."""
        n_samples, n_classes = scores.shape
        P = scores == scores.max(axis=1)[:, None]

        idx_incomplete = (scores * P.astype(float)).sum(axis=1) <= qhat
        max_iter = n_classes

        for _ in range(max_iter):
            if idx_incomplete.sum() == 0:
                break

            P_inc = P[idx_incomplete]
            scores_inc = scores[idx_incomplete]

            cumsum = P_inc.cumsum(axis=1)
            lower_edge = (P_inc > 0).argmax(axis=1)
            upper_edge = cumsum.argmax(axis=1)

            left_ok = (lower_edge - 1) >= 0
            right_ok = (upper_edge + 1) < n_classes

            expand_left = np.zeros(scores_inc.shape[0], dtype=bool)
            expand_left[~right_ok & left_ok] = True

            both = left_ok & right_ok
            if both.any():
                left_scores = scores_inc[
                    np.arange(scores_inc.shape[0])[both],
                    lower_edge[both] - 1
                ]
                right_scores = scores_inc[
                    np.arange(scores_inc.shape[0])[both],
                    upper_edge[both] + 1
                ]
                expand_left[both] = left_scores > right_scores

            valid_l = expand_left & left_ok
            P_inc[valid_l, lower_edge[valid_l] - 1] = True
            valid_r = (~expand_left) & right_ok
            P_inc[valid_r, upper_edge[valid_r] + 1] = True

            P[idx_incomplete] = P_inc
            idx_incomplete = (scores * P.astype(float)).sum(axis=1) <= qhat
        else:
            P[idx_incomplete] = True

        return P

    # --- Shared helpers ---

    def _save_seed_results(self, seed, lows, highs, y_test):
        """Save per-seed CSV with intervals and ground truth."""
        if self.output_dir:
            seed_df = pd.DataFrame({
                'low': lows, 'up': highs, 'y_test': y_test,
            })
            seed_path = os.path.join(self.output_dir, f"results_seed_{seed:03d}.csv")
            seed_df.to_csv(seed_path, index=False)

    @staticmethod
    def _compute_metrics(lows, highs, y_test):
        """Compute coverage and width from intervals."""
        in_interval = (lows <= y_test) & (y_test <= highs)
        coverage = float(np.mean(in_interval))
        width = float(np.mean(highs - lows))
        return width, coverage

    def run_ablation(self, signal_configs: dict = None) -> dict:
        """Run ablation study with different signal combinations."""
        if signal_configs is None:
            signal_configs = {
                "S2_only": ["s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5"],
                "S2+S3": ["s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5",
                          "s3_entropy", "s3_margin", "s3_std"],
                "S1+S2": ["s1_mean_ent", "s1_max_ent", "s1_min_conf", "s1_std_ent", "s1_ntok",
                          "s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5"],
                "S1+S2+S3": None,  # all features
            }

        results = {}
        for name, cols in signal_configs.items():
            print(f"\n{'='*60}")
            print(f"Ablation: {name}")
            print(f"{'='*60}")

            orig_dir = self.output_dir
            if self.output_dir:
                self.output_dir = os.path.join(orig_dir, name)
                os.makedirs(self.output_dir, exist_ok=True)

            results[name] = self.run(feature_cols=cols)
            self.output_dir = orig_dir

        return results
