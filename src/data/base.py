"""Abstract base class for all datasets."""

from abc import ABC, abstractmethod
import math


class BaseDataset(ABC):
    """Base dataset interface for VLM-as-a-Judge experiments.

    All datasets must implement these methods to be compatible
    with the actor/judge inference runners.
    """

    @abstractmethod
    def __len__(self):
        """Return total number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return a dict with keys: image_path, question, gt_score, metadata."""
        pass

    @abstractmethod
    def get_image_path(self, idx):
        """Return absolute path to the image for sample idx."""
        pass

    @abstractmethod
    def get_question(self, idx):
        """Return the question/prompt string for sample idx."""
        pass

    @abstractmethod
    def get_gt_score(self, idx):
        """Return the ground-truth score (already mapped to target scale)."""
        pass

    def map_score(self, raw_score, gt_scale, target_scale, method="bin"):
        """Map a raw score from gt_scale to target_scale.

        Args:
            raw_score: Original score value
            gt_scale: [low, high] of original scale (e.g., [1, 10])
            target_scale: [low, high] of target scale (e.g., [1, 5])
            method: "bin" for binning, "linear" for linear interpolation

        Returns:
            Mapped score in target_scale
        """
        src_low, src_high = gt_scale
        tgt_low, tgt_high = target_scale

        if method == "bin":
            # Bin mapping: divide source range into equal bins
            # e.g., 1-10 → 1-5: {1,2}→1, {3,4}→2, {5,6}→3, {7,8}→4, {9,10}→5
            n_bins = tgt_high - tgt_low + 1
            bin_size = (src_high - src_low + 1) / n_bins
            bin_idx = min(int((raw_score - src_low) / bin_size), n_bins - 1)
            return tgt_low + bin_idx
        elif method == "linear":
            # Linear interpolation
            normalized = (raw_score - src_low) / (src_high - src_low)
            return tgt_low + normalized * (tgt_high - tgt_low)
        else:
            raise ValueError(f"Unknown mapping method: {method}")
