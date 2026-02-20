"""MLLM-as-a-Judge dataset loader (scoring subset).

Paper: Chen et al., "MLLM-as-a-Judge", ICML 2024 Oral
Source: github.com/Dongping-Chen/MLLM-Judge
Scale: 1-5 (human GT) — already on target scale, no mapping needed

Actual data format (confirmed after download):
    File: data/mllm_judge/raw/Dataset/Benchmark/score.jsonl
    Images: data/mllm_judge/raw/Dataset/image/{id}.jpg
    Fields: score_id, name, answer, id, instruction, image_path, original_dataset, human
    Scores: 1-5 (integer), 5,719 samples across 1,373 images
    Models: cogvlm, gemini, gpt4, llava, qwen (multiple answers per image)
"""

import json
import os
from typing import Optional

from .base import BaseDataset
from ..utils.io import load_yaml


class MLLMJudgeDataset(BaseDataset):
    """Loads the scoring subset of MLLM-as-a-Judge.

    Data structure:
        data/mllm_judge/
        └── raw/
            └── Dataset/
                ├── Benchmark/
                │   └── score.jsonl    ← 5,719 scoring samples
                ├── Benchmark_Lite/
                │   └── score_lite.jsonl  ← 1,430 subset
                └── image/             ← 4,118 images (1,373 used in scoring)

    Each sample has:
        - instruction: the question/task about the image
        - answer: the model's response (from llava/cogvlm/gpt4/gemini/qwen)
        - image_path: image filename (e.g., "100.jpg")
        - human: 1-5 score from human annotators
        - name: which model generated the answer
        - original_dataset: source benchmark (coco, chartqa, etc.)
    """

    # Default file search order
    SCORE_FILE_CANDIDATES = [
        "raw/Dataset/Benchmark/score.jsonl",
        "raw/Dataset/Benchmark_Lite/score_lite.jsonl",
        "Dataset/Benchmark/score.jsonl",
        "Benchmark/score.jsonl",
        "score.jsonl",
    ]

    # Image directory search order
    IMAGE_DIR_CANDIDATES = [
        "raw/Dataset/image",
        "Dataset/image",
        "image",
        "images",
    ]

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_yaml(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")

        self.data_dir = self.config["data_dir"]
        self.gt_scale = self.config.get("gt_scale", [1, 5])
        self.target_scale = self.config.get("target_scale", [1, 5])
        self.scale_mapping = self.config.get("scale_mapping", "none")
        self.subset_size = self.config.get("subset_size", None)
        self.filter_model = self.config.get("filter_model", None)

        # Find image directory
        self.image_dir = self._find_image_dir()

        # Load data
        self.samples = self._load_raw_data()

        # Apply filters
        if self.filter_model:
            self.samples = [s for s in self.samples if s["model_name"] == self.filter_model]
            print(f"[MLLMJudgeDataset] Filtered to model '{self.filter_model}': {len(self.samples)} samples")

        if self.subset_size is not None and self.subset_size < len(self.samples):
            self.samples = self.samples[:self.subset_size]
            print(f"[MLLMJudgeDataset] Using subset of {self.subset_size} samples")

    def _find_image_dir(self):
        """Find the image directory."""
        for candidate in self.IMAGE_DIR_CANDIDATES:
            path = os.path.join(self.data_dir, candidate)
            if os.path.isdir(path):
                return path
        return os.path.join(self.data_dir, "raw/Dataset/image")

    def _load_raw_data(self):
        """Load scoring data from JSONL file."""
        # Find the scoring file
        score_path = None
        for candidate in self.SCORE_FILE_CANDIDATES:
            path = os.path.join(self.data_dir, candidate)
            if os.path.exists(path):
                score_path = path
                break

        if score_path is None:
            raise FileNotFoundError(
                f"Could not find scoring data in {self.data_dir}. "
                f"Expected one of: {self.SCORE_FILE_CANDIDATES}. "
                f"Run `python scripts/download_data.py` first."
            )

        # Read JSONL
        raw_data = []
        with open(score_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))

        print(f"[MLLMJudgeDataset] Loaded {len(raw_data)} samples from {score_path}")

        # Normalize to our standard format
        samples = []
        for item in raw_data:
            image_name = item.get("image_path", "")
            image_path = os.path.join(self.image_dir, image_name)

            raw_score = float(item.get("human", 0))

            # Map score if scales differ, otherwise use as-is
            if self.gt_scale == self.target_scale or self.scale_mapping == "none":
                mapped_score = raw_score
            else:
                mapped_score = self.map_score(
                    raw_score, self.gt_scale, self.target_scale, self.scale_mapping
                )

            samples.append({
                "image_path": image_path,
                "question": item.get("instruction", ""),
                "answer": item.get("answer", ""),
                "gt_score_raw": raw_score,
                "gt_score": mapped_score,
                "model_name": item.get("name", ""),
                "metadata": {
                    "score_id": item.get("score_id"),
                    "sample_id": item.get("id"),
                    "original_dataset": item.get("original_dataset", ""),
                    "model_name": item.get("name", ""),
                },
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_image_path(self, idx):
        return self.samples[idx]["image_path"]

    def get_question(self, idx):
        return self.samples[idx]["question"]

    def get_gt_score(self, idx):
        return self.samples[idx]["gt_score"]

    def get_gt_score_raw(self, idx):
        return self.samples[idx]["gt_score_raw"]

    def get_answer(self, idx):
        return self.samples[idx].get("answer", "")

    def get_model_name(self, idx):
        return self.samples[idx].get("model_name", "")
