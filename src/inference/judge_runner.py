"""Judge VLM inference runner with per-sample saving and resumability.

Runs a judge VLM on (image, question, actor_answer) triplets and saves:
    - sample_XXXX.json: judge text output, score, tokens, metadata
    - sample_XXXX.pt: full logit tensors + top-k logprob dicts

Supports two answer sources:
    1. Actor output files (from ActorRunner) — use_dataset_answers=False
    2. Dataset's pre-existing answers — use_dataset_answers=True (default for MLLM-Judge)
       This is needed because human GT scores correspond to the dataset's model answers.
"""

import gc
import os
import time

import torch
from tqdm import tqdm

from ..models.base import BaseVLM, GenerationOutput
from ..data.base import BaseDataset
from ..utils.io import (
    atomic_save_json,
    atomic_save_pt,
    load_json,
    sample_exists,
    get_sample_paths,
)


class JudgeRunner:
    """Runs judge VLM inference to score answers.

    Features:
        - Two answer modes: from actor outputs OR from dataset (for GT alignment)
        - Formats judge prompt with question + actor_answer
        - Saves full logprobs for Signal 2 + Signal 3 extraction
        - Resumability and atomic saves
    """

    def __init__(self, model: BaseVLM, dataset: BaseDataset,
                 actor_output_dir: str, output_dir: str,
                 prompt_template: str, save_full_logits: bool = True,
                 use_dataset_answers: bool = True):
        """
        Args:
            model: Judge VLM model
            dataset: Dataset with questions, images, and optionally answers
            actor_output_dir: Where actor outputs are saved (used when use_dataset_answers=False)
            output_dir: Where to save judge outputs
            prompt_template: Judge prompt with {question} and {actor_answer} placeholders
            save_full_logits: Whether to save full vocab logits (large but needed for Signal 1)
            use_dataset_answers: If True, use answers from dataset (for GT-aligned evaluation).
                                If False, use answers from actor output files.
        """
        self.model = model
        self.dataset = dataset
        self.actor_output_dir = actor_output_dir
        self.output_dir = output_dir
        self.prompt_template = prompt_template
        self.save_full_logits = save_full_logits
        self.use_dataset_answers = use_dataset_answers
        os.makedirs(output_dir, exist_ok=True)

    def run(self, start_idx: int = 0, end_idx: int = None, limit: int = None):
        """Run judge inference.

        Args:
            start_idx: First sample index to process
            end_idx: Last sample index (exclusive)
            limit: Max number of samples to process
        """
        if end_idx is None:
            end_idx = len(self.dataset)
        if limit is not None:
            end_idx = min(start_idx + limit, len(self.dataset))

        mode = "dataset answers" if self.use_dataset_answers else "actor outputs"
        print(f"[JudgeRunner] Using {mode} for evaluation")

        skipped = 0
        processed = 0
        errors = 0
        no_actor = 0

        pbar = tqdm(range(start_idx, end_idx), desc="Judge inference")

        for idx in pbar:
            # Resumability check
            if sample_exists(self.output_dir, idx):
                skipped += 1
                pbar.set_postfix(done=processed, skip=skipped, err=errors)
                continue

            # If using actor outputs, check they exist
            if not self.use_dataset_answers:
                if not sample_exists(self.actor_output_dir, idx):
                    no_actor += 1
                    pbar.set_postfix(done=processed, skip=skipped, err=errors, no_actor=no_actor)
                    continue

            try:
                self._process_sample(idx)
                processed += 1
            except Exception as e:
                print(f"\n[JudgeRunner] Error on sample {idx}: {e}")
                errors += 1

            pbar.set_postfix(done=processed, skip=skipped, err=errors)

            if processed % 50 == 0 and processed > 0:
                torch.cuda.empty_cache()

        print(f"\n[JudgeRunner] Done: {processed} processed, {skipped} skipped, "
              f"{errors} errors, {no_actor} missing actor outputs")

    def _process_sample(self, idx: int):
        """Run judge inference on a single sample."""
        # Get the answer to evaluate
        if self.use_dataset_answers:
            actor_answer = self.dataset.get_answer(idx)
        else:
            actor_json_path, _ = get_sample_paths(self.actor_output_dir, idx)
            actor_data = load_json(actor_json_path)
            actor_answer = actor_data["generated_text"]

        # Get image path and question from dataset
        image_path = self.dataset.get_image_path(idx)
        question = self.dataset.get_question(idx)

        # Format judge prompt
        prompt = self.prompt_template.format(
            question=question,
            actor_answer=actor_answer,
        )

        # Run generation
        output: GenerationOutput = self.model.generate(image_path, prompt)

        # Build JSON output
        json_data = {
            "sample_id": idx,
            "image_path": image_path,
            "question": question,
            "actor_answer": actor_answer,
            "answer_source": "dataset" if self.use_dataset_answers else "actor_output",
            "judge_output": output.generated_text,
            "tokens": output.tokens,
            "token_logprobs": output.token_logprobs,
            "num_tokens": len(output.tokens),
            "gt_score_raw": self.dataset.get_gt_score_raw(idx) if hasattr(self.dataset, 'get_gt_score_raw') else None,
            "gt_score": self.dataset.get_gt_score(idx),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Build PT output
        pt_data = {
            "top_logprobs": output.top_logprobs,
        }
        if self.save_full_logits and output.full_scores is not None:
            pt_data["full_scores"] = torch.stack(output.full_scores)

        # Atomic save
        json_path, pt_path = get_sample_paths(self.output_dir, idx)
        atomic_save_json(json_data, json_path)
        atomic_save_pt(pt_data, pt_path)

        if output.full_scores is not None:
            del output.full_scores
