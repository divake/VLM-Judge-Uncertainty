"""Actor VLM inference runner with per-sample saving and resumability.

Runs an actor VLM on (image, question) pairs and saves:
    - sample_XXXX.json: generated text, tokens, metadata
    - sample_XXXX.pt: full logit tensors + top-k logprob dicts
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
    sample_exists,
    get_sample_paths,
)


class ActorRunner:
    """Runs actor VLM inference and saves per-sample outputs.

    Features:
        - Resumability: skips samples that already have outputs
        - Atomic writes: crash-safe saving via temp files + rename
        - Full logit saving: stores raw vocab-size scores for future use
        - Progress tracking: tqdm progress bar with skip counting
    """

    def __init__(self, model: BaseVLM, dataset: BaseDataset, output_dir: str,
                 save_full_logits: bool = True):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.save_full_logits = save_full_logits
        os.makedirs(output_dir, exist_ok=True)

    def run(self, start_idx: int = 0, end_idx: int = None, limit: int = None):
        """Run actor inference on dataset samples.

        Args:
            start_idx: First sample index to process
            end_idx: Last sample index (exclusive). Defaults to len(dataset)
            limit: Max number of samples to process (overrides end_idx)
        """
        if end_idx is None:
            end_idx = len(self.dataset)
        if limit is not None:
            end_idx = min(start_idx + limit, len(self.dataset))

        skipped = 0
        processed = 0
        errors = 0

        pbar = tqdm(range(start_idx, end_idx), desc=f"Actor inference")

        for idx in pbar:
            # Resumability check
            if sample_exists(self.output_dir, idx):
                skipped += 1
                pbar.set_postfix(done=processed, skip=skipped, err=errors)
                continue

            try:
                result = self._process_sample(idx)
                processed += 1
            except Exception as e:
                print(f"\n[ActorRunner] Error on sample {idx}: {e}")
                errors += 1

            pbar.set_postfix(done=processed, skip=skipped, err=errors)

            # Periodic GPU cache cleanup
            if processed % 50 == 0 and processed > 0:
                torch.cuda.empty_cache()

        print(f"\n[ActorRunner] Done: {processed} processed, {skipped} skipped, {errors} errors")

    def _process_sample(self, idx: int):
        """Run inference on a single sample and save outputs."""
        image_path = self.dataset.get_image_path(idx)
        question = self.dataset.get_question(idx)

        # Format actor prompt (simple question for actor)
        prompt = question

        # Run generation
        output: GenerationOutput = self.model.generate(image_path, prompt)

        # Build JSON output (text + metadata)
        json_data = {
            "sample_id": idx,
            "image_path": image_path,
            "question": question,
            "generated_text": output.generated_text,
            "tokens": output.tokens,
            "token_logprobs": output.token_logprobs,
            "num_tokens": len(output.tokens),
            "gt_score_raw": self.dataset.get_gt_score_raw(idx) if hasattr(self.dataset, 'get_gt_score_raw') else None,
            "gt_score": self.dataset.get_gt_score(idx),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Build PT output (tensors + top-k dicts)
        pt_data = {
            "top_logprobs": output.top_logprobs,  # list of dicts
        }
        if self.save_full_logits and output.full_scores is not None:
            # Stack all per-step score tensors: (num_steps, vocab_size)
            pt_data["full_scores"] = torch.stack(output.full_scores)

        # Atomic save
        json_path, pt_path = get_sample_paths(self.output_dir, idx)
        atomic_save_json(json_data, json_path)
        atomic_save_pt(pt_data, pt_path)

        # Free memory from full_scores
        if output.full_scores is not None:
            del output.full_scores
