"""LLaVA model wrapper supporting both LLaVA-1.5 (HF) and LLaVA-Critic (LLaVA-NeXT).

Two model backends:
    1. LLaVA-1.5 (actor): Standard HF LlavaForConditionalGeneration + AutoProcessor
    2. LLaVA-Critic (judge): LLaVA-NeXT package with custom load_pretrained_model

Inference + logprob extraction adapted from sibling LLM repo qwen_eval.py.
"""

import copy
import gc
import os
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseVLM, GenerationOutput
from ..utils.io import load_yaml


class LLaVAModel(BaseVLM):
    """Unified wrapper for LLaVA-family models.

    Automatically selects the correct loading method based on model_class config:
        - "LlavaForConditionalGeneration" → HF standard (LLaVA-1.5)
        - "LlavaQwenForCausalLM" → LLaVA-NeXT package (LLaVA-Critic)
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_yaml(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")

        self.model_name = self.config["name"]
        self.hf_id = self.config["hf_id"]
        self.model_class = self.config.get("model_class", "LlavaForConditionalGeneration")
        self.dtype = getattr(torch, self.config.get("dtype", "float16"))
        self.device_map = self.config.get("device_map", "auto")
        self.max_new_tokens = self.config.get("max_new_tokens", 512)
        self.do_sample = self.config.get("do_sample", False)
        self.top_k = self.config.get("top_k", 50)
        self.save_full_scores = self.config.get("save_full_scores", True)

        self.model = None
        self.tokenizer = None
        self.processor = None  # For HF models
        self.image_processor = None  # For LLaVA-NeXT models
        self._backend = None  # "hf" or "llava_next"

    def _resolve_model_path(self):
        """Try local path first, then HuggingFace Hub."""
        local_candidates = [
            f"models/{self.model_name}",
            f"models/{self.hf_id.split('/')[-1]}",
        ]
        for local in local_candidates:
            if os.path.exists(local):
                print(f"[LLaVAModel] Using local path: {local}")
                return local
        return self.hf_id

    def load_model(self, device: str = "cuda:0"):
        """Load model using the appropriate backend."""
        print(f"[LLaVAModel] Loading {self.hf_id} (class={self.model_class})...")
        model_path = self._resolve_model_path()

        if self.model_class == "LlavaQwenForCausalLM":
            self._load_llava_next(model_path, device)
        else:
            self._load_hf_standard(model_path, device)

        print(f"[LLaVAModel] Loaded {self.model_name} successfully (backend={self._backend})")

    def _load_hf_standard(self, model_path: str, device: str):
        """Load LLaVA-1.5 via standard HuggingFace transformers."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._backend = "hf"
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _load_llava_next(self, model_path: str, device: str):
        """Load LLaVA-Critic via LLaVA-NeXT package.

        Direct loading approach to avoid builder.py compatibility issues
        with transformers 4.47. Key fix: convert text_config dict → Qwen2Config.
        """
        from transformers import AutoTokenizer, AutoConfig, Qwen2Config
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.mm_utils import process_images

        self._backend = "llava_next"

        # Fix config compatibility: text_config must be a proper Config object
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, 'text_config') and isinstance(config.text_config, dict):
            config.text_config = Qwen2Config(**config.text_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = LlavaQwenForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            torch_dtype=self.dtype,
            device_map="auto",
            config=config,
        )
        self.model.eval()

        # Load vision tower and get image processor
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self.image_processor = vision_tower.image_processor

    def generate(self, image_path: str, prompt: str) -> GenerationOutput:
        """Run inference and extract per-token logprobs.

        Routes to backend-specific generation method.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._backend == "llava_next":
            return self._generate_llava_next(image_path, prompt)
        else:
            return self._generate_hf_standard(image_path, prompt)

    def _generate_hf_standard(self, image_path: str, prompt: str) -> GenerationOutput:
        """Generate with HF standard LLaVA (LLaVA-1.5)."""
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_input, images=image, return_tensors="pt",
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=self.do_sample,
            )

        return self._extract_logprobs(
            generation, input_length,
            decode_fn=lambda ids: self.processor.decode(ids, skip_special_tokens=True),
            decode_token_fn=lambda tid: self.processor.decode([tid]).strip(),
        )

    def _generate_llava_next(self, image_path: str, prompt: str) -> GenerationOutput:
        """Generate with LLaVA-NeXT (LLaVA-Critic)."""
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.model.device)
                        for _image in image_tensor]

        # Build conversation using LLaVA-Critic's template
        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        input_length = input_ids.shape[-1]
        image_sizes = [image.size]

        with torch.no_grad():
            generation = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=self.do_sample,
                repetition_penalty=1.1,
            )

        return self._extract_logprobs(
            generation, input_length,
            decode_fn=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
            decode_token_fn=lambda tid: self.tokenizer.decode([tid]).strip(),
        )

    def _extract_logprobs(self, generation, input_length, decode_fn, decode_token_fn):
        """Extract per-token logprobs from generation output.

        Adapted from sibling repo qwen_eval.py (lines 109-132).
        Shared between both backends.
        """
        sequences = generation.sequences
        scores = generation.scores

        # Fix input_length: for models using inputs_embeds (e.g., LLaVA-NeXT),
        # sequences may not include the full input prefix. Compute the actual
        # offset from the output shape: total_len - num_generated = actual_prefix.
        actual_input_length = sequences.shape[1] - len(scores)
        if actual_input_length != input_length:
            input_length = actual_input_length

        generated_ids = sequences[0, input_length:].tolist()
        generated_text = decode_fn(generated_ids)

        tokens = []
        token_logprobs = []
        top_logprobs = []
        full_scores_list = []

        for step_idx, step_scores in enumerate(scores):
            log_probs = F.log_softmax(step_scores, dim=-1)

            token_id = sequences[0, input_length + step_idx]
            lp = log_probs[0, token_id].item()
            # Replace NaN/inf with fallback (can happen in degenerate repetition)
            if not (lp == lp):  # NaN check
                lp = -100.0

            topk_lp, topk_ids = log_probs.topk(self.top_k, dim=-1)
            topk_lp = topk_lp[0].tolist()
            topk_ids = topk_ids[0].tolist()

            topk_dict = {}
            for tok_id, logp in zip(topk_ids, topk_lp):
                # Replace NaN with fallback
                if not (logp == logp):
                    logp = -100.0
                tok_str = decode_token_fn(tok_id)
                topk_dict[tok_str] = logp

            token_str = decode_token_fn(token_id.item())
            tokens.append(token_str)
            token_logprobs.append(lp)
            top_logprobs.append(topk_dict)

            if self.save_full_scores:
                full_scores_list.append(step_scores[0].cpu())

        return GenerationOutput(
            generated_text=generated_text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            full_scores=full_scores_list if self.save_full_scores else None,
        )

    def unload_model(self):
        """Free GPU memory."""
        for attr in ['model', 'processor', 'tokenizer', 'image_processor']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[LLaVAModel] Unloaded {self.model_name}")
