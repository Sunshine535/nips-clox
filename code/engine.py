"""vLLM-based generation engine for CLOX experiments.

Provides batched, high-throughput inference with full logprob support
for topology estimation. Designed for multi-GPU tensor parallelism.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any


def detect_gpu_count() -> int:
    """Auto-detect available GPU count from CUDA_VISIBLE_DEVICES or torch."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        ids = [x.strip() for x in cvd.split(",") if x.strip()]
        if ids:
            return len(ids)
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def auto_tp(model_name: str, available_gpus: int | None = None) -> int:
    """Pick tensor_parallel_size to use all available GPUs.

    Uses the largest power-of-2 <= available GPUs, ensuring models fit.
    For small models (≤14B), caps at 2 to avoid communication overhead.
    """
    if available_gpus is None:
        available_gpus = detect_gpu_count()
    name_lower = model_name.lower()
    # MoE models (e.g. Qwen3.5-35B-A3B): activated params are small
    if re.search(r'-a\d+b', name_lower):
        max_tp = min(2, available_gpus)
    # Extract largest "XB" number to determine model size
    elif (m := re.findall(r'(\d+(?:\.\d+)?)b', name_lower)):
        largest_b = max(float(x) for x in m)
        if largest_b <= 14:
            max_tp = min(2, available_gpus)
        else:
            max_tp = available_gpus
    else:
        max_tp = available_gpus
    # Round down to power of 2
    p = 1
    while p * 2 <= max_tp:
        p *= 2
    return p


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int]
    logprobs: list[float]
    top_logprobs: list[dict[int, float]]
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def token_entropy(self) -> list[float]:
        """Per-token entropy from top-K logprob distribution."""
        entropies = []
        for pos_lps in self.top_logprobs:
            if not pos_lps:
                entropies.append(0.0)
                continue
            lps = list(pos_lps.values())
            probs = [math.exp(lp) for lp in lps]
            total = sum(probs)
            if total < 1e-10:
                entropies.append(0.0)
                continue
            probs = [p / total for p in probs]
            ent = -sum(p * math.log(max(p, 1e-30)) for p in probs if p > 0)
            entropies.append(ent)
        return entropies


class VLLMEngine:
    """Wrapper around vLLM for batched inference with logprobs."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        quantization: str | None = None,
        seed: int = 42,
    ):
        from vllm import LLM
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            seed=seed,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Apply model-specific chat template."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def wrap_prompt(self, text: str) -> str:
        """Wrap raw text in chat template with /no_think for Qwen3."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. /no_think"},
                {"role": "user", "content": text},
            ]
            return self.apply_chat_template(messages)
        except Exception:
            return text

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
        logprobs: int = 20,
        stop: list[str] | None = None,
        wrap: bool = True,
    ) -> list[list[GenerationOutput]]:
        from vllm import SamplingParams
        if wrap:
            prompts = [self.wrap_prompt(p) for p in prompts]
        default_stop = ["</s>", "<|im_end|>", "<|endoftext|>"]
        all_stop = list(set((stop or []) + default_stop))
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01) if n > 1 or temperature > 0 else 0.0,
            top_p=top_p,
            n=n,
            logprobs=logprobs,
            stop=all_stop,
        )
        vllm_outputs = self.llm.generate(prompts, params)
        results = []
        for req_output in vllm_outputs:
            prompt_len = len(req_output.prompt_token_ids)
            samples = []
            for completion in req_output.outputs:
                token_ids = list(completion.token_ids)
                lps = []
                top_lps = []
                if completion.logprobs:
                    for idx, pos_dict in enumerate(completion.logprobs):
                        sel_id = token_ids[idx] if idx < len(token_ids) else None
                        if sel_id is not None and sel_id in pos_dict:
                            lps.append(pos_dict[sel_id].logprob)
                        elif pos_dict:
                            lps.append(next(iter(pos_dict.values())).logprob)
                        else:
                            lps.append(0.0)
                        top_lps.append({
                            tid: lp_obj.logprob for tid, lp_obj in pos_dict.items()
                        })
                samples.append(GenerationOutput(
                    text=completion.text,
                    token_ids=token_ids,
                    logprobs=lps,
                    top_logprobs=top_lps,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(token_ids),
                ))
            results.append(samples)
        return results

    def generate_single(self, prompt: str, max_tokens: int = 512,
                        temperature: float = 0.0, logprobs: int = 20,
                        wrap: bool = True) -> GenerationOutput:
        return self.generate([prompt], max_tokens=max_tokens,
                             temperature=temperature, n=1, logprobs=logprobs,
                             wrap=wrap)[0][0]

    def generate_multi(self, prompt: str, n: int = 5, max_tokens: int = 512,
                       temperature: float = 0.7, logprobs: int = 20,
                       wrap: bool = True) -> list[GenerationOutput]:
        return self.generate([prompt], max_tokens=max_tokens,
                             temperature=temperature, n=n, logprobs=logprobs,
                             wrap=wrap)[0]

    def generate_batch(self, prompts: list[str], max_tokens: int = 512,
                       temperature: float = 0.0, logprobs: int = 20,
                       wrap: bool = True) -> list[GenerationOutput]:
        """One sample per prompt, batched."""
        results = self.generate(prompts, max_tokens=max_tokens,
                                temperature=temperature, n=1, logprobs=logprobs,
                                wrap=wrap)
        return [r[0] for r in results]


def extract_answer(text: str) -> str:
    # Strip <think>...</think> tags from Qwen3-style thinking models
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip Qwen3.5-style "Thinking Process:" meta-commentary blocks
    # These appear when /no_think fails — model generates untagged thinking
    text = re.sub(
        r'^Thinking Process:.*?(?=\n(?:Q:|A:|Question:|Answer:|Let\'s|Step 1|\d+\.))',
        '', text, flags=re.DOTALL,
    ).strip()
    # Fallback: if entire output is a thinking block, try to extract from it
    if text.startswith('Thinking Process:') or text.startswith('1.  **Analyze'):
        # Look for final answer/result within the thinking block (take LAST match)
        calc_matches = re.findall(
            r'(?:final answer|answer|result|Wilfred|=)\s*(?:is\s*)?[=:\s]*\$?(-?\d+(?:\.\d+)?)',
            text, re.IGNORECASE,
        )
        if calc_matches:
            return calc_matches[-1]
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"####\s*(.+)",
        r"(?:final answer|the answer is|answer:)\s*[=:\s]*([^\n.;]+)",
        r"(?:Therefore|So|Thus|Hence),?\s*(?:the (?:final )?answer is)?\s*([^\n.;]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            if 0 < len(ans) < 200:
                return ans
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        if any(skip in line.lower() for skip in ["let's", "fill in", "step", "masked", "blank", "repair"]):
            continue
        num = re.search(r"[-−]?\d+(?:\.\d+)?", line)
        if num:
            return num.group()
        if len(line) < 50:
            return line
    return lines[-1].strip() if lines else ""


def split_into_steps(text: str) -> list[str]:
    numbered = re.split(r'\n(?=(?:Step\s+)?\d+[\.\):])', text)
    if len(numbered) >= 3:
        return [s.strip() for s in numbered if s.strip() and len(s.split()) >= 3]
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    if len(sentences) >= 3:
        return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
    parts = text.split("\n")
    return [s.strip() for s in parts if s.strip() and len(s.split()) >= 3]
