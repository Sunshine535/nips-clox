from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkExample:
    question: str
    answer: str
    answer_type: str
    difficulty: str
    benchmark: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[,\$%]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def load_gsm8k(max_examples: int | None = None) -> list[BenchmarkExample]:
    import glob, os as _os
    from datasets import load_dataset, Dataset
    # Try direct arrow file load first (fastest, works offline)
    ds = None
    cache_root = _os.environ.get("HF_DATASETS_CACHE", "") or _os.environ.get("HF_HOME", "")
    if cache_root:
        search_roots = [cache_root, _os.path.join(cache_root, "datasets")]
        for root in search_roots:
            pattern = _os.path.join(root, "openai___gsm8k", "main", "*", "*", "gsm8k-test.arrow")
            matches = glob.glob(pattern)
            if matches:
                ds = Dataset.from_file(matches[0])
                break
    if ds is None:
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test")
        except Exception:
            ds = load_dataset("gsm8k", "main", split="test")
    examples = []
    for i, item in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        answer_text = item["answer"]
        final_answer = ""
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
        else:
            numbers = re.findall(r"-?\d+\.?\d*", answer_text)
            final_answer = numbers[-1] if numbers else answer_text.strip()

        n_steps = answer_text.count("\n")
        if n_steps <= 3:
            difficulty = "easy"
        elif n_steps <= 6:
            difficulty = "medium"
        else:
            difficulty = "hard"

        examples.append(BenchmarkExample(
            question=item["question"],
            answer=_normalize_answer(final_answer),
            answer_type="numeric",
            difficulty=difficulty,
            benchmark="gsm8k",
            example_id=f"gsm8k_{i}",
            metadata={"full_solution": answer_text, "n_steps": n_steps},
        ))
    return examples


def load_math(max_examples: int | None = None, levels: list[int] | None = None) -> list[BenchmarkExample]:
    from datasets import load_dataset
    # EleutherAI/hendrycks_math has per-subject configs
    configs = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    all_items = []
    for cfg in configs:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
            all_items.extend(list(ds))
        except Exception:
            pass
    if not all_items:
        try:
            all_items = list(load_dataset("hendrycks/competition_math", split="test"))
        except Exception:
            raise RuntimeError("Could not load MATH dataset")
    ds = all_items
    examples = []
    for i, item in enumerate(ds):
        if max_examples and len(examples) >= max_examples:
            break
        level = int(re.search(r"\d+", item.get("level", "3")).group()) if re.search(r"\d+", item.get("level", "3")) else 3
        if levels and level not in levels:
            continue

        answer = item.get("answer", "")
        boxed = re.search(r"\\boxed\{([^}]+)\}", item.get("solution", ""))
        if boxed:
            answer = boxed.group(1)

        difficulty_map = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}
        examples.append(BenchmarkExample(
            question=item["problem"],
            answer=_normalize_answer(answer),
            answer_type="math_expression",
            difficulty=difficulty_map.get(level, "medium"),
            benchmark="math",
            example_id=f"math_{i}",
            metadata={
                "level": level,
                "type": item.get("type", ""),
                "solution": item.get("solution", ""),
            },
        ))
    return examples


def load_strategyqa(
    max_examples: int | None = None,
    allow_train_eval: bool = False,
) -> list[BenchmarkExample]:
    """Load StrategyQA. By default forbids the metaeval train fallback.

    Pass `allow_train_eval=True` ONLY for development/diagnostic. The
    `metaeval/strategy-qa` train split has been used in evaluation by mistake
    in the past (GPT-5.5 Pro §3 P0 row 4). Forbidding it by default forces
    callers to acknowledge the risk explicitly.
    """
    from datasets import load_dataset
    candidates = [("ChilleD/StrategyQA", "test")]
    if allow_train_eval:
        candidates.append(("metaeval/strategy-qa", "train"))

    ds = None
    used_repo = used_split = None
    for repo, split in candidates:
        try:
            ds = load_dataset(repo, split=split)
            used_repo, used_split = repo, split
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            "Could not load StrategyQA test split. To allow train fallback, "
            "pass allow_train_eval=True (development only)."
        )

    examples = []
    for i, item in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        answer = str(item.get("answer", "")).lower()
        if answer in ("true", "1", "yes"):
            answer = "yes"
        elif answer in ("false", "0", "no"):
            answer = "no"

        examples.append(BenchmarkExample(
            question=item["question"],
            answer=answer,
            answer_type="boolean",
            difficulty="medium",
            benchmark="strategyqa",
            example_id=f"strategyqa_{i}",
            metadata={
                "facts": item.get("facts", []),
                "decomposition": item.get("decomposition", ""),
                "source_repo": used_repo,
                "source_split": used_split,
            },
        ))
    return examples


def load_arc_challenge(max_examples: int | None = None) -> list[BenchmarkExample]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for i, item in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = " ".join(f"({l}) {t}" for l, t in zip(labels, texts))
        answer_key = item["answerKey"]

        examples.append(BenchmarkExample(
            question=f"{item['question']}\nChoices: {choice_str}",
            answer=answer_key.upper(),
            answer_type="multiple_choice",
            difficulty="hard",
            benchmark="arc_challenge",
            example_id=f"arc_{i}",
            metadata={"choices": dict(zip(labels, texts))},
        ))
    return examples


def load_bbh(
    max_examples: int | None = None,
    subtasks: list[str] | None = None,
    per_subtask_cap: int | None = None,
) -> list[BenchmarkExample]:
    from datasets import load_dataset

    default_subtasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "logical_deduction_five_objects", "multistep_arithmetic_two",
        "navigate", "tracking_shuffled_objects_three_objects",
    ]
    selected = subtasks or default_subtasks

    if per_subtask_cap is None and max_examples is not None:
        per_subtask_cap = max(1, max_examples // max(len(selected), 1))

    examples = []
    for subtask in selected:
        try:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
        except Exception:
            continue
        count = 0
        for i, item in enumerate(ds):
            if per_subtask_cap is not None and count >= per_subtask_cap:
                break
            examples.append(BenchmarkExample(
                question=item["input"],
                answer=_normalize_answer(str(item["target"])),
                answer_type="text",
                difficulty="hard",
                benchmark=f"bbh_{subtask}",
                example_id=f"bbh_{subtask}_{i}",
                metadata={"subtask": subtask},
            ))
            count += 1
    return examples


def compute_bbh_macro_average(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    from collections import defaultdict
    subtask_correct: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        subtask = r.get("metadata", {}).get("subtask") or ""
        if not subtask:
            bname = r.get("benchmark", "")
            if bname.startswith("bbh_"):
                subtask = bname[4:]
        if subtask:
            subtask_correct[subtask].append(bool(r.get("correct", False)))

    per_subtask_acc = {}
    for st, corrects in sorted(subtask_correct.items()):
        per_subtask_acc[st] = sum(corrects) / max(len(corrects), 1)

    macro_avg = (
        sum(per_subtask_acc.values()) / max(len(per_subtask_acc), 1)
        if per_subtask_acc else 0.0
    )
    return {
        "macro_average": macro_avg,
        "per_subtask": per_subtask_acc,
        "n_subtasks": len(per_subtask_acc),
    }


BENCHMARK_REGISTRY: dict[str, Any] = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "strategyqa": load_strategyqa,
    "arc_challenge": load_arc_challenge,
    "bbh": load_bbh,
}


def load_benchmark(name: str, max_examples: int | None = None, **kwargs) -> list[BenchmarkExample]:
    loader = BENCHMARK_REGISTRY.get(name)
    if loader is None:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY)}")
    return loader(max_examples=max_examples, **kwargs)


FEW_SHOT_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "Q: There are 15 trees in the grove. Grove workers will plant trees today. "
        "After they are done, there will be 21 trees. How many trees did the workers plant today?\n"
        "A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after planting. "
        "So 21 - 15 = 6 trees were planted. The answer is 6.\n\n"
        "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
        "A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n"
    ),
    "math": (
        "Problem: Find the value of $x$ if $2x + 3 = 11$.\n"
        "Solution: Subtract 3 from both sides: $2x = 8$. Divide by 2: $x = 4$. The answer is $\\boxed{4}$.\n"
    ),
    "strategyqa": (
        "Q: Would a vegetarian enjoy a BLT sandwich?\n"
        "A: Let's think step by step. A BLT contains bacon, which is meat. "
        "Vegetarians do not eat meat. So a vegetarian would not enjoy a BLT. The answer is no.\n"
    ),
    "arc_challenge": (
        "Q: Which of the following is a renewable resource?\n"
        "Choices: (A) Coal (B) Solar energy (C) Natural gas (D) Oil\n"
        "A: Let's think step by step. Coal, natural gas, and oil are fossil fuels and non-renewable. "
        "Solar energy comes from the sun and is renewable. The answer is (B).\n"
    ),
}
