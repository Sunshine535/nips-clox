"""Answer-type-aware extraction and checking.

Replaces the permissive substring fallback in evaluation.py.
Each answer type has strict extraction and comparison logic.
"""
from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"^Thinking Process:.*?(?=\n(?:Q:|A:|Question:|Answer:|Let\'s|Step 1|\d+\.))",
        "", text, flags=re.DOTALL,
    )
    return text.strip()


def extract_numeric(text: str) -> str | None:
    text = strip_thinking_blocks(text)
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"####\s*(.+)",
        r"(?:final answer|the answer is|answer:)\s*[=:\s]*\$?\s*([-−]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
        r"(?:Therefore|So|Thus|Hence),?\s*(?:the (?:final )?answer is)?\s*\$?\s*([-−]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            ans = ans.replace(",", "").replace("−", "-")
            if ans:
                return ans
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        m = re.search(r"([-−]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", line)
        if m:
            return m.group(1).replace(",", "").replace("−", "-")
    return None


def extract_multiple_choice(text: str) -> str | None:
    text = strip_thinking_blocks(text)
    patterns = [
        r"(?:the answer is|answer:)\s*\(?([a-eA-E])\)?",
        r"(?:Therefore|So|Thus|Hence),?\s*(?:the answer is)?\s*\(?([a-eA-E])\)?",
        r"\(([a-eA-E])\)\s*$",
        r"(?:^|\n)\s*([a-eA-E])\s*[\.\):]",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    last_lines = text.strip().split("\n")[-3:]
    for line in reversed(last_lines):
        m = re.search(r"\b([A-E])\b", line)
        if m:
            return m.group(1).upper()
    return None


def extract_boolean(text: str) -> str | None:
    text = strip_thinking_blocks(text).lower()
    patterns = [
        r"(?:the answer is|answer:)\s*(yes|no|true|false)",
        r"(?:Therefore|So|Thus|Hence),?\s*(?:the answer is)?\s*(yes|no|true|false)",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).lower()
            return "yes" if ans in ("yes", "true") else "no"
    last_lines = text.strip().split("\n")[-3:]
    for line in reversed(last_lines):
        line_l = line.lower().strip()
        if line_l in ("yes", "no", "yes.", "no.", "true", "false", "true.", "false."):
            return "yes" if line_l.rstrip(".") in ("yes", "true") else "no"
    return None


def extract_answer_typed(text: str, answer_type: str) -> str:
    if answer_type == "numeric" or answer_type == "math_expression":
        result = extract_numeric(text)
        return result if result else ""
    elif answer_type == "multiple_choice":
        result = extract_multiple_choice(text)
        return result if result else ""
    elif answer_type == "boolean":
        result = extract_boolean(text)
        return result if result else ""
    else:
        result = extract_numeric(text)
        if result:
            return result
        result = extract_multiple_choice(text)
        if result:
            return result
        return text.strip().split("\n")[-1].strip() if text.strip() else ""


_LATEX_NORMALIZE = [
    (r"\\boxed\{([^}]+)\}", r"\1"),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"\\cdot|\\times", r"*"),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\,|\\;|\\!|\\:|\\ ", ""),
    (r"\\!", ""),
    (r"\$", ""),
    (r"\s+", ""),
]


def normalize_math_expression(text: str) -> str:
    """LaTeX-aware normalization for MATH expression equivalence.

    Strips boxes, common spacing macros, $-delimiters, and converts
    \\cdot/\\times → *. Returns lowercase string with all whitespace removed.
    """
    if text is None:
        return ""
    s = str(text)
    for pat, repl in _LATEX_NORMALIZE:
        s = re.sub(pat, repl, s)
    return s.lower().strip()


def check_answer_strict(prediction: str, reference: str, answer_type: str = "text") -> bool:
    if not prediction or not reference:
        return False

    if answer_type == "math_expression":
        # Strict for MATH-style answers: try numeric first, then LaTeX-normalized
        # string equivalence. This is more robust than the previous numeric-only
        # path which silently ignored expressions like "x+1" or "(2,3)".
        pred_clean = prediction.strip().replace(",", "").replace("−", "-")
        ref_clean = reference.strip().replace(",", "").replace("−", "-")
        if pred_clean == ref_clean:
            return True
        try:
            pn = float(pred_clean); rn = float(ref_clean)
            return abs(pn - rn) < 1e-6
        except (ValueError, TypeError):
            pass
        return normalize_math_expression(prediction) == normalize_math_expression(reference)

    if answer_type == "numeric":
        pred_clean = prediction.strip().replace(",", "").replace("−", "-")
        ref_clean = reference.strip().replace(",", "").replace("−", "-")
        if pred_clean == ref_clean:
            return True
        try:
            return abs(float(pred_clean) - float(ref_clean)) < 1e-6
        except (ValueError, TypeError):
            return False

    if answer_type == "multiple_choice":
        p = prediction.strip().upper()
        r = reference.strip().upper()
        if len(p) == 1 and len(r) == 1:
            return p == r
        p_letter = re.search(r"([A-E])", p)
        r_letter = re.search(r"([A-E])", r)
        if p_letter and r_letter:
            return p_letter.group(1) == r_letter.group(1)
        return False

    if answer_type == "boolean":
        p = prediction.strip().lower().rstrip(".")
        r = reference.strip().lower().rstrip(".")
        p_bool = p in ("yes", "true", "1")
        r_bool = r in ("yes", "true", "1")
        p_neg = p in ("no", "false", "0")
        r_neg = r in ("no", "false", "0")
        if (p_bool or p_neg) and (r_bool or r_neg):
            return p_bool == r_bool
        return False

    pred_norm = prediction.strip().lower()
    ref_norm = reference.strip().lower()
    return pred_norm == ref_norm
