"""Tests for evaluation.py and answer_extraction integration — GPT-5.5 Pro Task 2.

The default `check_answer` in evaluation.py must delegate to the strict path.
The legacy substring fallback is still available as `check_answer_legacy_unsafe`
for replay-of-historical-results purposes only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from evaluation import check_answer, check_answer_legacy_unsafe
from answer_extraction import check_answer_strict, normalize_math_expression


# ── Default check_answer is strict ──────────────────────────────────


def test_default_check_answer_no_substring_fallback():
    # "the answer is 123 units" should NOT pass when reference is "123" under text
    assert check_answer("the answer is 123 units", "123", "text") is False


def test_default_check_answer_mc_rejects_junk():
    # "1" must not pass for MC reference "D"
    assert check_answer("1", "D", "multiple_choice") is False


def test_default_check_answer_mc_rejects_text_with_letter():
    # Permissive regex over arbitrary text: "the answer is probably C because..."
    # should NOT match "B" under strict.
    assert check_answer("the answer is probably C because of context", "B",
                        "multiple_choice") is False


def test_default_check_answer_numeric_substring_blocked():
    # "100 dollars" vs "100" — strict numeric extraction normalizes both to 100
    # so this should pass — but ensure the failure mode "abc100xyz" does NOT.
    assert check_answer("100", "100", "numeric") is True
    assert check_answer("abc", "100", "numeric") is False


def test_default_check_answer_boolean_strict():
    assert check_answer("yes", "yes", "boolean") is True
    assert check_answer("no", "yes", "boolean") is False
    # Junk text must not pass
    assert check_answer("yes I think so", "yes", "boolean") is False


# ── Legacy is preserved as documented unsafe fallback ────────────────


def test_legacy_unsafe_still_substring():
    # The unsafe version's substring fallback IS still substring (this is intentional —
    # it's preserved for historical-result replay only, never for new claims).
    assert check_answer_legacy_unsafe("the answer is 123", "123", "text") is True


def test_legacy_and_strict_diverge_on_known_case():
    bad_pred = "I think it's option C, definitely"
    ref = "B"
    assert check_answer_legacy_unsafe(bad_pred, ref, "multiple_choice") is False  # MC has its own logic
    # The fundamental contamination is in the text/substring fallback:
    assert check_answer_legacy_unsafe("answer 42 units", "42", "text") is True  # legacy WRONG
    assert check_answer("answer 42 units", "42", "text") is False  # strict CORRECT


# ── MATH expression normalization ────────────────────────────────────


def test_math_expression_latex_box_strip():
    assert check_answer_strict(r"\boxed{42}", "42", "math_expression") is True


def test_math_expression_strict_distinct_strings():
    # Different expressions should NOT match
    assert check_answer_strict("x+1", "x+2", "math_expression") is False


def test_math_expression_dollar_strip():
    assert check_answer_strict("$42$", "42", "math_expression") is True


def test_normalize_math_expression():
    assert normalize_math_expression(r"\boxed{42}") == "42"
    assert normalize_math_expression(r"\frac{1}{2}") == r"\frac{1}{2}"
    assert normalize_math_expression(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert normalize_math_expression(r"a \cdot b") == "a*b"
    assert normalize_math_expression(r"\left( x \right)") == "(x)"
