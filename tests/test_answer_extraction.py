"""Tests for answer_extraction.py — strict answer-type-aware checking.

GPT-5.5 Pro P0: metric contamination from permissive substring fallback.
These tests verify that the new strict checking does NOT have that bug.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from answer_extraction import (
    extract_numeric, extract_multiple_choice, extract_boolean,
    extract_answer_typed, check_answer_strict,
)


class TestExtractNumeric:
    def test_boxed(self):
        assert extract_numeric("The answer is \\boxed{42}") == "42"

    def test_hash(self):
        assert extract_numeric("blah blah\n#### 123") == "123"

    def test_final_answer(self):
        assert extract_numeric("Therefore the final answer is 3.14") == "3.14"

    def test_negative(self):
        assert extract_numeric("The answer is -7") == "-7"

    def test_no_number(self):
        assert extract_numeric("I don't know") is None

    def test_comma_number(self):
        r = extract_numeric("The total is 1,234,567")
        assert r is not None
        assert "1234567" in r.replace(",", "")


class TestExtractMC:
    def test_the_answer_is_a(self):
        assert extract_multiple_choice("The answer is (A)") == "A"

    def test_therefore_b(self):
        assert extract_multiple_choice("Therefore, the answer is B.") == "B"

    def test_just_letter(self):
        assert extract_multiple_choice("After analysis:\nC") == "C"

    def test_no_letter(self):
        assert extract_multiple_choice("The number is 42") is None


class TestExtractBoolean:
    def test_yes(self):
        assert extract_boolean("The answer is yes") == "yes"

    def test_no(self):
        assert extract_boolean("Therefore, the answer is no.") == "no"

    def test_true(self):
        assert extract_boolean("The answer is True") == "yes"

    def test_false(self):
        assert extract_boolean("The answer is False") == "no"

    def test_no_boolean(self):
        assert extract_boolean("The answer is 42") is None


class TestCheckAnswerStrict:
    # P0 bug: old code returned `pred in ref or ref in pred` for these
    def test_mc_rejects_junk_prediction(self):
        assert check_answer_strict("1", "D", "multiple_choice") is False

    def test_mc_rejects_text_vs_letter(self):
        assert check_answer_strict("the main reason is...", "C", "multiple_choice") is False

    def test_mc_accepts_correct_letter(self):
        assert check_answer_strict("A", "A", "multiple_choice") is True

    def test_mc_case_insensitive(self):
        assert check_answer_strict("b", "B", "multiple_choice") is True

    def test_numeric_exact(self):
        assert check_answer_strict("42", "42", "numeric") is True

    def test_numeric_float_equal(self):
        assert check_answer_strict("3.14", "3.14", "numeric") is True

    def test_numeric_rejects_different(self):
        assert check_answer_strict("41", "42", "numeric") is False

    def test_boolean_yes_true(self):
        assert check_answer_strict("yes", "true", "boolean") is True

    def test_boolean_no_false(self):
        assert check_answer_strict("no", "false", "boolean") is True

    def test_boolean_rejects_mismatch(self):
        assert check_answer_strict("yes", "no", "boolean") is False

    def test_no_substring_fallback(self):
        # The critical P0 fix: must NOT pass via substring containment
        assert check_answer_strict("123", "the answer is 123 units", "text") is False

    def test_text_exact_match(self):
        assert check_answer_strict("paris", "Paris", "text") is True

    def test_empty_prediction(self):
        assert check_answer_strict("", "42", "numeric") is False

    def test_empty_reference(self):
        assert check_answer_strict("42", "", "numeric") is False


class TestExtractAnswerTyped:
    def test_numeric_from_cot(self):
        text = "Step 1: 5+3=8\nStep 2: 8*2=16\n#### 16"
        assert extract_answer_typed(text, "numeric") == "16"

    def test_mc_from_cot(self):
        text = "Looking at the options... The answer is (B)"
        assert extract_answer_typed(text, "multiple_choice") == "B"

    def test_boolean_from_cot(self):
        text = "Considering all factors, the answer is yes."
        assert extract_answer_typed(text, "boolean") == "yes"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
