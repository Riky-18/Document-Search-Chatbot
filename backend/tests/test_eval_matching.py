"""Tests for format-tolerant keyword normalization and matching in eval harness."""

from __future__ import annotations

import pytest
from eval.run_eval import normalize


def test_inequality_normalization_and_matching() -> None:
    """Verify 'n >= d', '$n \\geq d$', and 'n≥d' all match keyword 'n≥d'."""
    keyword = "n≥d"
    norm_kw = normalize(keyword)
    assert norm_kw == "n>=d"

    # All these variations must match the keyword
    variations = [
        "n >= d",
        r"$n \geq d$",
        "n≥d",
        "The condition requires n >= d for full faithfulness.",
        r"We must have $n \geq d$ in Theorem 1.1.",
        "It holds when n≥d.",
    ]
    for ans in variations:
        norm_ans = normalize(ans)
        assert norm_kw in norm_ans, f"Expected '{norm_kw}' to be in normalized answer '{norm_ans}' from '{ans}'"


def test_range_matching_multiple_keywords() -> None:
    """Verify '4–6' matches both keywords '4' and '6'."""
    answer = "4–6"
    norm_ans = normalize(answer)

    keywords = ["4", "6"]
    for kw in keywords:
        norm_kw = normalize(kw)
        assert norm_kw in norm_ans, f"Expected '{norm_kw}' to match answer '{answer}'"


def test_wrong_answer_does_not_match() -> None:
    """Verify wrong or unrelated answers do not match."""
    norm_kw_inequality = normalize("n≥d")
    norm_kw_4 = normalize("4")
    norm_kw_6 = normalize("6")

    wrong_answers = [
        "The paper does not mention any dimension criteria.",
        "n <= d",
        r"$n \leq d$",
        "10 to 12 projects",
        "none of the above",
    ]

    for ans in wrong_answers:
        norm_ans = normalize(ans)
        assert norm_kw_inequality not in norm_ans
        assert norm_kw_4 not in norm_ans
        assert norm_kw_6 not in norm_ans
