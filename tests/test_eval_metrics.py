"""Acceptance criteria §6.1-6.3: hand-computed metric fixture, edge cases, and
order-preservation. Pure arithmetic, no chunk/span knowledge -- every value here
is hand-checkable (docs/design/phase-3.md §6.1)."""

import math

import pytest

from docsense.eval.metrics import dcg, mean_or_none, ndcg_at_k, recall_at_k, reciprocal_rank

# The design doc's hand-computed fixture (§6.1).
GAINS = [0, 2, 0, 1]
FIRST_COVERED_RANKS = [1, 3, None]
IDEAL_GAINS = [2, 1, 1]
K = 4
THRESHOLD = 1


def test_recall_at_1_is_zero():
    assert recall_at_k(FIRST_COVERED_RANKS, 1) == pytest.approx(0.0, rel=1e-6)


def test_recall_at_4_is_two_thirds():
    assert recall_at_k(FIRST_COVERED_RANKS, 4) == pytest.approx(2 / 3, rel=1e-6)


def test_reciprocal_rank_at_4_is_half():
    assert reciprocal_rank(GAINS, 4) == pytest.approx(0.5, rel=1e-6)


def test_reciprocal_rank_at_1_is_zero():
    assert reciprocal_rank(GAINS, 1) == pytest.approx(0.0, rel=1e-6)


def test_dcg_of_ideal_gains():
    expected = 2 / 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert dcg(IDEAL_GAINS) == pytest.approx(3.130930, rel=1e-6)
    assert dcg(IDEAL_GAINS) == pytest.approx(expected, rel=1e-6)


def test_dcg_of_gains():
    expected = 2 / math.log2(3) + 1 / math.log2(5)
    assert dcg(GAINS) == pytest.approx(1.692536, rel=1e-6)
    assert dcg(GAINS) == pytest.approx(expected, rel=1e-6)


def test_ndcg_at_4_matches_hand_computation():
    assert ndcg_at_k(GAINS, IDEAL_GAINS, K) == pytest.approx(0.540586, rel=1e-6)
    assert ndcg_at_k(GAINS, IDEAL_GAINS, K) == pytest.approx(
        dcg(GAINS) / dcg(IDEAL_GAINS), rel=1e-6
    )


# --- Edge cases (§6.2) -------------------------------------------------------


def test_ndcg_of_ideal_against_itself_is_one():
    assert ndcg_at_k(IDEAL_GAINS, IDEAL_GAINS, 3) == pytest.approx(1.0)


def test_ndcg_perfect_short_ranking_is_one():
    assert ndcg_at_k([1, 0], [1], 2) == pytest.approx(1.0)


def test_reversed_ranking_scores_strictly_lower_than_ideal():
    ideal_order = [2, 1, 1]
    reversed_order = [1, 1, 2]
    ideal_score = ndcg_at_k(ideal_order, ideal_order, 3)
    reversed_score = ndcg_at_k(reversed_order, ideal_order, 3)
    assert reversed_score < ideal_score


def test_k_exceeding_list_length_does_not_raise_or_index_out_of_range():
    assert recall_at_k([0, None], 100) == pytest.approx(0.5)
    assert reciprocal_rank([0, 1], 100) == pytest.approx(0.5)
    assert ndcg_at_k([1, 0], [1, 1], 100) >= 0.0


def test_recall_at_k_empty_sequence_raises():
    with pytest.raises(ValueError):
        recall_at_k([], 1)


def test_ndcg_empty_ideal_gains_raises():
    with pytest.raises(ValueError):
        ndcg_at_k([1, 0], [], 1)


def test_ndcg_ideal_gains_summing_to_zero_raises():
    with pytest.raises(ValueError):
        ndcg_at_k([1, 0], [0, 0], 1)


@pytest.mark.parametrize("k", [0, -1])
def test_k_less_than_one_raises(k):
    with pytest.raises(ValueError):
        recall_at_k([0], k)
    with pytest.raises(ValueError):
        reciprocal_rank([1], k)
    with pytest.raises(ValueError):
        ndcg_at_k([1], [1], k)


def test_mean_or_none_empty_is_none():
    assert mean_or_none([]) is None


def test_mean_or_none_nonempty():
    assert mean_or_none([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# --- Order-preservation (§6.3) -----------------------------------------------


def test_ndcg_is_order_preserving_not_resorted():
    gains = [2, 0, 1]
    permuted = [1, 0, 2]
    ideal = [2, 1]
    assert ndcg_at_k(gains, ideal, 3) != ndcg_at_k(permuted, ideal, 3)


def test_reciprocal_rank_is_order_preserving_not_resorted():
    gains = [0, 0, 2]
    permuted = [2, 0, 0]
    assert reciprocal_rank(gains, 3) != reciprocal_rank(permuted, 3)


def test_ndcg_does_not_mutate_arguments():
    gains = [2, 0, 1]
    ideal_gains = [1, 2, 0]
    gains_copy = list(gains)
    ideal_copy = list(ideal_gains)
    ndcg_at_k(gains, ideal_gains, 3)
    assert gains == gains_copy
    assert ideal_gains == ideal_copy
