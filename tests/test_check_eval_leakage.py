from scripts.check_eval_leakage import find_leaked_examples, shingles


def test_shingles_shorter_than_n_is_empty():
    assert shingles("too short", n=13) == set()


def test_identical_passage_is_leaked():
    eval_texts = ["the quick brown fox jumps over the lazy dog near the old stone bridge today"]
    train_texts = [
        "the quick brown fox jumps over the lazy dog near the old stone bridge today",
        "a completely unrelated sentence about transformer attention mechanisms in nlp",
    ]
    assert find_leaked_examples(eval_texts, train_texts, n=10) == [0]


def test_no_overlap_is_clean():
    eval_texts = ["gradient descent converges under a lipschitz smoothness assumption"]
    train_texts = ["batch normalization reduces internal covariate shift during training"]
    assert find_leaked_examples(eval_texts, train_texts, n=10) == []
