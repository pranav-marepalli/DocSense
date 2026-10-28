"""Acceptance criterion §6.15: config/eval.yaml key-set completeness and sanity."""

from docsense.config import load_config

# §4's config key table, flattened to dotted paths.
EXPECTED_KEYS = {
    "retriever_config",
    "seed",
    "eval.gold_path",
    "eval.runs_dir",
    "eval.record_format",
    "eval.k_values",
    "eval.primary_k",
    "eval.retrieval_top_k",
    "eval.binary_relevance_threshold",
    "eval.per_query_detail",
    "eval.slice_by",
    "eval.min_queries_warn",
    "labeling.queries_path",
    "labeling.pool_k",
    "labeling.snippet_chars",
}


def _flatten_keys(d: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key, value in d.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            keys |= _flatten_keys(value, prefix=f"{dotted}.")
        else:
            keys.add(dotted)
    return keys


def test_eval_config_loads_and_matches_key_table():
    config = load_config("eval")
    assert _flatten_keys(config) == EXPECTED_KEYS


def test_eval_config_retrieval_top_k_covers_all_k_values():
    config = load_config("eval")
    assert config["eval"]["retrieval_top_k"] >= max(config["eval"]["k_values"])


def test_eval_config_primary_k_is_one_of_k_values():
    config = load_config("eval")
    assert config["eval"]["primary_k"] in config["eval"]["k_values"]
