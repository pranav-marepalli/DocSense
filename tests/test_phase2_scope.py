"""Acceptance criteria 15 and 16: config-completeness and generation-deferred
scope guards (decision 1)."""

import tomllib
from pathlib import Path

from docsense.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# §4's config key table, flattened to dotted paths.
EXPECTED_KEYS = {
    "variant_name",
    "seed",
    "chunks_path",
    "index_dir",
    "record_format",
    "embedder.type",
    "embedder.model_name",
    "embedder.dim",
    "embedder.normalize",
    "embedder.batch_size",
    "embedder.device",
    "embedder.max_seq_length",
    "embedder.query_prefix",
    "embedder.passage_prefix",
    "embedder.trust_remote_code",
    "index.type",
    "index.num_threads",
    "retrieval.top_k",
    "retrieval.max_chunks_per_paper",
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


def test_baseline_config_loads_and_matches_key_table():
    config = load_config("baseline")
    assert _flatten_keys(config) == EXPECTED_KEYS
    assert "generation" not in config


def test_offline_smoke_config_loads_and_matches_key_table():
    config = load_config("offline_smoke")
    assert _flatten_keys(config) == EXPECTED_KEYS
    assert "generation" not in config
    assert config["embedder"]["type"] == "hash"
    assert config["index"]["type"] == "numpy_flat_ip"


def test_generate_package_is_empty():
    generate_dir = REPO_ROOT / "src" / "docsense" / "generate"
    contents = sorted(
        p.name for p in generate_dir.iterdir() if not p.name.startswith("__pycache__")
    )
    assert contents == ["__init__.py"]
    assert (generate_dir / "__init__.py").read_text().strip() == ""


def test_numpy_is_a_declared_dependency():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "numpy" in pyproject["project"]["dependencies"]
