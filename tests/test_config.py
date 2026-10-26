import pytest

from docsense.config import load_config


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_load_config_reads_yaml(tmp_path, monkeypatch):
    import docsense.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    (tmp_path / "sample.yaml").write_text("chunk_size: 512\n")

    assert load_config("sample") == {"chunk_size": 512}
