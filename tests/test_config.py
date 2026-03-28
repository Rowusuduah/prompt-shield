"""
Tests for prompt_shield/config.py

No ML dependencies.
"""
from __future__ import annotations

import pytest
import yaml

from prompt_shield.config import load_config


def write_yaml(path, content: dict):
    with open(path, "w") as f:
        yaml.dump(content, f)


class TestLoadConfig:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_loads_valid_config(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [
                {
                    "name": "support",
                    "function": "myapp.llm.handle",
                    "test_inputs": ["What is the return policy?"],
                }
            ]
        })
        cfg = load_config(str(cfg_path))
        assert len(cfg["prompts"]) == 1
        assert cfg["prompts"][0]["name"] == "support"

    def test_empty_file_returns_empty_prompts(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        cfg_path.write_text("")
        cfg = load_config(str(cfg_path))
        assert cfg["prompts"] == []

    def test_missing_prompts_key_defaults_to_empty_list(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {"output": {"certificate": "cert.json"}})
        cfg = load_config(str(cfg_path))
        assert cfg["prompts"] == []

    def test_raises_on_missing_name(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [
                {"function": "myapp.fn", "test_inputs": ["input"]}
            ]
        })
        with pytest.raises(ValueError, match="name"):
            load_config(str(cfg_path))

    def test_raises_on_missing_test_inputs(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [{"name": "p1", "function": "myapp.fn"}]
        })
        with pytest.raises(ValueError, match="test_inputs"):
            load_config(str(cfg_path))

    def test_raises_on_empty_test_inputs(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [{"name": "p1", "function": "fn", "test_inputs": []}]
        })
        with pytest.raises(ValueError, match="non-empty"):
            load_config(str(cfg_path))

    def test_preserves_optional_fields(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [{
                "name": "p1",
                "function": "fn",
                "test_inputs": ["input"],
                "threshold": 0.25,
                "variants_per_input": 12,
                "levels": ["lexical"],
            }]
        })
        cfg = load_config(str(cfg_path))
        p = cfg["prompts"][0]
        assert p["threshold"] == 0.25
        assert p["variants_per_input"] == 12
        assert p["levels"] == ["lexical"]

    def test_preserves_output_section(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [{"name": "p", "function": "f", "test_inputs": ["i"]}],
            "output": {"certificate": "./cert.json"},
        })
        cfg = load_config(str(cfg_path))
        assert cfg["output"]["certificate"] == "./cert.json"

    def test_multiple_prompts(self, tmp_path):
        cfg_path = tmp_path / "shield.yaml"
        write_yaml(cfg_path, {
            "prompts": [
                {"name": "p1", "function": "f1", "test_inputs": ["i1"]},
                {"name": "p2", "function": "f2", "test_inputs": ["i2", "i3"]},
            ]
        })
        cfg = load_config(str(cfg_path))
        assert len(cfg["prompts"]) == 2
        assert cfg["prompts"][1]["name"] == "p2"
