"""
prompt_shield/config.py — YAML configuration loader for prompt-shield.

shield.yaml example:
    prompts:
      - name: customer_support
        function: myapp.llm.handle_support
        test_inputs:
          - "What is the return policy?"
          - "How do I cancel my subscription?"
        threshold: 0.30
        variants_per_input: 8
        levels: ["lexical", "semantic"]
    output:
      certificate: ./shield_cert.json
"""
from __future__ import annotations

from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load and validate a shield.yaml config file.

    Returns the config dict. Raises FileNotFoundError if the file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config loading. pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Validate basic structure
    if "prompts" not in cfg:
        cfg["prompts"] = []

    for i, prompt_cfg in enumerate(cfg["prompts"]):
        if "name" not in prompt_cfg:
            raise ValueError(f"Prompt #{i} is missing required field 'name'")
        if "test_inputs" not in prompt_cfg:
            raise ValueError(f"Prompt '{prompt_cfg['name']}' is missing required field 'test_inputs'")
        if not isinstance(prompt_cfg["test_inputs"], list) or len(prompt_cfg["test_inputs"]) == 0:
            raise ValueError(f"Prompt '{prompt_cfg['name']}' test_inputs must be a non-empty list")

    return cfg
