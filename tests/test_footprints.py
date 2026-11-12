"""Admission charges must follow an identified FQL condition or retain the declaration."""
import pytest

from ops.jobs.footprints import ram_gb


def row(args=(), **extra):
    return {"script": "scripts/train_realworld_fql.py", "ram_gb": 64, "args": list(args), **extra}


@pytest.mark.parametrize("condition", [None, "unknown", "e2e_cnn_noaug", "e2e_cnn_extra"])
def test_unknown_conditions_retain_declared_ram(condition):
    assert ram_gb(row(fql_condition=condition)) == 64
    assert ram_gb(row(["--set", "condition=["])) == 64


@pytest.mark.parametrize("condition,expected", [("e2e_cnn", 8), ("dynamo", 3), ("ours", 3)])
def test_config_condition_and_override(tmp_path, condition, expected):
    config = tmp_path / "fql.yaml"
    config.write_text("condition: " + condition + "\n")
    assert ram_gb(row(["--config", str(config)], fql_condition="conflicting")) == expected
    assert ram_gb(row(["--config=" + str(config)])) == expected
    assert ram_gb(row(["--config", str(config), "--set", "condition=e2e_cnn"])) == 8
    assert ram_gb(row(["--config", str(config), "--set", "condition=unknown"])) == 64


@pytest.mark.parametrize("content", ["condition: unknown", "{}", "[]", "condition: [", "condition: null"])
def test_unresolved_config_keeps_declaration(tmp_path, content):
    config = tmp_path / "fql.yaml"
    config.write_text(content)
    assert ram_gb(row(["--config", str(config)], fql_condition="dynamo")) == 64
    assert ram_gb(row(["--config", str(config) + ".missing"])) == 64
