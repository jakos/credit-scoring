from dataclasses import asdict

import pytest

from credit_scoring.registry.policy import Policy, load_policy


def test_packaged_policy_is_independent_of_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert asdict(load_policy()) == dict(
        roc_auc_min_improvement=0.005,
        pr_auc_max_decrease=0.01,
        psi_ceiling=0.1,
        bootstrap_roc_auc_min=0.7,
        psi_bins=10,
        psi_smoothing=0.000001,
    )


@pytest.mark.parametrize("content", ["", "unknown = 1", "[", "roc_auc_min_improvement = 'bad'"])
def test_invalid_files(tmp_path, content) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_policy(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("psi_bins", 1),
        ("psi_bins", 2.5),
        ("psi_smoothing", 0),
        ("psi_ceiling", -1),
        ("bootstrap_roc_auc_min", 2),
        ("roc_auc_min_improvement", float("nan")),
        ("pr_auc_max_decrease", float("inf")),
        ("psi_bins", True),
        ("psi_smoothing", "bad"),
    ],
)
def test_invalid_policy_values(key, value) -> None:
    values = asdict(load_policy())
    values[key] = value
    with pytest.raises(ValueError):
        Policy(**values)
