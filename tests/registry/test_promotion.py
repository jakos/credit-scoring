import pytest

from credit_scoring.registry.policy import load_policy
from credit_scoring.registry.promotion import decide


@pytest.mark.parametrize(
    "auc,pr,psi,approved",
    [
        (0.805, 0.69, 0.099999, True),
        (0.804999, 0.69, 0.09, False),
        (0.805, 0.689999, 0.09, False),
        (0.805, 0.69, 0.1, False),
        (0.7, 0.6, 0.2, False),
    ],
)
def test_champion_boundaries(auc, pr, psi, approved) -> None:
    result = decide(
        load_policy(), dict(roc_auc=auc, pr_auc=pr, psi=psi), dict(roc_auc=0.8, pr_auc=0.7)
    )
    assert result.approved is approved
    if auc == 0.7:
        assert set(result.reasons) == {"roc_auc_improvement", "pr_auc_decrease", "psi"}


@pytest.mark.parametrize(
    "auc,psi,approved", [(0.7, 0.099999, True), (0.699999, 0, False), (0.7, 0.1, False)]
)
def test_bootstrap(auc, psi, approved) -> None:
    assert (
        decide(
            load_policy(), dict(roc_auc=auc, pr_auc=0.5, psi=psi, ks=float("nan"), gini=-1)
        ).approved
        is approved
    )


@pytest.mark.parametrize("key", ["roc_auc", "pr_auc", "psi"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_nonfinite_required_metrics(key, value) -> None:
    metrics = dict(roc_auc=0.9, pr_auc=0.8, psi=0.01)
    if value is None:
        del metrics[key]
    else:
        metrics[key] = value
    assert not decide(load_policy(), metrics).approved
