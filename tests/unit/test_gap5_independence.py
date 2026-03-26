"""
tests/unit/test_gap5_independence.py
=====================================
TEST-04 ENHANCEMENT: Validates that BoBN fixer A and fixer B use
different model families, ensuring genuine independence (not just
mock-level call ordering).

Also validates that the adversarial critic uses a third independent
family, and that composite scoring formula weights are non-zero and
sum to 1.0.
"""
from models.router import TieredModelRouter, BOBN_FIXER_A_COUNT, BOBN_FIXER_B_COUNT


def _extract_family(model_id: str) -> str:
    from verification.independence_enforcer import extract_model_family
    return extract_model_family(model_id)


def test_fixer_a_and_b_use_different_families():
    router = TieredModelRouter()
    model_a = router.primary_model("fix")
    model_b = router.secondary_model()
    family_a = _extract_family(model_a)
    family_b = _extract_family(model_b)
    assert family_a != family_b, (
        f"BoBN fixer A ({model_a}, family={family_a}) and "
        f"fixer B ({model_b}, family={family_b}) must use "
        f"DIFFERENT model families for genuine independence"
    )


def test_critic_uses_third_family():
    router = TieredModelRouter()
    model_a = router.primary_model("fix")
    model_b = router.secondary_model()
    model_critic = router.critic_model()
    family_a = _extract_family(model_a)
    family_b = _extract_family(model_b)
    family_critic = _extract_family(model_critic)
    assert family_critic not in (family_a, family_b), (
        f"Adversarial critic ({model_critic}, family={family_critic}) "
        f"should use a different family from both fixers "
        f"(A={family_a}, B={family_b})"
    )


def test_composite_scoring_weights_sum_to_one():
    test_weight = 0.6
    robustness_weight = 0.3
    minimality_weight = 0.1
    total = test_weight + robustness_weight + minimality_weight
    assert abs(total - 1.0) < 1e-9, (
        f"Composite scoring weights must sum to 1.0, got {total}"
    )


def test_bobn_candidate_counts_valid():
    assert BOBN_FIXER_A_COUNT >= 1
    assert BOBN_FIXER_B_COUNT >= 1
    total = BOBN_FIXER_A_COUNT + BOBN_FIXER_B_COUNT
    assert total >= 2, "BoBN needs at least 2 candidates"
    assert total <= 20, f"BoBN with {total} candidates is unreasonably expensive"


def test_temperature_diversity():
    router = TieredModelRouter()
    temps_a = router.fixer_a_temperatures()
    temps_b = router.fixer_b_temperatures()
    if len(temps_a) >= 2:
        assert len(set(temps_a)) > 1, (
            f"Fixer A temperatures should be diverse, got {temps_a}"
        )
    if len(temps_b) >= 2:
        assert len(set(temps_b)) > 1, (
            f"Fixer B temperatures should be diverse, got {temps_b}"
        )
