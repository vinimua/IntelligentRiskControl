from scripts import seed_deployment_kg as seed


def test_seed_contains_alert_codes_generated_by_deployment_observe():
    expected_alerts = {
        "CHALLENGER_AUC_REGRESSION",
        "CHALLENGER_KS_REGRESSION",
        "HIGH_DEPLOYMENT_SCORE_PSI",
        "BAD_RATE_DRIFT_HIGH",
        "TRAIN_VALID_GAP_LARGE",
        "RECOVERY_RATE_LOW",
        "OOT_DEPLOYMENT_RISK",
        "DISCRIMINATION_GATE_FAILED",
        "CALIBRATION_GATE_FAILED",
    }

    assert expected_alerts.issubset(seed.DEPLOYMENT_ALERTS)


def test_seed_relations_cover_alert_to_risk_to_strategy_path():
    alerts_with_risk = {item["alert"] for item in seed.ALERT_RISK_RELATIONS}
    risks_with_strategy = {item["risk"] for item in seed.RISK_STRATEGY_RELATIONS}
    seeded_risks = set(seed.DEPLOYMENT_RISKS)

    assert alerts_with_risk == set(seed.DEPLOYMENT_ALERTS)
    assert seeded_risks.issubset(risks_with_strategy)


def test_seed_strategies_have_stage_constraints():
    for strategy in seed.DEPLOYMENT_STRATEGIES.values():
        assert strategy["action_type"]
        assert strategy["allowed_stages"]

    allowed_pairs = set(seed.STAGE_STRATEGY_ALLOWS)
    for strategy_code, strategy in seed.DEPLOYMENT_STRATEGIES.items():
        for stage in strategy["allowed_stages"]:
            assert (stage, strategy_code) in allowed_pairs
