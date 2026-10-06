"""
Comprehensive Integration Test Suite for AEGIS AI — Stage 4:
- Upgraded Master Data Integrity Auditor (DataAuditor)
- Upgraded Empirical Data Cleaner & Telemetry (DataCleaner)
- Upgraded Fairness & Bias Engine (BiasDetector with non-fabrication)
- Upgraded Population Stability & Drift Engine (DriftDetector with PSI & KS-test)
- Upgraded Adversarial Robustness Engine (RobustnessTester)
- Transparent Rule-Based Risk Engine (RiskCalculator)
- Master 9-Stage Auditor Engine (MasterAuditorEngine)
- Cross-Domain Acceptance Testing (Manufacturing, Banking, E-Commerce)
- Stage 4 Flask REST API Endpoints
"""

import pytest
import os
import json
import pandas as pd
import numpy as np

from app import app
from data_auditor import DataAuditor
from data_cleaner import DataCleaner
from bias_detector import BiasDetector
from drift_detector import DriftDetector
from robustness_tester import RobustnessTester
from risk_calculator import RiskCalculator
from auditor_engine import MasterAuditorEngine
import demo_datasets
import auth


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_data_auditor_empirical_metrics():
    """Test DataAuditor calculates real IQR outliers, null counts, duplicate ratios, and constant columns."""
    df = pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(100)],
        "Age": [None if i % 10 == 0 else 25 + (i * 2) for i in range(100)],
        "Income": [50000 if i != 99 else 2500000 for i in range(100)],  # 1 extreme outlier
        "Constant_Flag": [1] * 100,  # 1 constant column
        "Category": ["A" if i % 2 == 0 else "B" for i in range(100)]
    })

    auditor = DataAuditor(df)
    res = auditor.audit()

    assert res["total_rows"] == 100
    assert res["total_cols"] == 5
    assert res["missing_count"] == 10
    assert res["missing_pct"] == 2.0  # 10 / (100 * 5) * 100
    assert "Constant_Flag" in res["constant_columns"]
    assert "ID" in res["id_columns"]
    assert res["column_stats"]["Income"]["outlier_count"] >= 1
    assert 0 <= res["completeness_score"] <= 100
    print(f"\n[Test 1] DataAuditor: Completeness={res['completeness_score']}/100, ConstantCols={res['constant_columns']}")


def test_data_cleaner_before_repair_apply_after():
    """Test DataCleaner generates repair plan and records exact deltas before and after execution."""
    raw_df = pd.DataFrame({
        "User_ID": ["U1", "U2", "U3", "U1", "U5"],  # 1 duplicate row
        "Age": [25, None, 45, 25, 180],              # 1 missing, 1 outlier (180)
        "City": [" new york ", "London", "LONDON", " new york ", "Paris"],  # whitespace + casing
        "Income": [50000, 60000, None, 50000, 80000] # 1 missing
    })

    # 1. Repair Plan
    plan = DataCleaner.generate_repair_plan(raw_df)
    assert plan["total_issues_detected"] >= 3
    action_ids = [a["action_id"] for a in plan["plan_actions"]]
    assert "remove_duplicates" in action_ids

    # 2. Execution
    cleaner = DataCleaner(raw_df)
    res = cleaner.execute_cleaning_plan({
        "remove_duplicates": True,
        "impute_missing": True,
        "handle_outliers": True,
        "standardize_strings": True
    })

    assert res["before_snapshot"]["total_rows"] == 5
    assert res["before_snapshot"]["missing_values"] == 2
    assert res["before_snapshot"]["duplicate_rows"] == 1

    assert res["after_snapshot"]["total_rows"] == 4  # Duplicate purged
    assert res["after_snapshot"]["missing_values"] == 0  # Imputed
    assert res["after_snapshot"]["duplicate_rows"] == 0

    assert res["telemetry"]["values_modified_count"] >= 2
    assert res["telemetry"]["rows_dropped_count"] == 1
    assert len(res["cleaning_log"]) >= 3
    print(f"\n[Test 2] DataCleaner: Before Rows={res['before_snapshot']['total_rows']} -> After Rows={res['after_snapshot']['total_rows']}, Modified Values={res['telemetry']['values_modified_count']}")


def test_bias_detector_with_and_without_protected_attribute():
    """Test BiasDetector computes Disparate Impact when demographic attribute exists, and reports UNAVAILABLE without fabricating when absent."""
    # 1. With Protected Attribute (Gender)
    bank_df = pd.DataFrame({
        "Gender": ["Male"] * 100 + ["Female"] * 100,
        "Income": [50000] * 200,
        "Loan_Approved": [1] * 80 + [0] * 20 + [1] * 60 + [0] * 40  # 80% male vs 60% female
    })
    det_with = BiasDetector(bank_df, protected_col="Gender", target_col="Loan_Approved")
    res_with = det_with.evaluate_bias()

    assert res_with["has_protected_attribute"] is True
    assert res_with["fairness_evaluable"] is True
    assert res_with["disparate_impact_ratio"] == round(0.60 / 0.80, 4)  # 0.75
    assert res_with["is_compliant"] is False  # Fails 80% rule
    assert "Disparate Impact" in res_with["explanation"]
    assert res_with["fairness_score"] is not None

    # 2. Without Protected Attribute (Pure Sensor Data)
    mfg_df = pd.DataFrame({
        "Torque_Nm": [45.2, 50.1, 48.3, 52.0],
        "Rotational_Speed_rpm": [1500, 1480, 1520, 1490],
        "Failure": [0, 0, 1, 0]
    })
    det_without = BiasDetector(mfg_df, target_col="Failure")
    res_without = det_without.evaluate_bias()

    assert res_without["has_protected_attribute"] is False
    assert res_without["fairness_evaluable"] is False
    assert res_without["fairness_score"] is None
    assert "Fairness analysis cannot be performed with the available attributes" in res_without["explanation"]
    print(f"\n[Test 3] BiasDetector: With Attribute DI={res_with['disparate_impact_ratio']}, Without Attribute Evaluable={res_without['fairness_evaluable']}")


def test_drift_detector_psi_and_ks_test():
    """Test DriftDetector calculates PSI and two-sample Kolmogorov-Smirnov statistics against an explicit baseline."""
    np.random.seed(42)
    base = pd.DataFrame({
        "Feature_A": np.random.normal(50, 10, 500),
        "Feature_B": np.random.normal(100, 15, 500)
    })

    # Shift Feature_A significantly (Drift) but keep Feature_B stable
    curr = pd.DataFrame({
        "Feature_A": np.random.normal(75, 12, 500), # Drifted
        "Feature_B": np.random.normal(101, 15, 500)  # Stable
    })

    detector = DriftDetector(base, curr, baseline_label="v1.0", current_label="v2.0")
    res = detector.detect_drift()

    assert res["total_features_evaluated"] == 2
    assert "Feature_A" in res["feature_drift_details"]
    assert "Feature_B" in res["feature_drift_details"]

    feat_a_res = res["feature_drift_details"]["Feature_A"]
    feat_b_res = res["feature_drift_details"]["Feature_B"]

    assert feat_a_res["psi_score"] > 0.10  # Drift detected
    assert feat_a_res["ks_statistic"] > 0.30
    assert feat_a_res["is_statistically_significant"] is True
    assert feat_b_res["status"] == "STABLE"
    print(f"\n[Test 4] DriftDetector: Feature A PSI={feat_a_res['psi_score']:.4f} [{feat_a_res['status']}], Feature B PSI={feat_b_res['psi_score']:.4f} [{feat_b_res['status']}]")


def test_robustness_tester_perturbations_baseline_vs_perturbed():
    """Test RobustnessTester computes empirical perturbations and difference metrics on trained estimator."""
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    df = pd.DataFrame({
        "Sensor_1": np.random.normal(100, 20, 200),
        "Sensor_2": np.random.normal(50, 10, 200),
        "Target": np.random.choice([0, 1], size=200)
    })
    rf = RandomForestClassifier(random_state=42).fit(df[["Sensor_1", "Sensor_2"]], df["Target"])

    tester = RobustnessTester(rf, df, target_col="Target")
    res = tester.evaluate_robustness()

    assert res["model_available"] is True
    assert len(res["tests"]) >= 4
    for t in res["tests"]:
        assert "baseline_value" in t
        assert "perturbed_value" in t
        assert "difference" in t
        assert "degradation_pct" in t
        assert t["status"] in ["PASS", "WARNING", "FAIL"]

    # Test with no model
    no_model_tester = RobustnessTester(None, df)
    no_res = no_model_tester.evaluate_robustness()
    assert no_res["model_available"] is False
    assert "train a model in Model Factory first" in no_res["message"]
    print(f"\n[Test 5] RobustnessTester: Robustness Score={res['robustness_score']}/100, No Model Message='{no_res['message']}'")


def test_transparent_risk_calculator():
    """Test RiskCalculator produces transparent weighted 5-pillar breakdowns and non-ML rule-based rationale."""
    calc = RiskCalculator(
        quality_metrics={"completeness_score": 95.0, "missing_pct": 1.0, "duplicate_pct": 0.0, "outlier_pct": 2.0, "constant_columns": []},
        fairness_metrics={"fairness_evaluable": True, "fairness_score": 90.0, "disparate_impact_ratio": 0.92, "protected_attribute": "Gender"},
        drift_metrics={"overall_psi_score": 0.04, "drifted_features_count": 0},
        robustness_metrics={"model_available": True, "robustness_score": 88.0, "avg_degradation_pct": 6.0},
        model_metrics={"model_available": True, "metrics": {"accuracy": 0.94}}
    )
    res = calc.calculate_all()

    assert 0 <= res["trust_score"] <= 100
    assert 0 <= res["readiness_score"] <= 100
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert res["is_rule_based"] is True
    assert "deterministic" in res["methodology_note"].lower()
    assert len(res["pillar_breakdown"]) == 5
    print(f"\n[Test 6] RiskCalculator: Trust Score={res['trust_score']}/100, Risk Level={res['risk_level']}, Verdict={res['deployment_verdict']}")


def test_cross_industry_auditor_acceptance():
    """Acceptance test: Verify Manufacturing, Banking, and E-Commerce datasets produce distinct empirical findings."""
    # 1. Manufacturing (IoT Sensor dataset - no demographic column)
    mfg_df = demo_datasets.get_demo_dataset("manufacturing")
    engine_mfg = MasterAuditorEngine(mfg_df, dataset_name="Manufacturing_Machine_Failure_100k.csv", project_id="prj_mfg")
    dossier_mfg = engine_mfg.run_full_audit()

    assert dossier_mfg["stages"]["5_fairness"]["data"]["fairness_evaluable"] is False
    assert "No demographic or protected group features detected" in dossier_mfg["stages"]["5_fairness"]["data"]["explanation"]

    # 2. Banking (Loan Defaults - has Gender / demographic features)
    bank_df = demo_datasets.get_demo_dataset("banking")
    engine_bank = MasterAuditorEngine(bank_df, dataset_name="Banking_Loan_Defaults_100k.csv", project_id="prj_bank", protected_attribute="Gender")
    dossier_bank = engine_bank.run_full_audit()

    assert dossier_bank["stages"]["5_fairness"]["data"]["fairness_evaluable"] is True
    assert dossier_bank["stages"]["5_fairness"]["data"]["protected_attribute"] == "Gender"

    # 3. E-Commerce (Product Returns - Customer data)
    ecom_df = demo_datasets.get_demo_dataset("ecommerce")
    engine_ecom = MasterAuditorEngine(ecom_df, dataset_name="Ecommerce_Product_Returns_100k.csv", project_id="prj_ecom")
    dossier_ecom = engine_ecom.run_full_audit()

    assert dossier_mfg["stages"]["1_data_quality"]["total_cols"] != dossier_bank["stages"]["1_data_quality"]["total_cols"] or dossier_mfg["trust_score"] != dossier_bank["trust_score"]
    assert dossier_ecom["stages"]["1_data_quality"]["total_rows"] > 0
    print(f"\n[Test 7] Cross-Industry Acceptance Passed: Mfg Trust={dossier_mfg['trust_score']} (Fairness Evaluable={dossier_mfg['stages']['5_fairness']['data']['fairness_evaluable']}), Bank Trust={dossier_bank['trust_score']} (Fairness Evaluable={dossier_bank['stages']['5_fairness']['data']['fairness_evaluable']})")


def test_stage4_flask_rest_api_endpoints(client):
    """Test all Stage 4 Flask REST API endpoints."""
    # 1. POST /api/v1/auditor/repair-plan
    res_plan = client.post('/api/v1/auditor/repair-plan', json={
        'dataset_id': 'banking',
        'project_id': '1'
    })
    assert res_plan.status_code == 200, res_plan.data
    plan_data = json.loads(res_plan.data)
    assert plan_data['status'] == 'success'
    assert 'plan' in plan_data

    # 2. POST /api/v1/auditor/clean-and-transform
    res_clean = client.post('/api/v1/auditor/clean-and-transform', json={
        'dataset_id': 'banking',
        'project_id': '1',
        'cleaning_config': {
            'remove_duplicates': True,
            'impute_missing': True,
            'handle_outliers': True
        }
    })
    assert res_clean.status_code == 200, res_clean.data
    clean_data = json.loads(res_clean.data)
    assert clean_data['status'] == 'success'
    assert 'before_snapshot' in clean_data
    assert 'after_snapshot' in clean_data
    assert 'telemetry' in clean_data

    # 3. POST /api/v1/auditor/run-full-audit
    res_audit = client.post('/api/v1/auditor/run-full-audit', json={
        'dataset_id': 'banking',
        'project_id': '1',
        'protected_attribute': 'Gender'
    })
    assert res_audit.status_code == 200, res_audit.data
    audit_data = json.loads(res_audit.data)
    assert audit_data['status'] == 'success'
    assert 'dossier' in audit_data
    dossier = audit_data['dossier']
    assert 'trust_score' in dossier
    assert 'readiness_score' in dossier
    assert 'stages' in dossier
    assert len(dossier['stages']) == 9

    # 4. GET /api/v1/auditor/report/1
    res_rep = client.get('/api/v1/auditor/report/1')
    assert res_rep.status_code == 200, res_rep.data
    rep_data = json.loads(res_rep.data)
    assert rep_data['status'] == 'success'
    assert 'report' in rep_data

    # 5. GET /api/v1/auditor/reports/1
    res_list = client.get('/api/v1/auditor/reports/1')
    assert res_list.status_code == 200, res_list.data
    list_data = json.loads(res_list.data)
    assert list_data['status'] == 'success'
    assert len(list_data['reports']) >= 1

    print("\n[Test 8] All Stage 4 Flask REST API endpoints validated successfully!")


if __name__ == '__main__':
    pytest.main(['-v', 'test_stage4_integration.py'])
