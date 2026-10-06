"""
AEGIS AI - Stage 5 Automated Integration Test Suite
File: test_stage5_integration.py
Description: Validates:
  1. Deployment Gate 7-Pillar Evidence & Transparent Readiness Checklist
  2. Human Decision Room operations with reason recording & immutable ledger
  3. Post-Deployment Monitoring Center with empirical drift & KS calculations
  4. Performance degradation tracking & "performance unavailable without ground-truth" enforcement
  5. Incident creation and human resolution workflow
  6. Visual Project Lineage graph generation
"""

import os
import json
import time
import pandas as pd
import numpy as np
import pytest

from app import app
import auth
from monitoring_engine import MonitoringEngine
import model_trainer


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_01_training_and_deployment_gate_evidence(client):
    """Verifies that a model can be trained and evaluated across all 7 deployment gate pillars."""
    # 1. Setup baseline project dataset
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "Feature_A": np.random.normal(50, 10, 300),
        "Feature_B": np.random.normal(100, 15, 300),
        "Region": np.random.choice(["North", "South", "East", "West"], size=300),
        "Outcome": np.random.choice([0, 1], size=300, p=[0.7, 0.3])
    })

    # Train model
    trainer = model_trainer.ModelTrainer(
        df=baseline_df,
        target_col="Outcome",
        problem_type="classification"
    )
    res = trainer.train_single_model("rf")
    model_record = auth.create_model_record(
        project_id="test_prj_stage5",
        name=res["metrics"]["algorithm_name"],
        algorithm="rf",
        problem_type="classification",
        target_column="Outcome",
        metrics=res["metrics"],
        feature_config={"selected_features": ["Feature_A", "Feature_B", "Region"]}
    )
    assert model_record is not None
    assert "id" in model_record

    # Query Deployment Gate endpoint
    resp = client.get(f"/api/v1/deployment/gate/test_prj_stage5")
    assert resp.status_code == 200
    gate_data = resp.get_json()
    assert gate_data["status"] == "success"
    assert "deployment_status" in gate_data
    assert gate_data["human_review_required"] is True
    assert "HUMAN REVIEW REQUIRED" in gate_data["human_notice"]

    # Verify all 7 key dimensions are present
    dims = {d["id"]: d for d in gate_data["evidence_dimensions"]}
    required_dims = ["dataset_quality", "model_performance", "fairness", "drift", "robustness", "risk", "business_impact"]
    for req in required_dims:
        assert req in dims, f"Missing required evidence dimension: {req}"
        assert "status" in dims[req]
        assert "evidence" in dims[req]
        assert "timestamp" in dims[req]
        assert "details" in dims[req]

    # Verify forbidden words are not automatically claimed
    assert gate_data["deployment_status"] not in ["CERTIFIED", "SAFE", "COMPLIANT"]


def test_02_human_decision_room_logging(client):
    """Verifies that human decisions with mandatory justification are recorded in immutable ledger."""
    project_id = "test_prj_stage5"

    # 1. Attempt decision without justification -> should fail
    fail_resp = client.post("/api/v1/deployment/review", json={
        "project_id": project_id,
        "decision": "APPROVE",
        "reason": ""  # Blank reason
    })
    assert fail_resp.status_code == 400

    # 2. Record valid APPROVE decision
    ok_resp = client.post("/api/v1/deployment/review", json={
        "project_id": project_id,
        "decision": "APPROVE",
        "reason": "All 7 lifecycle evidence dimensions satisfied with trust score > 90 and minimal drift.",
        "reviewer_name": "Chief Risk Officer",
        "reviewer_role": "AI Governance Lead",
        "reviewer_email": "cro@enterprise.org"
    })
    assert ok_resp.status_code == 200
    dec_data = ok_resp.get_json()
    assert dec_data["status"] == "success"
    assert dec_data["decision_record"]["decision"] == "APPROVE"
    assert dec_data["decision_record"]["reviewer_name"] == "Chief Risk Officer"

    # 3. Query gate to verify ledger history
    gate_resp = client.get(f"/api/v1/deployment/gate/{project_id}")
    gate_data = gate_resp.get_json()
    assert gate_data["decisions_count"] >= 1
    assert gate_data["decisions_history"][0]["decision"] == "APPROVE"


def test_03_monitoring_drift_and_incident_generation(client):
    """Verifies continuous drift monitoring, incident dispatch when PSI > 0.10, and resolution workflow."""
    project_id = "test_prj_stage5"
    np.random.seed(101)

    # 1. Baseline dataset
    base_df = pd.DataFrame({
        "Delivery_Days": np.random.normal(3.5, 0.8, 400),
        "Package_Weight": np.random.normal(12.0, 3.0, 400),
        "Delayed": np.random.choice([0, 1], size=400, p=[0.8, 0.2])
    })

    # 2. Current production batch with severe drift on Delivery_Days (PSI > 0.25)
    curr_df = pd.DataFrame({
        "Delivery_Days": np.random.normal(7.8, 1.9, 400),  # Heavy shift
        "Package_Weight": np.random.normal(12.2, 3.1, 400),
        "Delayed": np.random.choice([0, 1], size=400, p=[0.8, 0.2])
    })

    # Run monitoring engine directly
    engine = MonitoringEngine(
        baseline_df=base_df,
        current_df=curr_df,
        target_col="Delayed",
        project_id=project_id,
        baseline_label="Baseline_Q1.csv",
        current_label="Production_Batch_Q2.csv"
    )
    cycle_res = engine.run_monitoring_cycle()

    # Check empirical PSI calculation
    assert cycle_res["overall_status"] in ["WARNING_MONITOR", "INCIDENT_ALERT"]
    assert cycle_res["incidents_count"] > 0
    drift_details = cycle_res["drift"]["feature_drift_details"]
    assert "Delivery_Days" in drift_details
    assert drift_details["Delivery_Days"]["psi_score"] >= 0.10

    # 3. Run monitoring via REST API endpoint
    api_resp = client.post("/api/v1/monitoring/run", json={
        "project_id": project_id,
        "baseline_dataset_name": "Baseline_Q1.csv",
        "baseline_records": base_df.to_dict(orient="records"),
        "current_dataset_name": "Production_Batch_Q2.csv",
        "current_records": curr_df.to_dict(orient="records"),
        "include_ground_truth": True
    })
    assert api_resp.status_code == 200
    mon_res = api_resp.get_json()
    assert mon_res["status"] == "success"
    assert "run_id" in mon_res["result"]

    # 4. Check incidents endpoint
    inc_resp = client.get(f"/api/v1/monitoring/incidents/{project_id}")
    inc_data = inc_resp.get_json()
    assert inc_data["status"] == "success"
    assert inc_data["count"] > 0
    open_inc = inc_data["incidents"][0]
    assert open_inc["status"] == "OPEN"

    # 5. Resolve incident
    res_resp = client.post(f"/api/v1/monitoring/incidents/{open_inc['id']}/resolve", json={
        "resolution_notes": "Recalibrated logistics routing model on new distribution; verified stable."
    })
    assert res_resp.status_code == 200
    assert res_resp.get_json()["status"] == "success"


def test_04_performance_monitoring_ground_truth_policy(client):
    """
    Verifies that when ground truth is absent, the system strictly outputs
    'Performance monitoring unavailable until ground-truth outcomes are available.'
    without fabricating performance.
    """
    project_id = "test_prj_stage5"
    base_df = pd.DataFrame({
        "Score_A": [10, 20, 30, 40, 50],
        "Target": [0, 1, 0, 1, 0]
    })
    # Current batch WITHOUT target label
    curr_df_no_label = pd.DataFrame({
        "Score_A": [12, 22, 32, 42, 52]
    })

    engine = MonitoringEngine(
        baseline_df=base_df,
        current_df=curr_df_no_label,
        target_col="Target",
        project_id=project_id
    )
    res = engine.run_monitoring_cycle()

    assert res["performance"]["performance_available"] is False
    assert "Performance monitoring unavailable until ground-truth outcomes are available." in res["performance"]["message"]
    assert res["performance"]["baseline_metric"] is None
    assert res["performance"]["current_metric"] is None


def test_05_visual_project_lineage_graph(client):
    """Verifies complete end-to-end lineage graph construction."""
    project_id = "test_prj_stage5"
    resp = client.get(f"/api/v1/lineage/{project_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    lineage = data["lineage"]

    # Verify core stages in graph
    stages = [node["stage"] for node in lineage["nodes"]]
    assert "INTAKE" in stages
    assert "CLEANING" in stages
    assert "MODELING" in stages
    assert "AUDIT" in stages
    assert "DEPLOYMENT_GATE" in stages
    assert "PRODUCTION" in stages
    assert "MONITORING" in stages

    assert len(lineage["edges"]) >= 5


if __name__ == "__main__":
    pytest.main(["-v", "test_stage5_integration.py"])
