"""
Comprehensive Integration Test Suite for AEGIS AI — Stage 3:
- Synthetic Data Lab (SyntheticDataGenerator + REST API)
- AI Stress Lab (AIStressTester + Persistence + REST API)
- What-If Lab (Multi-Scenario Simulation + Deltas + REST API)
"""
import pytest
import os
import json
import pandas as pd
import numpy as np

from app import app
from synthetic_generator import SyntheticDataGenerator
from stress_tester import AIStressTester
from model_trainer import ModelTrainer, ModelPredictor, save_model_artifact
import auth

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_synthetic_data_generator_domain_preset():
    """Test standalone synthetic generator on Manufacturing preset with fault injection."""
    res = SyntheticDataGenerator.generate_synthetic_dataset(
        domain='Manufacturing',
        n_rows=200,
        target_type='classification',
        class_imbalance_ratio=0.2,
        missing_pct=0.05,
        duplicate_pct=0.02,
        outlier_pct=0.02,
        noise_pct=0.1,
        casing_inconsistency_pct=0.05
    )

    df = res['dataframe']
    manifest = res['manifest']

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 190
    assert 'Machine_Failure' in df.columns
    assert manifest['is_synthetic'] is True
    assert 'missing_values' in manifest['injections_applied']
    print(f"\nSynthetic Domain Test Passed: Generated {len(df)} rows, injections: {manifest['injections_applied']}")

def test_synthetic_data_generator_schema_clone():
    """Test synthetic generator cloning schema from an existing DataFrame."""
    active_df = pd.DataFrame({
        'Feature_A': np.random.normal(50, 10, 100),
        'Feature_B': np.random.choice(['Alpha', 'Beta', 'Gamma'], 100),
        'Target_Val': np.random.choice([0, 1], 100)
    })

    res = SyntheticDataGenerator.generate_synthetic_dataset(
        source_df=active_df,
        n_rows=300,
        target_type='classification',
        missing_pct=0.04
    )

    df = res['dataframe']
    assert len(df) == 300
    assert list(df.columns) == list(active_df.columns)
    assert res['manifest']['is_synthetic'] is True
    print(f"\nSynthetic Schema Clone Test Passed: Columns {df.columns.tolist()}")

def test_ai_stress_tester_and_whatif(tmp_path):
    """Test AIStressTester and counterfactual simulation on a real trained model bundle."""
    # 1. Train real model
    np.random.seed(42)
    df = pd.DataFrame({
        'sensor_1': np.random.normal(100, 15, 300),
        'sensor_2': np.random.uniform(10, 50, 300),
        'category': np.random.choice(['Low', 'Medium', 'High'], 300),
        'failure': np.random.choice([0, 1], 300, p=[0.75, 0.25])
    })

    trainer = ModelTrainer(df=df, target_col='failure', problem_type='classification', selected_features=['sensor_1', 'sensor_2', 'category'])
    train_res = trainer.train_single_model(algorithm_id='random_forest')
    bundle_path = save_model_artifact(project_id="1", model_id="test_rf_model", bundle=train_res['bundle'])

    assert os.path.exists(bundle_path)

    # 2. Stress Test with real holdout perturbations
    tester = AIStressTester()
    exp_res = tester.run_stress_experiment(
        bundle_path=bundle_path,
        holdout_df=df,
        stress_type='gaussian_noise',
        severity_pct=0.25,
        project_id=1
    )

    assert 'resilience_score' in exp_res
    assert 0 <= exp_res['resilience_score'] <= 100
    assert len(exp_res['sensitivity_curve']) == 5
    assert len(exp_res['recommendations']) > 0
    print(f"\nStress Tester Gaussian Noise Resilience: {exp_res['resilience_score']:.1f}, Degradation: {exp_res['degradation_pct']:.1f}%")

    # 3. Test What-If counterfactual inference
    predictor = ModelPredictor(train_res['bundle'])
    base_feats = {'sensor_1': 100.0, 'sensor_2': 30.0, 'category': 'Medium'}
    scen_a_feats = {'sensor_1': 150.0, 'sensor_2': 45.0, 'category': 'High'}
    scen_b_feats = {'sensor_1': 70.0, 'sensor_2': 15.0, 'category': 'Low'}

    pred_base = predictor.predict_single(base_feats)
    pred_a = predictor.predict_single(scen_a_feats)
    pred_b = predictor.predict_single(scen_b_feats)

    assert 'prediction' in pred_base
    assert 'prediction' in pred_a
    assert 'prediction' in pred_b
    print(f"\nWhat-If Simulation: Base={pred_base['prediction']}, A={pred_a['prediction']}, B={pred_b['prediction']}")

def test_stage3_api_endpoints(client):
    """Test Flask API endpoints for Stage 3."""
    # 1. POST /api/v1/synthetic/generate
    res = client.post('/api/v1/synthetic/generate', json={
        'source_mode': 'domain',
        'domain': 'Banking & Finance',
        'num_rows': 250,
        'target_type': 'classification',
        'fault_missing_pct': 0.05
    })
    assert res.status_code == 200, res.data
    data = json.loads(res.data)
    assert data['status'] == 'success'
    assert 'audit_manifest' in data
    assert data['audit_manifest']['is_synthetic'] is True
    assert len(data['preview_rows']) > 0

    # 2. POST /api/v1/synthetic/load-to-project
    res_load = client.post('/api/v1/synthetic/load-to-project', json={
        'project_id': 1,
        'dataset_name': 'Synthetic_Banking_Test.csv',
        'domain': 'Banking & Finance'
    })
    assert res_load.status_code == 200
    load_data = json.loads(res_load.data)
    assert load_data['status'] == 'success'
    assert load_data['total_rows'] == 250

    # 3. Train a model on this dataset to test stress and what-if endpoints
    target_col = load_data.get('target_column', 'Loan_Approved')
    res_train = client.post('/api/v1/models/train', json={
        'algorithm': 'rf',
        'target_column': target_col,
        'problem_type': 'classification',
        'feature_columns': [c for c in load_data['columns'] if c != target_col and not c.endswith('_ID')],
        'dataset_name': load_data['dataset_name'],
        'project_id': 1
    })
    assert res_train.status_code == 200, res_train.data
    train_data = json.loads(res_train.data)
    model_id = train_data['model']['id']

    # 4. POST /api/v1/stress/run-experiment
    res_stress = client.post('/api/v1/stress/run-experiment', json={
        'model_id': model_id,
        'stress_type': 'missing_values',
        'severity_pct': 0.20,
        'dataset_name': load_data['dataset_name'],
        'project_id': 1
    })
    assert res_stress.status_code == 200, res_stress.data
    stress_data = json.loads(res_stress.data)
    assert stress_data['status'] == 'success'
    assert 'resilience_score' in stress_data

    # 5. GET /api/v1/stress/experiments/<project_id>
    res_hist = client.get('/api/v1/stress/experiments/1')
    assert res_hist.status_code == 200
    hist_data = json.loads(res_hist.data)
    assert len(hist_data['experiments']) >= 1

    # 6. POST /api/v1/whatif/simulate
    sample_feat = {c: 10.0 for c in load_data['columns'] if c != target_col and not c.endswith('_ID')}
    res_whatif = client.post('/api/v1/whatif/simulate', json={
        'model_id': model_id,
        'baseline_features': sample_feat,
        'scenario_a_features': {k: v * 2 for k, v in sample_feat.items()},
        'scenario_b_features': {k: v * 0.5 for k, v in sample_feat.items()}
    })
    assert res_whatif.status_code == 200, res_whatif.data
    whatif_data = json.loads(res_whatif.data)
    assert whatif_data['status'] == 'success'
    assert 'baseline' in whatif_data
    assert 'scenario_a' in whatif_data
    assert 'scenario_b' in whatif_data
    print("\nAll Stage 3 REST API endpoints validated successfully!")

if __name__ == '__main__':
    pytest.main(['-v', 'test_stage3_integration.py'])
