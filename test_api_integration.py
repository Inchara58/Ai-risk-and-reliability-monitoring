import json
import sys
from app import app
from auth import init_db

init_db()

client = app.test_client()

print("=" * 60)
print("RUNNING STAGE 2 INTEGRATION TEST SUITE")
print("=" * 60)

# 1. Test Feature Analysis
print("\n[1] Testing Feature Analysis on Manufacturing Dataset...")
res = client.post('/api/v1/models/analyze-features', json={
    'projectId': 'prj_mfg_default',
    'datasetName': 'Manufacturing_Machine_Failure_100k.csv'
})
print("Status:", res.status_code)
data = res.get_json()
assert res.status_code == 200, f"Error: {data}"
assert data['status'] == 'success'
assert 'Machine_Failure' in data['columns']
assert 'Machine_ID' in data['id_columns']
print(f"Columns: {len(data['columns'])}, ID Columns: {data['id_columns']}, Recommended: {data['recommended_target']}")

# 2. Test Model Training (Benchmark All 3 Classification Models)
print("\n[2] Testing Model Training Benchmark (Classification: Machine_Failure)...")
train_res = client.post('/api/v1/models/train', json={
    'projectId': 'prj_mfg_default',
    'datasetName': 'Manufacturing_Machine_Failure_100k.csv',
    'targetColumn': 'Machine_Failure',
    'problemType': 'classification',
    'selectedFeatures': [c for c in data['columns'] if c not in ['Machine_Failure', 'Machine_ID']],
    'algorithm': 'all'
})
print("Status:", train_res.status_code)
train_data = train_res.get_json()
assert train_res.status_code == 200, f"Error: {train_data}"
assert train_data['status'] == 'success'
assert len(train_data['models']) == 3
best_model = train_data['best_model']
print(f"Benchmark completed successfully! Trained {len(train_data['models'])} models.")
print(f"Best model: {best_model['model_name']} ({best_model['version']}) with F1 = {best_model['metrics']['f1']:.4f}")

# 3. Test List Models for Project
print(f"\n[3] Testing Model Registry for Project 'prj_mfg_default'...")
list_res = client.get('/api/v1/models/projects/prj_mfg_default')
list_data = list_res.get_json()
assert list_res.status_code == 200
assert len(list_data['models']) >= 3
print(f"Total registered models for project: {len(list_data['models'])}")

# 4. Test Single Record Prediction
print(f"\n[4] Testing Single Record Prediction with Model '{best_model['model_id']}'...")
sample_input = {
    "Air_Temperature_K": 300.5,
    "Process_Temperature_K": 310.2,
    "Rotational_Speed_RPM": 1500,
    "Torque_Nm": 42.5,
    "Tool_Wear_Min": 110,
    "Vibration_mm_s": 2.4,
    "Voltage_V": 220.0,
    "Pressure_bar": 5.1,
    "Load_Percentage": 65.0,
    "Machine_Type": "Medium"
}
pred_res = client.post(f"/api/v1/models/{best_model['model_id']}/predict", json={
    'features': sample_input
})
pred_data = pred_res.get_json()
assert pred_res.status_code == 200, f"Error: {pred_data}"
assert pred_data['status'] == 'success'
print(f"Prediction: {pred_data['prediction']} (label: {pred_data['prediction_label']})")
print(f"Latency: {pred_data['latency_ms']:.2f} ms")
print(f"Probabilities: {pred_data.get('probabilities')}")
print(f"Local Attributions: {len(pred_data.get('local_attributions', []))} features calculated")

# 5. Test Batch Prediction
print(f"\n[5] Testing Batch Prediction (10 records)...")
batch_inputs = [sample_input.copy() for _ in range(10)]
for i, item in enumerate(batch_inputs):
    item['Air_Temperature_K'] += i * 2.0
    item['Tool_Wear_Min'] += i * 15

batch_res = client.post(f"/api/v1/models/{best_model['model_id']}/predict-batch", json={
    'records': batch_inputs
})
batch_data = batch_res.get_json()
assert batch_res.status_code == 200, f"Error: {batch_data}"
assert batch_data['status'] == 'success'
assert batch_data['total_records'] == 10
assert len(batch_data['scored_records']) == 10
assert 'AEGIS_Prediction' in batch_data['scored_records'][0]
print(f"Batch scoring completed! Processed {batch_data['total_records']} rows in {batch_data['latency_ms']:.2f} ms")
print("Sample scored row prediction:", batch_data['scored_records'][0]['AEGIS_Prediction'])

# 6. Test Regression Pipeline (Logistics Delivery Delay Continuous Target)
print("\n[6] Testing Regression Pipeline on Logistics Dataset...")
reg_train_res = client.post('/api/v1/models/train', json={
    'projectId': 'prj_logistics_default',
    'datasetName': 'Logistics_Delivery_Delays_100k.csv',
    'targetColumn': 'Delivery_Days',
    'problemType': 'regression',
    'selectedFeatures': ['Distance_KM', 'Package_Weight', 'Warehouse_Delay', 'Fuel_Cost', 'Processing_Time'],
    'algorithm': 'all'
})
reg_train_data = reg_train_res.get_json()
assert reg_train_res.status_code == 200, f"Error: {reg_train_data}"
assert reg_train_data['status'] == 'success'
best_reg = reg_train_data['best_model']
print(f"Regression benchmark completed! Best model: {best_reg['model_name']} with R² = {best_reg['metrics']['r2']:.4f}, RMSE = {best_reg['metrics']['rmse']:.2f}")

# Single record regression prediction
reg_pred_res = client.post(f"/api/v1/models/{best_reg['model_id']}/predict", json={
    'features': {
        'Distance_KM': 750.0,
        'Package_Weight': 45.0,
        'Warehouse_Delay': 5.2,
        'Fuel_Cost': 160.0,
        'Processing_Time': 6.0
    }
})
reg_pred_data = reg_pred_res.get_json()
assert reg_pred_res.status_code == 200
print(f"Regression prediction: {reg_pred_data['prediction']:.2f} days (latency: {reg_pred_data['latency_ms']:.2f} ms)")

print("\n" + "=" * 60)
print("ALL STAGE 2 INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
