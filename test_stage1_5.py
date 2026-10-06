import app

def run_tests():
    client = app.app.test_client()

    print("=== 1. Testing GET /api/v1/demo/datasets ===")
    r = client.get('/api/v1/demo/datasets')
    print("Status:", r.status_code)
    datasets = r.get_json()['datasets']
    for d in datasets:
        print(f"- {d['id']}: {d['dataset_name']} ({d['row_count']} rows, {d['feature_count']} cols, Target: {d['target_column']})")

    print("\n=== 2. Testing All 5 Demo Datasets Load, Preview, Audit, Clean ===")
    for ds_id in ['manufacturing', 'ecommerce', 'banking', 'healthcare', 'logistics']:
        print(f"\n--- Testing {ds_id.upper()} ---")
        # Preview
        r_prev = client.get(f'/api/v1/demo/datasets/{ds_id}')
        assert r_prev.status_code == 200, f'Preview failed for {ds_id}'
        prev_data = r_prev.get_json()['preview']
        snap = prev_data['audit_snapshot']
        print(f"Preview {ds_id}: rows={prev_data['metadata']['row_count']}, cols={prev_data['metadata']['feature_count']}, missing={snap['missing_pct']}%, dupes={snap['duplicate_pct']}%, outliers={snap['outlier_count']}, balance={snap['target_imbalance']}, trust={snap['trust_score']}")
        
        # Load
        r_load = client.post(f'/api/v1/demo/datasets/{ds_id}/load', json={'projectId': 'prj_test'})
        assert r_load.status_code == 200, f'Load failed for {ds_id}'
        
        # Audit
        r_audit = client.post(f'/api/v1/demo/datasets/{ds_id}/audit')
        assert r_audit.status_code == 200, f'Audit failed for {ds_id}'
        audit_res = r_audit.get_json()
        print(f"Audit {ds_id}: rows={audit_res['audit']['total_rows']}, trust_score={audit_res['risk_metrics']['trust_score']}, risk_level={audit_res['risk_metrics']['risk_level']}")
        
        # Clean
        r_clean = client.post(f'/api/v1/demo/datasets/{ds_id}/clean', json={'actions': ['impute_missing', 'remove_duplicates', 'cap_outliers', 'standardize_categories']})
        assert r_clean.status_code == 200, f'Clean failed for {ds_id}'
        clean_data = r_clean.get_json()
        print(f"Clean {ds_id}: Cleaned {clean_data['cleaned_rows']} rows across {len(clean_data['cleaning_logs'])} cleaning transforms, sample={len(clean_data['sample_cleaned'])}")

    print("\nALL 5 DEMO DATASETS PASSED ACCEPTANCE VERIFICATION PERFECTLY!")

if __name__ == '__main__':
    run_tests()
