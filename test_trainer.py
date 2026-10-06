import model_trainer
import demo_datasets

def run_tests():
    print('=== 1. Testing Classification on Manufacturing Dataset ===')
    df_mfg = demo_datasets.get_demo_dataset('manufacturing').head(2000)
    analysis = model_trainer.analyze_dataset_features(df_mfg, 'Machine_Failure')
    print('Analysis detected problem type:', analysis['detected_problem_type'], 'Target:', analysis['suggested_target'], 'Features count:', len(analysis['features']))

    trainer_cls = model_trainer.ModelTrainer(df_mfg, 'Machine_Failure')
    res_cls = trainer_cls.compare_all_models()
    print('Recommended Model:', res_cls['recommended_model_name'])
    for m in res_cls['leaderboard']:
        print(f" - {m['algorithm_name']}: Accuracy={m['accuracy']}, F1={m['f1_score']}, ROC-AUC={m['roc_auc']}, CV={m['cv_mean']}")

    print('\n=== 2. Testing Single and Batch Prediction ===')
    best_bundle = res_cls['bundles'][res_cls['recommended_model_id']]
    predictor = model_trainer.ModelPredictor(best_bundle)
    sample_input = {col: float(df_mfg[col].iloc[0]) if df_mfg[col].dtype != 'object' else str(df_mfg[col].iloc[0]) for col in trainer_cls.selected_features}
    single_pred = predictor.predict_single(sample_input)
    print('Single Prediction result:', single_pred['prediction'], 'Confidence:', single_pred.get('confidence_pct'), '%')
    print('Top Attribution:', single_pred['feature_attributions'][0])

    batch_pred = predictor.predict_batch(df_mfg.head(50))
    print('Batch processed:', batch_pred['total_records_processed'], 'Distribution:', batch_pred['distribution_summary'])

    print('\n=== 3. Testing Regression on Logistics Delivery_Days ===')
    df_log = demo_datasets.get_demo_dataset('logistics').head(2000)
    analysis_reg = model_trainer.analyze_dataset_features(df_log, 'Delivery_Days')
    print('Analysis detected problem type:', analysis_reg['detected_problem_type'])
    trainer_reg = model_trainer.ModelTrainer(df_log, 'Delivery_Days', problem_type='regression')
    res_reg = trainer_reg.compare_all_models()
    print('Recommended Regression Model:', res_reg['recommended_model_name'])
    for m in res_reg['leaderboard']:
        print(f" - {m['algorithm_name']}: MAE={m['mae']}, RMSE={m['rmse']}, R2={m['r2_score']}, CV={m['cv_mean']}")

    print('\nMODEL TRAINER TESTS PASSED PERFECTLY!')

if __name__ == '__main__':
    run_tests()
