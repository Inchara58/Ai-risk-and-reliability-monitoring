"""
AI Risk & Reliability Platform - Master Auditor Engine & Pre-Deployment Evidence Engine (Stage 4)
File: auditor_engine.py
Description: Master analytical engine executing the 9-stage unified audit workflow:
             1. DATA QUALITY
             2. CLEANING
             3. FEATURE ANALYSIS
             4. MODEL PERFORMANCE
             5. FAIRNESS
             6. DRIFT
             7. ROBUSTNESS
             8. RISK
             9. RECOMMENDATIONS
             Produces comprehensive pre-deployment evidence dossiers with 100% empirical grounding.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from data_auditor import DataAuditor
from data_cleaner import DataCleaner
from bias_detector import BiasDetector
from drift_detector import DriftDetector
from robustness_tester import RobustnessTester
from risk_calculator import RiskCalculator


class MasterAuditorEngine:
    """
    Master Pre-Deployment Evidence & AI Reliability Auditor.
    Coordinates all modular auditing engines to generate structured analytical evidence.
    """

    def __init__(self, df: pd.DataFrame, 
                 dataset_name: str = "Dataset.csv", 
                 project_id: str = "1",
                 model_bundle: Optional[Dict[str, Any]] = None,
                 model_record: Optional[Dict[str, Any]] = None,
                 baseline_df: Optional[pd.DataFrame] = None,
                 protected_attribute: Optional[str] = None,
                 target_column: Optional[str] = None):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.project_id = project_id
        self.model_bundle = model_bundle
        self.model_record = model_record
        self.baseline_df = baseline_df
        self.protected_attribute = protected_attribute
        self.target_column = target_column

    def run_full_audit(self) -> Dict[str, Any]:
        """
        Executes the full 9-stage empirical audit workflow in strict sequence.
        Returns evidence payload and full dossier data.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_id = f"audit_{int(time.time())}_{self.project_id}"
        issues_collected: List[Dict[str, Any]] = []
        recommendations_collected: List[Dict[str, Any]] = []

        total_rows, total_cols = self.df.shape

        # =========================================================================
        # STAGE 1: DATA QUALITY AUDIT
        # =========================================================================
        data_auditor = DataAuditor(self.df)
        quality_res = data_auditor.audit()

        # Check issues for Stage 1
        if quality_res["missing_count"] > 0:
            issues_collected.append({
                "stage": "DATA_QUALITY",
                "category": "Missing Values",
                "severity": "CRITICAL" if quality_res["missing_pct"] > 15 else ("HIGH" if quality_res["missing_pct"] > 3 else "MEDIUM"),
                "title": f"{quality_res['missing_count']} Missing Values Detected ({quality_res['missing_pct']}%)",
                "affected_items": [col for col, c in quality_res["null_counts"].items() if c > 0],
                "action": "Execute automated imputation in Cleaning Workbench before training models."
            })
            recommendations_collected.append({
                "pillar": "Data Quality",
                "priority": "HIGH" if quality_res["missing_pct"] > 5 else "MEDIUM",
                "title": "Remediate Missing Feature Values",
                "recommendation": f"Apply statistical imputation (median for numeric, mode for categorical) across {len([c for c, count in quality_res['null_counts'].items() if count > 0])} column(s) with missing data.",
                "evidence": f"Total missing cells: {quality_res['missing_count']} ({quality_res['missing_pct']}%)"
            })

        if quality_res["duplicate_count"] > 0:
            issues_collected.append({
                "stage": "DATA_QUALITY",
                "category": "Duplicate Records",
                "severity": "HIGH" if quality_res["duplicate_pct"] > 5 else "MEDIUM",
                "title": f"{quality_res['duplicate_count']} Duplicate Rows Found ({quality_res['duplicate_pct']}%)",
                "affected_items": ["Dataset Rows"],
                "action": "Purge duplicate rows to prevent overfitting and sample inflation."
            })
            recommendations_collected.append({
                "pillar": "Data Quality",
                "priority": "HIGH",
                "title": "Purge Duplicate Rows",
                "recommendation": f"Remove {quality_res['duplicate_count']} exact duplicate record(s) before model training.",
                "evidence": f"Duplicate rows represent {quality_res['duplicate_pct']}% of total dataset volume."
            })

        if quality_res["constant_columns"]:
            issues_collected.append({
                "stage": "DATA_QUALITY",
                "category": "Zero Variance",
                "severity": "LOW",
                "title": f"Constant Single-Value Columns: {', '.join(quality_res['constant_columns'])}",
                "affected_items": quality_res["constant_columns"],
                "action": "Drop constant columns as they provide zero predictive variance."
            })
            recommendations_collected.append({
                "pillar": "Feature Engineering",
                "priority": "LOW",
                "title": "Drop Zero-Variance Features",
                "recommendation": f"Remove constant columns ({', '.join(quality_res['constant_columns'])}) from training feature set.",
                "evidence": f"{len(quality_res['constant_columns'])} column(s) have only 1 unique value across all rows."
            })

        # =========================================================================
        # STAGE 2: CLEANING & REPAIR PLAN
        # =========================================================================
        repair_plan = DataCleaner.generate_repair_plan(self.df)

        # =========================================================================
        # STAGE 3: FEATURE ANALYSIS
        # =========================================================================
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df.select_dtypes(include=["object"]).columns.tolist()

        feature_profiles = []
        for col in self.df.columns:
            stats_entry = quality_res["column_stats"].get(col, {})
            feature_profiles.append({
                "name": col,
                "dtype": stats_entry.get("dtype", str(self.df[col].dtype)),
                "missing_pct": stats_entry.get("missing_pct", 0.0),
                "unique_values": stats_entry.get("unique_values", 0),
                "is_numeric": col in numeric_features,
                "is_id": stats_entry.get("is_id_column", False),
                "is_constant": stats_entry.get("is_constant", False),
                "outlier_count": stats_entry.get("outlier_count", 0),
                "outlier_pct": stats_entry.get("outlier_pct", 0.0),
                "mean": stats_entry.get("mean"),
                "std": stats_entry.get("std"),
                "min": stats_entry.get("min"),
                "max": stats_entry.get("max")
            })

        # =========================================================================
        # STAGE 4: MODEL PERFORMANCE
        # =========================================================================
        model_eval_stage = {}
        if self.model_bundle and "metrics" in self.model_bundle:
            m_metrics = self.model_bundle["metrics"]
            model_eval_stage = {
                "model_available": True,
                "model_name": m_metrics.get("algorithm_name", "Trained Model"),
                "problem_type": self.model_bundle.get("problem_type", "classification"),
                "target_column": self.model_bundle.get("target_column", self.target_column),
                "accuracy": m_metrics.get("accuracy"),
                "precision": m_metrics.get("precision"),
                "recall": m_metrics.get("recall"),
                "f1_score": m_metrics.get("f1_score"),
                "roc_auc": m_metrics.get("roc_auc"),
                "r2_score": m_metrics.get("r2_score"),
                "rmse": m_metrics.get("rmse"),
                "mae": m_metrics.get("mae"),
                "cv_mean_accuracy": m_metrics.get("cv_mean_accuracy"),
                "cv_std": m_metrics.get("cv_std"),
                "feature_importances": m_metrics.get("feature_importances", [])[:8],
                "confusion_matrix": m_metrics.get("confusion_matrix"),
                "status": "EVALUATED",
                "message": "Empirical model performance calculated on holdout validation set."
            }
        elif self.model_record and "metrics" in self.model_record:
            rec_metrics = self.model_record["metrics"]
            model_eval_stage = {
                "model_available": True,
                "model_name": self.model_record.get("name", "Trained Model"),
                "problem_type": self.model_record.get("problem_type", "classification"),
                "target_column": self.model_record.get("target_column", self.target_column),
                "accuracy": rec_metrics.get("accuracy"),
                "precision": rec_metrics.get("precision"),
                "recall": rec_metrics.get("recall"),
                "f1_score": rec_metrics.get("f1_score"),
                "roc_auc": rec_metrics.get("roc_auc"),
                "r2_score": rec_metrics.get("r2_score"),
                "rmse": rec_metrics.get("rmse"),
                "mae": rec_metrics.get("mae"),
                "feature_importances": rec_metrics.get("feature_importances", [])[:8],
                "confusion_matrix": rec_metrics.get("confusion_matrix"),
                "status": "EVALUATED",
                "message": "Empirical model performance loaded from registered model artifact."
            }
        else:
            model_eval_stage = {
                "model_available": False,
                "status": "UNAVAILABLE",
                "message": "Model evaluation unavailable — train a model in Model Factory first.",
                "accuracy": None,
                "f1_score": None,
                "roc_auc": None
            }
            recommendations_collected.append({
                "pillar": "Model Factory",
                "priority": "LOW",
                "title": "Train Benchmark Model",
                "recommendation": "Train a model in the Model Factory to generate empirical holdout metrics and enable pre-deployment stress testing.",
                "evidence": "No active trained model artifact is attached to this audit session."
            })

        # =========================================================================
        # STAGE 5: FAIRNESS & DEMOGRAPHIC PARITY
        # =========================================================================
        # Determine target column
        target_col = self.target_column or (self.model_bundle.get("target_column") if self.model_bundle else None)
        if not target_col:
            target_col = self.df.columns[-1]

        bias_detector = BiasDetector(
            df=self.df,
            protected_col=self.protected_attribute,
            target_col=target_col
        )
        fairness_res = bias_detector.evaluate_bias()

        if fairness_res.get("fairness_evaluable"):
            if not fairness_res.get("is_compliant", True):
                issues_collected.append({
                    "stage": "FAIRNESS",
                    "category": "Demographic Disparity",
                    "severity": "HIGH",
                    "title": f"EEOC Disparate Impact Alert (Ratio: {fairness_res['disparate_impact_ratio']})",
                    "affected_items": [str(fairness_res.get("protected_attribute"))],
                    "action": "Review selection criteria and apply bias mitigation reweighting before production signoff."
                })
                recommendations_collected.append({
                    "pillar": "Fairness & Ethics",
                    "priority": "HIGH",
                    "title": "Address Demographic Selection Disparity",
                    "recommendation": f"Investigate demographic selection rate imbalance for '{fairness_res.get('protected_attribute')}' ({fairness_res['unprivileged_group']} vs {fairness_res['privileged_group']}).",
                    "evidence": f"Disparate Impact Ratio is {fairness_res['disparate_impact_ratio']}, violating the standard 80% rule [0.80 - 1.25]."
                })
        else:
            recommendations_collected.append({
                "pillar": "Fairness Governance",
                "priority": "LOW",
                "title": "Protected Attribute Status",
                "recommendation": "No demographic or protected group features detected. If this is a regulated domain model, ensure fair sampling across demographic segments.",
                "evidence": "No suitable demographic attributes available for disparate impact calculations."
            })

        # =========================================================================
        # STAGE 6: DATA & POPULATION DRIFT
        # =========================================================================
        if self.baseline_df is not None and len(self.baseline_df) > 10:
            drift_engine = DriftDetector(
                self.baseline_df, 
                self.df,
                baseline_label="Baseline Reference Dataset (v1.0)",
                current_label=f"Current Dataset ({self.dataset_name})"
            )
        else:
            # Explicit baseline split experiment (e.g. 70/30 split or chronological)
            split_idx = int(len(self.df) * 0.5)
            if split_idx > 10:
                base_part = self.df.iloc[:split_idx]
                curr_part = self.df.iloc[split_idx:]
                drift_engine = DriftDetector(
                    base_part, 
                    curr_part,
                    baseline_label="Baseline Partition (First 50% Benchmark Slice)",
                    current_label="Evaluation Partition (Second 50% Production Slice)"
                )
            else:
                drift_engine = DriftDetector(self.df, self.df)

        drift_res = drift_engine.detect_drift()

        if drift_res.get("drifted_features_count", 0) > 0:
            issues_collected.append({
                "stage": "DRIFT",
                "category": "Distribution Shift",
                "severity": "HIGH" if drift_res["overall_status"] == "CRITICAL_DRIFT_ALERT" else "MEDIUM",
                "title": f"Statistical Drift Detected in {drift_res['drifted_features_count']} Feature(s)",
                "affected_items": drift_res.get("drifted_features", []),
                "action": "Monitor shifted features in production pipeline and consider updating reference distributions."
            })
            recommendations_collected.append({
                "pillar": "Data Drift",
                "priority": "MEDIUM",
                "title": "Stabilize Drifted Features",
                "recommendation": f"Calibrate feature distributions for drifted variables: {', '.join(drift_res.get('drifted_features', [])[:4])}.",
                "evidence": f"Overall PSI is {drift_res['overall_psi_score']:.4f} with {drift_res['drifted_features_count']} feature(s) showing significant distribution divergence."
            })

        # =========================================================================
        # STAGE 7: MODEL ROBUSTNESS & STRESS TESTING
        # =========================================================================
        estimator_or_bundle = self.model_bundle or (self.model_record.get("bundle") if self.model_record else None)
        prob_type = self.model_bundle.get("problem_type", "classification") if self.model_bundle else "classification"
        
        robustness_tester = RobustnessTester(
            model_bundle_or_estimator=estimator_or_bundle,
            df=self.df,
            target_col=target_col,
            problem_type=prob_type
        )
        robustness_res = robustness_tester.evaluate_robustness()

        if robustness_res.get("model_available"):
            if robustness_res.get("robustness_score", 100) < 70:
                issues_collected.append({
                    "stage": "ROBUSTNESS",
                    "category": "Adversarial Vulnerability",
                    "severity": "HIGH",
                    "title": f"High Sensitivity to Input Perturbations (Robustness Score: {robustness_res['robustness_score']})",
                    "affected_items": ["Model Inputs"],
                    "action": "Apply adversarial data augmentation or feature regularization during training."
                })
                recommendations_collected.append({
                    "pillar": "Robustness",
                    "priority": "HIGH",
                    "title": "Enhance Noise Resilience",
                    "recommendation": f"Model performance degrades by {robustness_res.get('avg_degradation_pct', 0)}% under noise injection. Retrain with noise-augmented datasets.",
                    "evidence": f"Robustness rating is {robustness_res.get('robustness_rating')}."
                })

        # =========================================================================
        # STAGE 8: TRANSPARENT RULE-BASED RISK ENGINE
        # =========================================================================
        risk_engine = RiskCalculator(
            quality_metrics=quality_res,
            fairness_metrics=fairness_res,
            drift_metrics=drift_res,
            robustness_metrics=robustness_res,
            model_metrics=model_eval_stage
        )
        risk_res = risk_engine.calculate_all()

        # =========================================================================
        # STAGE 9: RECOMMENDATION ENGINE & PRE-DEPLOYMENT DOSSIER
        # =========================================================================
        # If no issues, add positive recommendation
        if not issues_collected:
            recommendations_collected.append({
                "pillar": "System Reliability",
                "priority": "INFO",
                "title": "System Clean & Ready",
                "recommendation": "All data completeness and statistical stability checks passed successfully. System is certified for production staging.",
                "evidence": f"Trust Score: {risk_res['trust_score']}/100, Zero Critical Defects Detected."
            })

        # Synthesize full Pre-Deployment Dossier Report
        dossier = {
            "audit_id": audit_id,
            "project_id": self.project_id,
            "dataset_name": self.dataset_name,
            "dataset_version": "v1.0",
            "model_name": model_eval_stage.get("model_name", "None"),
            "model_version": "v1" if model_eval_stage.get("model_available") else "N/A",
            "audit_timestamp": timestamp,
            "deployment_verdict": risk_res["deployment_verdict"],
            "deployment_recommendation": risk_res["deployment_recommendation"],
            "trust_score": risk_res["trust_score"],
            "readiness_score": risk_res["readiness_score"],
            "risk_level": risk_res["risk_level"],
            "stages": {
                "1_data_quality": {
                    "stage_name": "Data Quality & Integrity",
                    "status": "PASS" if quality_res["completeness_score"] >= 85 else ("WARNING" if quality_res["completeness_score"] >= 65 else "FAIL"),
                    "completeness_score": quality_res["completeness_score"],
                    "total_rows": quality_res["total_rows"],
                    "total_cols": quality_res["total_cols"],
                    "missing_count": quality_res["missing_count"],
                    "missing_pct": quality_res["missing_pct"],
                    "duplicate_count": quality_res["duplicate_count"],
                    "duplicate_pct": quality_res["duplicate_pct"],
                    "outlier_count": quality_res["outlier_count"],
                    "constant_columns": quality_res["constant_columns"],
                    "id_columns": quality_res["id_columns"],
                    "high_cardinality_cols": quality_res["high_cardinality_cols"],
                    "memory_mb": quality_res["memory_mb"],
                    "column_stats": quality_res["column_stats"]
                },
                "2_cleaning": {
                    "stage_name": "Data Cleaning & Repair Plan",
                    "status": "ACTION_REQUIRED" if repair_plan["total_issues_detected"] > 0 else "CLEAN",
                    "total_issues_detected": repair_plan["total_issues_detected"],
                    "estimated_rows_impacted": repair_plan["estimated_rows_impacted"],
                    "repair_plan": repair_plan
                },
                "3_feature_analysis": {
                    "stage_name": "Feature Profiling & Cardinality",
                    "status": "PASS",
                    "numeric_features_count": len(numeric_features),
                    "categorical_features_count": len(categorical_features),
                    "feature_profiles": feature_profiles
                },
                "4_model_performance": {
                    "stage_name": "Model Performance & Validation",
                    "status": model_eval_stage.get("status", "UNAVAILABLE"),
                    "data": model_eval_stage
                },
                "5_fairness": {
                    "stage_name": "Fairness & Demographic Parity",
                    "status": fairness_res.get("status", "UNAVAILABLE"),
                    "data": fairness_res
                },
                "6_drift": {
                    "stage_name": "Data Drift & Population Stability",
                    "status": drift_res.get("overall_status", "DATASET_STABLE"),
                    "data": drift_res
                },
                "7_robustness": {
                    "stage_name": "Adversarial Robustness & Noise Resilience",
                    "status": robustness_res.get("status", "UNAVAILABLE"),
                    "data": robustness_res
                },
                "8_risk": {
                    "stage_name": "Composite Risk & Governance Assessment",
                    "status": risk_res["risk_level"],
                    "data": risk_res
                },
                "9_recommendations": {
                    "stage_name": "Evidence Recommendations & Remediation Actions",
                    "status": "COMPLETE",
                    "issues_count": len(issues_collected),
                    "recommendations_count": len(recommendations_collected),
                    "issues": issues_collected,
                    "recommendations": recommendations_collected
                }
            }
        }

        return dossier


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 70)
    print("      AEGIS AI — MASTER AUDITOR & EVIDENCE ENGINE DEMO")
    print("=" * 70)

    # Demo with synthetic Banking dataset
    test_df = pd.DataFrame({
        "Customer_ID": [f"CUST_{i}" for i in range(200)],
        "Age": [None if i % 15 == 0 else 25 + (i % 40) for i in range(200)],
        "Income": [None if i % 20 == 0 else 40000 + (i * 500) for i in range(200)],
        "Credit_Score": [600 + (i * 2) % 250 for i in range(200)],
        "Gender": ["Male" if i % 2 == 0 else "Female" for i in range(200)],
        "Loan_Approved": [1 if (600 + (i * 2) % 250) > 680 else 0 for i in range(200)]
    })

    engine = MasterAuditorEngine(
        df=test_df,
        dataset_name="Banking_Demo.csv",
        project_id="prj_test_101",
        protected_attribute="Gender",
        target_column="Loan_Approved"
    )

    dossier = engine.run_full_audit()

    print(f"\nAudit ID: {dossier['audit_id']}")
    print(f"Trust Score      : {dossier['trust_score']} / 100")
    print(f"Readiness Score  : {dossier['readiness_score']} / 100")
    print(f"Risk Level       : {dossier['risk_level']}")
    print(f"Verdict          : {dossier['deployment_verdict']} — {dossier['deployment_recommendation']}")
    print(f"\nIssues Detected  : {len(dossier['stages']['9_recommendations']['issues'])}")
    for iss in dossier['stages']['9_recommendations']['issues']:
        print(f" * [{iss['severity']}] {iss['title']} -> {iss['action']}")

    print(f"\nRecommendations  : {len(dossier['stages']['9_recommendations']['recommendations'])}")
    for rec in dossier['stages']['9_recommendations']['recommendations']:
        print(f" * [{rec['priority']}] {rec['title']}: {rec['recommendation']}")
    print("=" * 70)
