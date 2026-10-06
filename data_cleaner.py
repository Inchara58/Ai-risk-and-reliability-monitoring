"""
AI Risk & Reliability Platform - Automated Data Cleaning Engine (Stage 4)
File: data_cleaner.py
Description: Enterprise data repair and transformation engine.
             Provides Before Snapshot -> Repair Plan -> Apply Changes -> After Snapshot.
             Records exact column, row, and cell modification deltas with real data transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Tuple, Optional


class DataCleaner:
    """
    Automated data cleansing engine providing robust data preparation rules
    for enterprise ML pipelines with full audit telemetry.
    """

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.cleaning_log: List[Dict[str, Any]] = []
        self.columns_modified: set = set()
        self.values_modified_count: int = 0
        self.rows_dropped_count: int = 0

    @classmethod
    def generate_repair_plan(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Scans dataset and generates a proposed Repair Plan without applying changes yet.
        Identifies missing values, duplicates, outliers, and formatting issues.
        """
        total_rows, total_cols = df.shape
        plan_actions = []

        # 1. Check Duplicates
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            plan_actions.append({
                "action_id": "remove_duplicates",
                "title": "Purge Duplicate Records",
                "category": "Duplication",
                "severity": "HIGH" if (duplicate_count / max(1, total_rows)) > 0.05 else "MEDIUM",
                "affected_columns": list(df.columns),
                "affected_rows_count": duplicate_count,
                "description": f"Detected {duplicate_count} exact duplicate rows ({round(duplicate_count/max(1,total_rows)*100, 1)}% of dataset). Purge to eliminate model sample bias.",
                "default_enabled": True,
                "options": ["drop_all_duplicates", "keep_first"]
            })

        # 2. Check Missing Values per column
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                missing_pct = round((missing_count / total_rows) * 100, 2)
                is_num = pd.api.types.is_numeric_dtype(df[col])
                rec_strategy = "median" if is_num else "mode"
                
                plan_actions.append({
                    "action_id": f"impute_missing_{col}",
                    "title": f"Impute Missing Values in '{col}'",
                    "category": "Missingness",
                    "severity": "CRITICAL" if missing_pct > 20 else ("HIGH" if missing_pct > 5 else "MEDIUM"),
                    "affected_columns": [col],
                    "affected_values_count": missing_count,
                    "affected_rows_count": missing_count,
                    "description": f"Column '{col}' has {missing_count} missing entries ({missing_pct}%). Impute with {rec_strategy}.",
                    "default_enabled": True,
                    "strategy": rec_strategy,
                    "is_numeric": is_num,
                    "available_strategies": ["median", "mean", "zero", "drop_rows"] if is_num else ["mode", "unknown_category", "drop_rows"]
                })

        # 3. Check Outliers (1.5x IQR) for numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            valid = df[col].dropna()
            if len(valid) > 10:
                q25 = float(valid.quantile(0.25))
                q75 = float(valid.quantile(0.75))
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - (1.5 * iqr)
                    upper = q75 + (1.5 * iqr)
                    outliers_count = int(((valid < lower) | (valid > upper)).sum())
                    if outliers_count > 0:
                        outlier_pct = round((outliers_count / len(valid)) * 100, 2)
                        plan_actions.append({
                            "action_id": f"handle_outliers_{col}",
                            "title": f"Cap Outliers in '{col}'",
                            "category": "Outliers",
                            "severity": "MEDIUM" if outlier_pct < 10 else "HIGH",
                            "affected_columns": [col],
                            "affected_values_count": outliers_count,
                            "affected_rows_count": outliers_count,
                            "description": f"Found {outliers_count} IQR outliers ({outlier_pct}%) outside [{round(lower, 2)}, {round(upper, 2)}]. Cap to valid IQR boundaries.",
                            "default_enabled": True,
                            "strategy": "cap",
                            "lower_bound": round(lower, 3),
                            "upper_bound": round(upper, 3),
                            "available_strategies": ["cap", "drop_rows", "keep"]
                        })

        # 4. Check Categorical Whitespace & Casing
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            valid_str = df[col].dropna().astype(str)
            has_whitespace = any(s != s.strip() for s in valid_str.head(100))
            if has_whitespace:
                plan_actions.append({
                    "action_id": f"normalize_text_{col}",
                    "title": f"Standardize Casing & Whitespace in '{col}'",
                    "category": "Standardization",
                    "severity": "LOW",
                    "affected_columns": [col],
                    "affected_values_count": len(valid_str),
                    "affected_rows_count": len(valid_str),
                    "description": f"Standardize string trimming and title casing on '{col}' to unify categorical groups.",
                    "default_enabled": True,
                    "available_strategies": ["trim_and_titlecase", "lowercase", "uppercase"]
                })

        return {
            "total_issues_detected": len(plan_actions),
            "estimated_rows_impacted": sum(a.get("affected_rows_count", 0) for a in plan_actions),
            "plan_actions": plan_actions
        }

    def execute_cleaning_plan(self, plan_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes real data transformations based on chosen plan configurations.
        Returns full telemetry: Before vs After metrics, column deltas, and execution log.
        """
        initial_shape = self.raw_df.shape
        initial_missing = int(self.raw_df.isnull().sum().sum())
        initial_duplicates = int(self.raw_df.duplicated().sum())

        config = plan_config or {}
        custom_strategies = config.get("strategies", {})

        # Step 1: Duplicate Purging
        if config.get("remove_duplicates", True):
            dup_count = int(self.df.duplicated().sum())
            if dup_count > 0:
                self.df.drop_duplicates(inplace=True)
                self.rows_dropped_count += dup_count
                self.cleaning_log.append({
                    "step": "DEDUPLICATION",
                    "target_column": "ALL",
                    "action": "Purged Duplicate Rows",
                    "rows_affected": dup_count,
                    "values_affected": dup_count * initial_shape[1],
                    "details": f"Removed {dup_count} exact duplicate rows from dataset."
                })

        # Step 2: Missing Value Imputation
        if config.get("impute_missing", True):
            for col in self.df.columns:
                missing_num = int(self.df[col].isnull().sum())
                if missing_num > 0:
                    self.columns_modified.add(col)
                    self.values_modified_count += missing_num
                    col_strat = custom_strategies.get(f"missing_{col}")

                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        strat = col_strat or config.get("numeric_missing_strategy", "median")
                        if strat == "mean":
                            val = float(self.df[col].mean())
                        elif strat == "zero":
                            val = 0.0
                        else:
                            val = float(self.df[col].median()) if not np.isnan(self.df[col].median()) else 0.0
                        
                        self.df[col] = self.df[col].fillna(val)
                        self.cleaning_log.append({
                            "step": "IMPUTATION",
                            "target_column": col,
                            "action": f"Numeric Imputation ({strat})",
                            "rows_affected": missing_num,
                            "values_affected": missing_num,
                            "details": f"Replaced {missing_num} missing values in '{col}' with {strat} value ({round(val, 2)})."
                        })
                    else:
                        strat = col_strat or config.get("categorical_missing_strategy", "mode")
                        if strat == "mode" and not self.df[col].mode().empty:
                            val = str(self.df[col].mode()[0])
                        else:
                            val = "Unknown"
                        
                        self.df[col] = self.df[col].fillna(val)
                        self.cleaning_log.append({
                            "step": "IMPUTATION",
                            "target_column": col,
                            "action": f"Categorical Imputation ({strat})",
                            "rows_affected": missing_num,
                            "values_affected": missing_num,
                            "details": f"Replaced {missing_num} missing values in '{col}' with '{val}'."
                        })

        # Step 3: Outlier Handling
        if config.get("handle_outliers", True):
            outlier_action = config.get("outlier_action", "cap")
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                valid = self.df[col].dropna()
                if len(valid) > 10:
                    q25 = float(valid.quantile(0.25))
                    q75 = float(valid.quantile(0.75))
                    iqr = q75 - q25
                    if iqr > 0:
                        lower = q25 - (1.5 * iqr)
                        upper = q75 + (1.5 * iqr)
                        mask = (self.df[col] < lower) | (self.df[col] > upper)
                        out_count = int(mask.sum())
                        if out_count > 0:
                            self.columns_modified.add(col)
                            if outlier_action == "cap":
                                self.df[col] = np.clip(self.df[col], lower, upper)
                                self.values_modified_count += out_count
                                self.cleaning_log.append({
                                    "step": "OUTLIER_CAPPING",
                                    "target_column": col,
                                    "action": "IQR Boundary Capping",
                                    "rows_affected": out_count,
                                    "values_affected": out_count,
                                    "details": f"Capped {out_count} outlier values in '{col}' to range [{round(lower, 2)}, {round(upper, 2)}]."
                                })
                            elif outlier_action == "drop":
                                self.df = self.df[~mask].copy()
                                self.rows_dropped_count += out_count
                                self.cleaning_log.append({
                                    "step": "OUTLIER_DROPPING",
                                    "target_column": col,
                                    "action": "Purged Outlier Rows",
                                    "rows_affected": out_count,
                                    "values_affected": out_count * self.df.shape[1],
                                    "details": f"Dropped {out_count} outlier rows based on IQR bounds for '{col}'."
                                })

        # Step 4: Categorical Text Standardization
        if config.get("standardize_strings", True):
            str_cols = self.df.select_dtypes(include=["object"]).columns
            for col in str_cols:
                self.df[col] = self.df[col].astype(str).str.strip().str.title()
            if len(str_cols) > 0:
                self.cleaning_log.append({
                    "step": "TEXT_STANDARDIZATION",
                    "target_column": "CATEGORICAL_COLUMNS",
                    "action": "Standardize Casing and Trim Whitespace",
                    "rows_affected": len(self.df),
                    "values_affected": len(self.df) * len(str_cols),
                    "details": f"Standardized whitespace trimming and title casing across {len(str_cols)} categorical columns."
                })

        # Compute After Metrics
        final_shape = self.df.shape
        final_missing = int(self.df.isnull().sum().sum())
        final_duplicates = int(self.df.duplicated().sum())

        return {
            "status": "success",
            "before_snapshot": {
                "total_rows": initial_shape[0],
                "total_cols": initial_shape[1],
                "missing_values": initial_missing,
                "missing_pct": round((initial_missing / max(1, initial_shape[0] * initial_shape[1])) * 100, 2),
                "duplicate_rows": initial_duplicates,
                "duplicate_pct": round((initial_duplicates / max(1, initial_shape[0])) * 100, 2)
            },
            "after_snapshot": {
                "total_rows": final_shape[0],
                "total_cols": final_shape[1],
                "missing_values": final_missing,
                "missing_pct": round((final_missing / max(1, final_shape[0] * final_shape[1])) * 100, 2),
                "duplicate_rows": final_duplicates,
                "duplicate_pct": round((final_duplicates / max(1, final_shape[0])) * 100, 2)
            },
            "telemetry": {
                "columns_modified": list(self.columns_modified),
                "columns_modified_count": len(self.columns_modified),
                "rows_modified_count": self.rows_dropped_count,
                "values_modified_count": self.values_modified_count,
                "rows_dropped_count": self.rows_dropped_count,
                "total_actions_applied": len(self.cleaning_log)
            },
            "cleaning_log": self.cleaning_log
        }

    def clean_all(self) -> Tuple[pd.DataFrame, List[str]]:
        """Legacy compatibility wrapper executing default full pipeline."""
        telemetry = self.execute_cleaning_plan()
        text_logs = [f"[{entry['step']}] {entry['details']}" for entry in self.cleaning_log]
        return self.df, text_logs


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 65)
    print("      AEGIS AI — UPGRADED DATA CLEANING ENGINE DEMO")
    print("=" * 65)

    raw_df = pd.DataFrame({
        "Account_ID": ["ACC-101", "ACC-102", "ACC-103", "ACC-101", "ACC-105", "ACC-106"],
        "Age": [34, 45, None, 34, 150, 29],
        "Annual_Income": [75000, 120000, None, 75000, 450000, 62000],
        "Gender": [" male ", "Female", "male", " male ", "female", "Female"],
        "Credit_Score": [720, 680, 590, 720, 840, 710]
    })

    print(f"\n[1] Generating Repair Plan for dataset ({raw_df.shape[0]} rows)...")
    plan = DataCleaner.generate_repair_plan(raw_df)
    print(f" -> Issues Detected: {plan['total_issues_detected']}")
    for action in plan['plan_actions']:
        print(f"    * [{action['severity']}] {action['title']}: {action['description']}")

    print("\n[2] Executing Cleaning Plan...")
    cleaner = DataCleaner(raw_df)
    results = cleaner.execute_cleaning_plan()

    print("\n[3] Clean Telemetry Deltas:")
    print(f" -> Rows Before: {results['before_snapshot']['total_rows']} -> After: {results['after_snapshot']['total_rows']}")
    print(f" -> Missing Values Before: {results['before_snapshot']['missing_values']} -> After: {results['after_snapshot']['missing_values']}")
    print(f" -> Values Modified: {results['telemetry']['values_modified_count']}")
    print(f" -> Columns Modified: {results['telemetry']['columns_modified']}")

    print("\n[4] Execution Log:")
    for log_item in results['cleaning_log']:
        print(f" -> [{log_item['step']}] {log_item['details']}")
    print("=" * 65)
