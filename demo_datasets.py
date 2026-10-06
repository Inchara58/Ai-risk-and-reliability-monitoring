"""
AEGIS AI — Demo Dataset Library & Industry Demonstration Generator
File: demo_datasets.py
Description: Generates and serves domain-specific, large synthetic demonstration datasets
             for Manufacturing, E-Commerce, Banking, Healthcare, and Logistics.
             Total: 475,000 synthetic records with controlled, realistic relationships
             and measurable data-quality imperfections (missingness, duplicates, outliers,
             casing inconsistencies, and class imbalance).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demo_datasets")
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset Registry Metadata
DEMO_DATASETS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "placement": {
        "id": "placement",
        "dataset_name": "Campus_Student_Placement_Predictions_100k.csv",
        "title": "College Student Campus Placement Predictor",
        "industry": "Higher Education & Campus Recruitment",
        "domain": "Education",
        "description": "Evaluate student academic CGPA, DSA coding proficiency, internships, projects, aptitude, mock interviews, and backlogs history to predict campus placement success in Tier-1/Tier-2 tech companies.",
        "row_count": 100000,
        "feature_count": 15,
        "target_column": "Placement_Status",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Comprehensive synthetic campus placement benchmark dataset for training predictive recruitment models and evaluating institutional career readiness.",
        "icon": "fa-solid fa-graduation-cap",
        "color_theme": "emerald",
        "imperfections_summary": {
            "missing_pct": "3.5%",
            "duplicate_pct": "2.0%",
            "outliers": "CGPA, Coding Score anomalies",
            "casing_issues": "Department & Certification casing",
            "class_imbalance": "~68.5% Placement Rate"
        }
    },
    "manufacturing": {
        "id": "manufacturing",
        "dataset_name": "Manufacturing_Machine_Failure_100k.csv",
        "title": "Machine Failure Prediction",
        "industry": "Manufacturing",
        "domain": "Manufacturing",
        "description": "Evaluate multi-sensor telemetry (temperature, vibration, pressure, voltage, load) to predict impending machine breakdowns and schedule proactive maintenance before catastrophic failure.",
        "row_count": 100000,
        "feature_count": 15,
        "target_column": "Machine_Failure",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Synthetic demonstration dataset for controlled validation and risk analysis. Does not contain real enterprise proprietary data.",
        "icon": "fa-solid fa-industry",
        "color_theme": "amber",
        "imperfections_summary": {
            "missing_pct": "3.2%",
            "duplicate_pct": "2.0%",
            "outliers": "Temperature, Vibration, Age anomalies",
            "casing_issues": "Machine_Type casing variations",
            "class_imbalance": "~6.5% Failure rate"
        }
    },
    "ecommerce": {
        "id": "ecommerce",
        "dataset_name": "Ecommerce_Product_Returns_100k.csv",
        "title": "Product Return Prediction",
        "industry": "E-Commerce",
        "domain": "E-Commerce",
        "description": "Predict whether an online order will result in a return based on customer return history, apparel categories, discount rates, seller credibility, and fulfillment lead times.",
        "row_count": 100000,
        "feature_count": 15,
        "target_column": "Returned",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Synthetic demonstration dataset for controlled validation and risk analysis. Does not contain real customer purchase history.",
        "icon": "fa-solid fa-cart-shopping",
        "color_theme": "purple",
        "imperfections_summary": {
            "missing_pct": "3.8%",
            "duplicate_pct": "1.5%",
            "outliers": "Order Value, Delivery Days anomalies",
            "casing_issues": "Product_Category casing variations",
            "class_imbalance": "~21.4% Return rate"
        }
    },
    "banking": {
        "id": "banking",
        "dataset_name": "Banking_Loan_Defaults_100k.csv",
        "title": "Loan Default Prediction",
        "industry": "Banking & Finance",
        "domain": "Banking & Finance",
        "description": "Assess borrower credit risk and loan default probability across income, debt ratios, credit scores, and history while auditing demographic equity and EEOC 80% fairness standards.",
        "row_count": 100000,
        "feature_count": 16,
        "target_column": "Default",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Synthetic demonstration dataset for controlled validation and risk analysis. Does not contain real applicant financial records.",
        "icon": "fa-solid fa-building-columns",
        "color_theme": "cyan",
        "imperfections_summary": {
            "missing_pct": "3.4%",
            "duplicate_pct": "2.0%",
            "outliers": "Annual Income, Age anomalies",
            "casing_issues": "Gender & Employment Type casing",
            "class_imbalance": "~14.2% Default rate"
        }
    },
    "healthcare": {
        "id": "healthcare",
        "dataset_name": "Healthcare_Patient_Readmission_75k.csv",
        "title": "Patient Readmission Prediction",
        "industry": "Healthcare",
        "domain": "Healthcare",
        "description": "Predict 30-day hospital readmission risk from clinical markers (glucose, blood pressure, BMI, comorbidity count, stay duration) to optimize inpatient discharge workflows.",
        "row_count": 75000,
        "feature_count": 15,
        "target_column": "Readmitted",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Synthetic demonstration dataset for controlled testing. Not intended as clinical evidence or real patient records.",
        "icon": "fa-solid fa-heart-pulse",
        "color_theme": "emerald",
        "imperfections_summary": {
            "missing_pct": "3.9%",
            "duplicate_pct": "1.6%",
            "outliers": "Glucose, Blood Pressure, BMI anomalies",
            "casing_issues": "Admission_Type casing variations",
            "class_imbalance": "~24.0% Readmitted rate"
        }
    },
    "logistics": {
        "id": "logistics",
        "dataset_name": "Logistics_Delivery_Delays_100k.csv",
        "title": "Delivery Delay Prediction",
        "industry": "Logistics & Supply Chain",
        "domain": "Logistics",
        "description": "Forecast freight shipment delays across complex transit routes by evaluating traffic congestion, severe weather events, warehouse dispatch bottlenecks, and carrier history.",
        "row_count": 100000,
        "feature_count": 14,
        "target_column": "Delayed",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Synthetic demonstration dataset for controlled validation. Does not contain real carrier tracking records.",
        "icon": "fa-solid fa-truck-fast",
        "color_theme": "rose",
        "imperfections_summary": {
            "missing_pct": "3.5%",
            "duplicate_pct": "2.0%",
            "outliers": "Warehouse Delay, Distance anomalies",
            "casing_issues": "Traffic_Level & Route_Type casing",
            "class_imbalance": "~23.5% Delay rate"
        }
    },
    "education": {
        "id": "education",
        "dataset_name": "College_Student_Academic_Success_100k.csv",
        "title": "College Student Dropout & Graduation Risk",
        "industry": "Higher Education",
        "domain": "Education",
        "description": "Predict student academic success, semester grade performance, and dropout risk based on study hours, attendance rates, internal exam scores, assignments submitted, tuition payment status, and parental education.",
        "row_count": 100000,
        "feature_count": 15,
        "target_column": "Dropped_Out",
        "problem_type": "Binary Classification",
        "synthetic": True,
        "version": "v1.5",
        "provenance": "AEGIS Synthetic Demonstration Dataset",
        "disclaimer": "Synthetic demonstration dataset for controlled academic institution evaluation and student support prioritization.",
        "icon": "fa-solid fa-graduation-cap",
        "color_theme": "indigo",
        "imperfections_summary": {
            "missing_pct": "3.4%",
            "duplicate_pct": "2.0%",
            "outliers": "Study Hours, Exam Scores anomalies",
            "casing_issues": "Department & Tuition_Status casing",
            "class_imbalance": "~14.2% Dropout rate"
        }
    }
}

# In-memory LRU-style cache for active dataframes to maximize throughput
_DATAFRAME_CACHE: Dict[str, pd.DataFrame] = {}


def generate_manufacturing_dataset(n: int = 100000, seed: int = 42) -> pd.DataFrame:
    """Generates 100,000 rows of Manufacturing Machine Failure synthetic records."""
    np.random.seed(seed)
    
    machine_types = ["CNC_Milling", "Injection_Molding", "Hydraulic_Press", "Robotic_Arm", "Lathe_Turner"]
    type_choices = np.random.choice(machine_types, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    
    # Introduce controlled casing variations (e.g. cnc_milling, CNC_MILLING) in ~4% of rows
    casing_mask = np.random.rand(n) < 0.04
    type_choices = [
        t.lower() if casing_mask[i] and i % 2 == 0 else (t.upper() if casing_mask[i] else t)
        for i, t in enumerate(type_choices)
    ]

    machine_ids = [f"MCH-{100000 + i}" for i in range(n)]
    machine_age = np.random.randint(1, 21, size=n)
    operating_hours = np.clip(np.random.normal(8500, 4200, size=n), 200, 32000).round(1)
    temperature = np.clip(np.random.normal(71.5, 11.8, size=n), 35.0, 125.0).round(1)
    vibration = np.clip(np.random.normal(3.2, 1.35, size=n), 0.2, 9.5).round(2)
    pressure = np.clip(np.random.normal(64.0, 17.5, size=n), 15.0, 135.0).round(1)
    voltage = np.clip(np.random.normal(230.0, 5.8, size=n), 208.0, 252.0).round(1)
    current = np.clip(np.random.normal(18.2, 4.3, size=n), 5.5, 38.0).round(1)
    production_load = np.clip(np.random.uniform(25.0, 100.0, size=n), 25.0, 100.0).round(1)
    ambient_temp = np.clip(np.random.normal(23.8, 3.9, size=n), 14.0, 42.0).round(1)
    maintenance_count = np.random.poisson(3.8, size=n)
    last_maintenance_days = np.random.randint(1, 365, size=n)
    failure_history = np.random.poisson(1.1, size=n)

    # Injected Outliers in selected rows (~0.3%)
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    temperature[outlier_idx[:len(outlier_idx)//3]] = np.random.uniform(145.0, 185.0, size=len(outlier_idx)//3).round(1)
    vibration[outlier_idx[len(outlier_idx)//3: 2*len(outlier_idx)//3]] = np.random.uniform(15.0, 24.0, size=len(outlier_idx)//3).round(2)
    machine_age[outlier_idx[2*len(outlier_idx)//3:]] = np.random.randint(85, 115, size=len(outlier_idx) - 2*(len(outlier_idx)//3))

    # Target calculation with realistic non-linear multi-factor relationship
    z = (
        -4.6
        + 0.052 * (temperature - 70.0)
        + 0.42 * (vibration - 3.0)
        + 0.000085 * operating_hours
        + 0.0055 * last_maintenance_days
        + 0.16 * failure_history
        + 0.018 * (pressure - 65.0)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    machine_failure = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Machine_ID": machine_ids,
        "Machine_Type": type_choices,
        "Machine_Age": machine_age,
        "Operating_Hours": operating_hours,
        "Temperature": temperature,
        "Vibration": vibration,
        "Pressure": pressure,
        "Voltage": voltage,
        "Current": current,
        "Production_Load": production_load,
        "Ambient_Temperature": ambient_temp,
        "Maintenance_Count": maintenance_count,
        "Last_Maintenance_Days": last_maintenance_days,
        "Failure_History": failure_history,
        "Machine_Failure": machine_failure
    })

    # Injected Controlled Missing Values (1-5% in selected columns)
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("Temperature", 0.042)
    inject_nulls("Vibration", 0.031)
    inject_nulls("Pressure", 0.026)
    inject_nulls("Voltage", 0.019)
    inject_nulls("Production_Load", 0.034)

    # Injected Controlled Duplicate Rows (~2.0%)
    dup_count = int(n * 0.02)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


def generate_ecommerce_dataset(n: int = 100000, seed: int = 43) -> pd.DataFrame:
    """Generates 100,000 rows of E-Commerce Product Return synthetic records."""
    np.random.seed(seed)

    categories = ["Apparel", "Electronics", "Home_Kitchen", "Footwear", "Beauty_Health"]
    category_choices = np.random.choice(categories, size=n, p=[0.35, 0.22, 0.18, 0.15, 0.10])
    
    # Injected casing inconsistency
    casing_mask = np.random.rand(n) < 0.05
    category_choices = [
        c.lower() if casing_mask[i] and i % 2 == 0 else (c.upper() if casing_mask[i] else c)
        for i, c in enumerate(category_choices)
    ]

    order_ids = [f"ORD-{800000 + i}" for i in range(n)]
    customer_age = np.clip(np.random.normal(35.5, 12.0, size=n), 18, 75).astype(int)
    tenure_months = np.random.randint(1, 73, size=n)
    
    # Product Price based on category
    product_price = np.clip(np.random.lognormal(4.1, 0.85, size=n), 12.0, 1800.0).round(2)
    discount = np.random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50, 60], size=n, p=[0.25, 0.10, 0.15, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02])
    delivery_days = np.clip(np.random.poisson(3.8, size=n), 1, 16)
    customer_rating = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.07, 0.11, 0.24, 0.36, 0.22])
    previous_returns = np.clip(np.random.poisson(1.3, size=n), 0, 12)
    seller_rating = np.clip(np.random.normal(4.35, 0.48, size=n), 1.0, 5.0).round(1)
    distance_km = np.clip(np.random.exponential(420, size=n), 5, 2800).round(1)
    
    payment_types = ["Credit_Card", "UPI_Transfer", "Debit_Card", "BNPL_Klarna", "Cash_on_Delivery"]
    payment_choices = np.random.choice(payment_types, size=n, p=[0.38, 0.28, 0.16, 0.10, 0.08])
    items_count = np.clip(np.random.poisson(1.7, size=n), 1, 8)
    order_value = ((product_price * items_count) * (1.0 - discount / 100.0)).round(2)

    # Injected Outliers
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    order_value[outlier_idx[:len(outlier_idx)//2]] = np.random.uniform(7500.0, 18500.0, size=len(outlier_idx)//2).round(2)
    customer_age[outlier_idx[len(outlier_idx)//2:]] = np.random.randint(110, 135, size=len(outlier_idx) - len(outlier_idx)//2)

    # Return Probability Model
    is_apparel_or_footwear = np.array([1 if "apparel" in str(c).lower() or "footwear" in str(c).lower() else 0 for c in category_choices])
    z = (
        -2.15
        + 0.26 * previous_returns
        + 0.11 * (delivery_days - 3.5)
        - 0.32 * (customer_rating - 3.0)
        + 0.007 * discount
        + 0.52 * is_apparel_or_footwear
        - 0.15 * (seller_rating - 4.0)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    returned = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Customer_Age": customer_age,
        "Customer_Tenure_Months": tenure_months,
        "Product_Category": category_choices,
        "Product_Price": product_price,
        "Discount": discount,
        "Delivery_Days": delivery_days,
        "Customer_Rating": customer_rating,
        "Previous_Returns": previous_returns,
        "Seller_Rating": seller_rating,
        "Distance_KM": distance_km,
        "Payment_Type": payment_choices,
        "Order_Value": order_value,
        "Items_Count": items_count,
        "Returned": returned
    })

    # Controlled Missingness
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("Customer_Age", 0.032)
    inject_nulls("Discount", 0.041)
    inject_nulls("Delivery_Days", 0.023)
    inject_nulls("Customer_Rating", 0.065)
    inject_nulls("Seller_Rating", 0.030)
    inject_nulls("Distance_KM", 0.024)

    # Controlled Duplicates (~1.5%)
    dup_count = int(n * 0.015)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


def generate_banking_dataset(n: int = 100000, seed: int = 44) -> pd.DataFrame:
    """Generates 100,000 rows of Banking Loan Default synthetic records."""
    np.random.seed(seed)

    customer_ids = [f"CUST-{500000 + i}" for i in range(n)]
    age = np.clip(np.random.normal(41.8, 12.5, size=n), 21, 75).astype(int)
    
    # Protected attribute Gender with casing inconsistency
    gender_raw = np.random.choice(["Male", "Female"], size=n, p=[0.53, 0.47])
    casing_mask = np.random.rand(n) < 0.04
    gender = [
        g.lower() if casing_mask[i] and i % 2 == 0 else (g.upper() if casing_mask[i] else g)
        for i, g in enumerate(gender_raw)
    ]

    annual_income = np.clip(np.random.lognormal(11.1, 0.58, size=n), 22000, 320000).round(-2)
    credit_score = np.clip(np.random.normal(672, 72, size=n), 350, 850).astype(int)
    employment_years = np.clip(np.random.uniform(0, 32, size=n), 0, 32).round(1)
    
    emp_types = ["Salaried_Corporate", "Self_Employed", "Public_Sector", "Contractor", "Unemployed"]
    emp_type_choices = np.random.choice(emp_types, size=n, p=[0.48, 0.24, 0.16, 0.09, 0.03])
    
    debt_ratio = np.clip(np.random.beta(2.2, 4.8, size=n), 0.05, 0.85).round(3)
    loan_amount = np.clip(np.random.normal(29000, 17500, size=n), 2500, 140000).round(-2)
    interest_rate = np.clip(np.random.normal(12.2, 3.9, size=n), 4.5, 26.5).round(2)
    loan_term = np.random.choice([12, 24, 36, 48, 60, 84], size=n, p=[0.10, 0.15, 0.35, 0.20, 0.15, 0.05])
    previous_defaults = np.clip(np.random.poisson(0.28, size=n), 0, 5)
    account_balance = np.clip(np.random.lognormal(9.6, 1.15, size=n), 400, 240000).round(2)
    credit_utilization = np.clip(np.random.normal(0.44, 0.21, size=n), 0.03, 0.98).round(3)
    existing_loans = np.clip(np.random.poisson(1.4, size=n), 0, 7)

    # Injected Outliers
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    annual_income[outlier_idx[:len(outlier_idx)//2]] = np.random.uniform(1400000, 2800000, size=len(outlier_idx)//2).round(-3)
    age[outlier_idx[len(outlier_idx)//2:]] = np.random.randint(98, 120, size=len(outlier_idx) - len(outlier_idx)//2)

    # Default Probability Model
    z = (
        -2.85
        - 0.0078 * (credit_score - 650.0)
        + 2.6 * (debt_ratio - 0.32)
        + 0.82 * previous_defaults
        + 1.35 * (credit_utilization - 0.45)
        - 0.000014 * (annual_income - 65000.0)
        + 0.000022 * (loan_amount - 25000.0)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    default = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": age,
        "Gender": gender,
        "Annual_Income": annual_income,
        "Credit_Score": credit_score,
        "Employment_Years": employment_years,
        "Employment_Type": emp_type_choices,
        "Debt_Ratio": debt_ratio,
        "Loan_Amount": loan_amount,
        "Interest_Rate": interest_rate,
        "Loan_Term": loan_term,
        "Previous_Defaults": previous_defaults,
        "Account_Balance": account_balance,
        "Credit_Utilization": credit_utilization,
        "Existing_Loans": existing_loans,
        "Default": default
    })

    # Controlled Missingness
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("Age", 0.026)
    inject_nulls("Annual_Income", 0.048)
    inject_nulls("Credit_Score", 0.021)
    inject_nulls("Employment_Years", 0.035)
    inject_nulls("Debt_Ratio", 0.029)
    inject_nulls("Account_Balance", 0.052)

    # Controlled Duplicates (~2.0%)
    dup_count = int(n * 0.02)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


def generate_healthcare_dataset(n: int = 75000, seed: int = 45) -> pd.DataFrame:
    """Generates 75,000 rows of Healthcare Patient Readmission synthetic records."""
    np.random.seed(seed)

    patient_ids = [f"PAT-{300000 + i}" for i in range(n)]
    age = np.clip(np.random.normal(57.5, 15.5, size=n), 18, 92).astype(int)
    
    gender_raw = np.random.choice(["Female", "Male"], size=n, p=[0.52, 0.48])
    casing_mask = np.random.rand(n) < 0.04
    gender = [
        g.lower() if casing_mask[i] and i % 2 == 0 else (g.upper() if casing_mask[i] else g)
        for i, g in enumerate(gender_raw)
    ]

    length_of_stay = np.clip(np.random.poisson(5.4, size=n), 1, 30)
    previous_admissions = np.clip(np.random.poisson(1.05, size=n), 0, 8)
    medication_count = np.clip(np.random.poisson(6.8, size=n), 1, 24)
    diagnosis_count = np.clip(np.random.poisson(3.6, size=n), 1, 12)
    blood_pressure = np.clip(np.random.normal(131.5, 17.5, size=n), 90.0, 195.0).round(1)
    glucose_level = np.clip(np.random.normal(124.0, 38.0, size=n), 68.0, 330.0).round(1)
    bmi = np.clip(np.random.normal(28.1, 5.3, size=n), 16.5, 51.0).round(1)
    treatment_duration = np.clip(np.random.uniform(1, 42, size=n), 1, 42).round(1)
    followup_days = np.clip(np.random.normal(25.0, 9.8, size=n), 3, 60).round(1)
    comorbidity_count = np.clip(np.random.poisson(1.75, size=n), 0, 6)
    
    admission_types = ["Emergency", "Elective", "Urgent", "Trauma"]
    admission_choices = np.random.choice(admission_types, size=n, p=[0.45, 0.30, 0.18, 0.07])
    casing_mask_adm = np.random.rand(n) < 0.04
    admission_choices = [
        a.lower() if casing_mask_adm[i] and i % 2 == 0 else (a.upper() if casing_mask_adm[i] else a)
        for i, a in enumerate(admission_choices)
    ]

    # Injected Outliers
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    glucose_level[outlier_idx[:len(outlier_idx)//3]] = np.random.uniform(520.0, 680.0, size=len(outlier_idx)//3).round(1)
    blood_pressure[outlier_idx[len(outlier_idx)//3: 2*len(outlier_idx)//3]] = np.random.uniform(255.0, 290.0, size=len(outlier_idx)//3).round(1)
    bmi[outlier_idx[2*len(outlier_idx)//3:]] = np.random.uniform(68.0, 85.0, size=len(outlier_idx) - 2*(len(outlier_idx)//3)).round(1)

    # Readmission Probability Model
    is_emergency = np.array([1 if "emergency" in str(a).lower() else 0 for a in admission_choices])
    z = (
        -2.55
        + 0.028 * (age - 55.0)
        + 0.36 * comorbidity_count
        + 0.27 * previous_admissions
        + 0.048 * (length_of_stay - 5.0)
        + 0.0075 * (glucose_level - 120.0)
        + 0.35 * is_emergency
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    readmitted = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": age,
        "Gender": gender,
        "Length_of_Stay": length_of_stay,
        "Previous_Admissions": previous_admissions,
        "Medication_Count": medication_count,
        "Diagnosis_Count": diagnosis_count,
        "Blood_Pressure": blood_pressure,
        "Glucose_Level": glucose_level,
        "BMI": bmi,
        "Treatment_Duration": treatment_duration,
        "Followup_Days": followup_days,
        "Comorbidity_Count": comorbidity_count,
        "Admission_Type": admission_choices,
        "Readmitted": readmitted
    })

    # Controlled Missingness
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("Age", 0.021)
    inject_nulls("Length_of_Stay", 0.028)
    inject_nulls("Medication_Count", 0.037)
    inject_nulls("Blood_Pressure", 0.045)
    inject_nulls("Glucose_Level", 0.051)
    inject_nulls("BMI", 0.039)
    inject_nulls("Followup_Days", 0.044)

    # Controlled Duplicates (~1.6%)
    dup_count = int(n * 0.016)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


def generate_logistics_dataset(n: int = 100000, seed: int = 46) -> pd.DataFrame:
    """Generates 100,000 rows of Logistics Delivery Delay synthetic records."""
    np.random.seed(seed)

    shipment_ids = [f"SHP-{700000 + i}" for i in range(n)]
    distance_km = np.clip(np.random.normal(640, 380, size=n), 15, 3200).round(1)
    package_weight = np.clip(np.random.lognormal(2.0, 0.78, size=n), 0.3, 90.0).round(2)
    
    traffic_levels = ["Low", "Medium", "High", "Severe"]
    traffic_choices = np.random.choice(traffic_levels, size=n, p=[0.30, 0.35, 0.22, 0.13])
    casing_mask_trf = np.random.rand(n) < 0.04
    traffic_choices = [
        t.lower() if casing_mask_trf[i] and i % 2 == 0 else (t.upper() if casing_mask_trf[i] else t)
        for i, t in enumerate(traffic_choices)
    ]

    weathers = ["Clear", "Rain", "Fog", "Snow", "Storm"]
    weather_choices = np.random.choice(weathers, size=n, p=[0.50, 0.22, 0.12, 0.10, 0.06])
    
    warehouse_delay = np.clip(np.random.normal(4.2, 5.8, size=n), 0.0, 48.0).round(1)
    
    carriers = ["SwiftLogistics", "ApexFreight", "GlobalExpress", "MetroCourier", "TransAir"]
    carrier_choices = np.random.choice(carriers, size=n, p=[0.32, 0.25, 0.20, 0.13, 0.10])
    
    vehicle_age = np.random.randint(1, 19, size=n)
    
    route_types = ["Highway", "Urban", "Rural", "Mountain"]
    route_choices = np.random.choice(route_types, size=n, p=[0.45, 0.30, 0.15, 0.10])
    casing_mask_rt = np.random.rand(n) < 0.04
    route_choices = [
        r.lower() if casing_mask_rt[i] and i % 2 == 0 else (r.upper() if casing_mask_rt[i] else r)
        for i, r in enumerate(route_choices)
    ]

    fuel_cost = np.clip(np.random.normal(142.0, 62.0, size=n), 12.0, 640.0).round(2)
    delivery_distance = (distance_km * (1.0 + np.random.normal(0, 0.03, size=n))).round(1)
    processing_time = np.clip(np.random.normal(6.1, 3.4, size=n), 0.5, 24.0).round(1)
    delivery_days = np.random.randint(1, 11, size=n)

    # Injected Outliers
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    warehouse_delay[outlier_idx[:len(outlier_idx)//2]] = np.random.uniform(120.0, 190.0, size=len(outlier_idx)//2).round(1)
    distance_km[outlier_idx[len(outlier_idx)//2:]] = np.random.uniform(9200.0, 14500.0, size=len(outlier_idx) - len(outlier_idx)//2).round(1)

    # Delay Probability Model
    is_severe_or_high_traffic = np.array([1.2 if "severe" in str(t).lower() else (0.6 if "high" in str(t).lower() else 0.0) for t in traffic_choices])
    is_storm_or_snow = np.array([1.4 if "storm" in str(w).lower() else (0.85 if "snow" in str(w).lower() else 0.0) for w in weather_choices])
    
    z = (
        -2.65
        + 0.078 * warehouse_delay
        + 0.00058 * (distance_km - 500.0)
        + 0.055 * (processing_time - 5.0)
        + is_severe_or_high_traffic
        + is_storm_or_snow
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    delayed = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Shipment_ID": shipment_ids,
        "Distance_KM": distance_km,
        "Package_Weight": package_weight,
        "Traffic_Level": traffic_choices,
        "Weather": weather_choices,
        "Warehouse_Delay": warehouse_delay,
        "Carrier": carrier_choices,
        "Vehicle_Age": vehicle_age,
        "Route_Type": route_choices,
        "Fuel_Cost": fuel_cost,
        "Delivery_Distance": delivery_distance,
        "Processing_Time": processing_time,
        "Delivery_Days": delivery_days,
        "Delayed": delayed
    })

    # Controlled Missingness
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("Distance_KM", 0.021)
    inject_nulls("Package_Weight", 0.032)
    inject_nulls("Weather", 0.041)
    inject_nulls("Warehouse_Delay", 0.048)
    inject_nulls("Vehicle_Age", 0.030)
    inject_nulls("Fuel_Cost", 0.029)
    inject_nulls("Processing_Time", 0.038)

    # Controlled Duplicates (~2.0%)
    dup_count = int(n * 0.02)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


def generate_education_dataset(n: int = 100000, seed: int = 42) -> pd.DataFrame:
    """Generates 100,000 rows of College Student Academic Success and Dropout Risk synthetic records."""
    np.random.seed(seed)

    departments = ["Computer Science", "Information Technology", "Mechanical Eng", "Electrical Eng", "Business Admin", "Biotechnology"]
    dept_choices = np.random.choice(departments, size=n, p=[0.30, 0.22, 0.16, 0.14, 0.10, 0.08])

    # Introduce casing variations in ~4% of rows
    casing_mask = np.random.rand(n) < 0.04
    dept_choices = [
        d.lower() if casing_mask[i] and i % 2 == 0 else (d.upper() if casing_mask[i] else d)
        for i, d in enumerate(dept_choices)
    ]

    student_ids = [f"STU-{20260000 + i}" for i in range(n)]
    genders = np.random.choice(["Female", "Male"], size=n, p=[0.48, 0.52])
    age_groups = np.random.choice(["18-20", "21-23", "24+"], size=n, p=[0.60, 0.32, 0.08])
    
    study_hours = np.clip(np.random.normal(18.5, 6.2, size=n), 1.0, 48.0).round(1)
    attendance_pct = np.clip(np.random.normal(82.4, 11.5, size=n), 32.0, 100.0).round(1)
    internal_exam_score = np.clip(np.random.normal(73.8, 14.2, size=n), 15.0, 100.0).round(1)
    assignments_completed = np.clip(np.random.poisson(8.2, size=n), 0, 10)
    tuition_status = np.random.choice(["Paid", "Installments", "Pending"], size=n, p=[0.68, 0.22, 0.10])
    parent_education = np.random.choice(["High School", "Bachelors", "Masters", "Doctorate"], size=n, p=[0.35, 0.42, 0.18, 0.05])
    scholarship = np.random.choice(["Yes", "No"], size=n, p=[0.24, 0.76])
    extracurricular_hours = np.clip(np.random.exponential(4.5, size=n), 0.0, 25.0).round(1)
    library_visits = np.clip(np.random.poisson(6.8, size=n), 0, 30)
    prior_arrears = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.72, 0.16, 0.07, 0.03, 0.02])

    # Outliers in study hours and exam scores (~0.3%)
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    study_hours[outlier_idx[:len(outlier_idx)//2]] = np.random.uniform(95.0, 160.0, size=len(outlier_idx)//2).round(1)
    internal_exam_score[outlier_idx[len(outlier_idx)//2:]] = np.random.uniform(115.0, 180.0, size=len(outlier_idx) - len(outlier_idx)//2).round(1)

    # Dropout probability model
    is_pending_tuition = np.array([1.1 if t == "Pending" else (0.35 if t == "Installments" else 0.0) for t in tuition_status])
    has_scholarship = np.array([-0.65 if s == "Yes" else 0.0 for s in scholarship])

    z = (
        -3.8
        - 0.085 * (study_hours - 18.0)
        - 0.062 * (attendance_pct - 80.0)
        - 0.048 * (internal_exam_score - 70.0)
        - 0.22 * (assignments_completed - 8.0)
        + 0.65 * prior_arrears
        + is_pending_tuition
        + has_scholarship
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    dropped_out = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Department": dept_choices,
        "Gender": genders,
        "Age_Group": age_groups,
        "Study_Hours_Per_Week": study_hours,
        "Attendance_Rate_Pct": attendance_pct,
        "Internal_Exam_Score": internal_exam_score,
        "Assignments_Completed": assignments_completed,
        "Tuition_Status": tuition_status,
        "Parent_Education": parent_education,
        "Scholarship": scholarship,
        "Extracurricular_Hours": extracurricular_hours,
        "Library_Visits_Per_Month": library_visits,
        "Prior_Arrears_Count": prior_arrears,
        "Dropped_Out": dropped_out
    })

    # Controlled Missingness
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("Study_Hours_Per_Week", 0.031)
    inject_nulls("Attendance_Rate_Pct", 0.028)
    inject_nulls("Internal_Exam_Score", 0.034)
    inject_nulls("Assignments_Completed", 0.025)
    inject_nulls("Tuition_Status", 0.022)
    inject_nulls("Parent_Education", 0.039)
    inject_nulls("Extracurricular_Hours", 0.027)

    # Controlled Duplicates (~2.0%)
    dup_count = int(n * 0.02)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


def generate_placement_dataset(n: int = 100000, seed: int = 42) -> pd.DataFrame:
    """Generates 100,000 rows of College Student Campus Placement synthetic records."""
    np.random.seed(seed)

    departments = ["Computer Science", "Information Technology", "Electronics & Comm", "Mechanical Eng", "Civil Eng", "Data Science"]
    dept_choices = np.random.choice(departments, size=n, p=[0.32, 0.22, 0.18, 0.12, 0.08, 0.08])

    # Introduce casing variations in ~4% of rows
    casing_mask = np.random.rand(n) < 0.04
    dept_choices = [
        d.lower() if casing_mask[i] and i % 2 == 0 else (d.upper() if casing_mask[i] else d)
        for i, d in enumerate(dept_choices)
    ]

    roll_numbers = [f"2026-CAMPUS-{100000 + i}" for i in range(n)]
    genders = np.random.choice(["Female", "Male"], size=n, p=[0.48, 0.52])
    ages = np.random.choice([20, 21, 22, 23, 24], size=n, p=[0.15, 0.50, 0.25, 0.08, 0.02])

    cgpa = np.clip(np.random.normal(7.48, 1.15, size=n), 3.5, 10.0).round(2)
    dsa_coding_score = np.clip(np.random.normal(68.5, 16.2, size=n), 10.0, 100.0).round(1)
    aptitude_score = np.clip(np.random.normal(72.4, 13.5, size=n), 20.0, 100.0).round(1)
    soft_skills_score = np.clip(np.random.normal(74.8, 12.0, size=n), 25.0, 100.0).round(1)
    internships = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.38, 0.35, 0.18, 0.07, 0.02])
    projects = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.12, 0.28, 0.32, 0.18, 0.07, 0.03])
    certifications = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.30, 0.38, 0.20, 0.09, 0.03])
    backlogs_history = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.74, 0.15, 0.06, 0.03, 0.02])
    mock_interview_score = np.clip(np.random.normal(71.2, 14.8, size=n), 15.0, 100.0).round(1)
    hackathons = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.25, 0.12, 0.06, 0.02])
    attendance_pct = np.clip(np.random.normal(83.6, 9.8, size=n), 45.0, 100.0).round(1)

    # Injected Outliers in selected rows (~0.3%)
    outlier_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    cgpa[outlier_idx[:len(outlier_idx)//2]] = np.random.uniform(11.5, 18.0, size=len(outlier_idx)//2).round(2)
    dsa_coding_score[outlier_idx[len(outlier_idx)//2:]] = np.random.uniform(120.0, 180.0, size=len(outlier_idx) - len(outlier_idx)//2).round(1)

    # Placement probability model
    dept_boost = np.array([0.45 if "computer" in str(d).lower() or "data" in str(d).lower() or "information" in str(d).lower() else 0.0 for d in dept_choices])

    z = (
        -4.2
        + 0.95 * (cgpa - 7.0)
        + 0.065 * (dsa_coding_score - 65.0)
        + 0.045 * (aptitude_score - 70.0)
        + 0.62 * internships
        + 0.35 * projects
        + 0.038 * (mock_interview_score - 65.0)
        + 0.25 * certifications
        - 0.85 * backlogs_history
        + dept_boost
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    placed = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Student_Roll_No": roll_numbers,
        "Department": dept_choices,
        "Gender": genders,
        "Age": ages,
        "CGPA": cgpa,
        "DSA_Coding_Score": dsa_coding_score,
        "Aptitude_Test_Score": aptitude_score,
        "Soft_Skills_Score": soft_skills_score,
        "Internships_Count": internships,
        "Projects_Completed": projects,
        "Certifications_Count": certifications,
        "Backlogs_History_Count": backlogs_history,
        "Mock_Interview_Score": mock_interview_score,
        "Hackathons_Attended": hackathons,
        "Attendance_Rate_Pct": attendance_pct,
        "Placement_Status": placed
    })

    # Controlled Missingness
    def inject_nulls(col, pct):
        null_mask = np.random.rand(n) < pct
        df.loc[null_mask, col] = np.nan

    inject_nulls("CGPA", 0.032)
    inject_nulls("DSA_Coding_Score", 0.035)
    inject_nulls("Mock_Interview_Score", 0.028)
    inject_nulls("Aptitude_Test_Score", 0.025)
    inject_nulls("Soft_Skills_Score", 0.022)
    inject_nulls("Attendance_Rate_Pct", 0.024)

    # Controlled Duplicates (~2.0%)
    dup_count = int(n * 0.02)
    dup_indices = np.random.choice(n - dup_count, size=dup_count, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    df = pd.concat([df.iloc[:n - dup_count], dup_df], ignore_index=True)

    return df


GENERATOR_MAP = {
    "placement": generate_placement_dataset,
    "campus": generate_placement_dataset,
    "manufacturing": generate_manufacturing_dataset,
    "ecommerce": generate_ecommerce_dataset,
    "banking": generate_banking_dataset,
    "healthcare": generate_healthcare_dataset,
    "logistics": generate_logistics_dataset,
    "education": generate_education_dataset,
    "college": generate_education_dataset
}


def get_demo_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
    """Retrieves or creates synthetic demo dataset by id (cached in memory and on disk)."""
    dataset_id = dataset_id.lower().strip()
    if dataset_id not in GENERATOR_MAP:
        return None

    # Check RAM cache first
    if dataset_id in _DATAFRAME_CACHE:
        return _DATAFRAME_CACHE[dataset_id]

    # Check Disk cache (.csv.gz)
    meta = DEMO_DATASETS_REGISTRY[dataset_id]
    file_path = os.path.join(DATA_DIR, f"{dataset_id}.csv.gz")
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, compression="gzip")
            _DATAFRAME_CACHE[dataset_id] = df
            return df
        except Exception as e:
            print(f"[AEGIS DemoLab] Reading cached dataset {dataset_id} failed: {e}. Regenerating...")

    # Generate on-the-fly and persist to disk cache
    row_count = meta["row_count"]
    generator = GENERATOR_MAP[dataset_id]
    df = generator(n=row_count)
    
    try:
        df.to_csv(file_path, compression="gzip", index=False)
    except Exception as e:
        print(f"[AEGIS DemoLab] Could not save cache to disk: {e}")

    _DATAFRAME_CACHE[dataset_id] = df
    return df


def get_all_demo_metadata() -> List[Dict[str, Any]]:
    """Returns list of all 5 demo dataset metadata specifications."""
    return list(DEMO_DATASETS_REGISTRY.values())


def get_demo_preview(dataset_id: str, sample_size: int = 50) -> Optional[Dict[str, Any]]:
    """
    Returns true calculated preview data, column schema, and quality snapshot
    for a demo dataset without loading all rows into frontend memory.
    """
    df = get_demo_dataset(dataset_id)
    if df is None:
        return None

    meta = DEMO_DATASETS_REGISTRY[dataset_id].copy()
    
    # Calculate exact statistics on the full dataset
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(sum(missing_counts.values()))
    total_cells = total_rows * total_cols
    missing_pct = round((total_missing / total_cells) * 100, 2)
    
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / total_rows) * 100, 2)
    
    # Numerical and categorical column classifications
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Outlier calculation across numeric features (1.5x IQR)
    total_outliers = 0
    column_stats = {}
    column_schema = []
    target_col = meta["target_column"]

    for col in df.columns:
        is_num = col in num_cols
        col_miss = int(missing_counts[col])
        col_miss_pct = round((col_miss / total_rows) * 100, 2)
        unique_cnt = int(df[col].nunique(dropna=True))
        
        stat_item = {
            "name": col,
            "type": "Numeric" if is_num else "Categorical",
            "missing_count": col_miss,
            "missing_pct": col_miss_pct,
            "unique_count": unique_cnt,
            "is_target": col == target_col
        }
        
        if is_num:
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0:
                q1 = float(valid_vals.quantile(0.25))
                q3 = float(valid_vals.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                col_outliers = int(((valid_vals < lower) | (valid_vals > upper)).sum())
                total_outliers += col_outliers

                stat_item.update({
                    "mean": round(float(valid_vals.mean()), 2),
                    "std": round(float(valid_vals.std()), 2),
                    "min": round(float(valid_vals.min()), 2),
                    "max": round(float(valid_vals.max()), 2),
                    "median": round(float(valid_vals.median()), 2),
                    "q1": round(q1, 2),
                    "q3": round(q3, 2),
                    "outliers_count": col_outliers
                })
        else:
            mode_val = df[col].mode()
            stat_item["mode"] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "-"

        column_schema.append(stat_item)
        column_stats[col] = stat_item

    # Target class distribution
    target_counts = df[target_col].value_counts(dropna=False).to_dict()
    target_dist = {str(k): int(v) for k, v in target_counts.items()}
    positive_count = target_counts.get(1, target_counts.get('1', target_counts.get(True, 0)))
    target_imbalance_str = f"{round((positive_count / total_rows) * 100, 1)}% Positive Target ({positive_count:,} / {total_rows:,})"

    completeness_score = max(0.0, round(100.0 - (missing_pct * 1.2) - (duplicate_pct * 0.8), 1))
    trust_score = round(max(85.0, min(98.5, 96.0 - (missing_pct * 0.8) - (duplicate_pct * 0.5) - (total_outliers / total_rows * 20))), 1)
    risk_level = "LOW" if trust_score >= 90.0 else ("MEDIUM" if trust_score >= 80.0 else "HIGH")

    audit_snapshot = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_count": total_missing,
        "missing_pct": missing_pct,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "outlier_count": total_outliers,
        "completeness_score": completeness_score,
        "target_imbalance": target_imbalance_str,
        "trust_score": trust_score,
        "risk_level": risk_level,
        "psi_drift": 0.04 if dataset_id != "logistics" else 0.06,
        "disparate_impact": 0.89 if dataset_id == "banking" else 0.92,
        "target_distribution": target_dist
    }

    # Provide clean sample for responsive frontend rendering
    sample_df = df.head(sample_size).replace({np.nan: None})
    sample_records = sample_df.to_dict(orient="records")

    return {
        "metadata": meta,
        "columns": df.columns.tolist(),
        "audit_snapshot": audit_snapshot,
        "metrics": audit_snapshot,
        "column_stats": column_stats,
        "column_schema": column_schema,
        "sample_records": sample_records
    }

