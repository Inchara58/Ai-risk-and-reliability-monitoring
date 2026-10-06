"""
AEGIS AI — STAGE 6 INTEGRATION TEST SUITE
Tests:
1. Copilot Question Answering & Strict Project-Evidence Grounding
2. Assessment Report Generation (16 Sections)
3. Model Card Generation (Standardized ML Governance Specs)
4. Project Timeline & Lifecycle Provenance
5. Global Command Center Search API
6. Multi-Domain End-to-End Adaptability across all 5 Demo Datasets:
   - Manufacturing (Machine_Failure)
   - E-Commerce (Returned)
   - Banking (Default)
   - Healthcare (Readmitted)
   - Logistics (Delayed)
"""

import json
import unittest
import requests

BASE_URL = "http://127.0.0.1:5000"

class TestStage6Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a test project for Stage 6 validation
        payload = {
            "name": "Stage 6 Enterprise Governance Suite",
            "description": "Validation project for Copilot, 16-section Assessment Report, Model Card, and Command Center",
            "domain": "Manufacturing",
            "dataset_name": "Manufacturing_Machine_Failure_100k.csv"
        }
        res = requests.post(f"{BASE_URL}/api/v1/projects", json=payload)
        assert res.status_code in [200, 201], f"Failed to create project: {res.text}"
        data = res.json()
        cls.project_id = data["project"]["id"]

        # Run model training to establish model artifacts
        train_res = requests.post(f"{BASE_URL}/api/v1/models/train", json={
            "project_id": cls.project_id,
            "algorithm": "rf",
            "target_column": "Machine_Failure"
        })
        assert train_res.status_code in [200, 201], f"Model train failed: {train_res.text}"

        # Run stress test to establish stress lab artifacts
        stress_res = requests.post(f"{BASE_URL}/api/v1/stress/run-experiment", json={
            "project_id": cls.project_id,
            "stress_type": "gaussian_noise",
            "severity_pct": 20
        })
        assert stress_res.status_code == 200, f"Stress test failed: {stress_res.text}"

        # Run deployment signoff
        signoff_res = requests.post(f"{BASE_URL}/api/v1/deployment/review", json={
            "project_id": cls.project_id,
            "decision": "APPROVE",
            "reason": "Passed all Stage 6 reliability, fairness, and stress-testing gates.",
            "reviewer_name": "Dr. Elena Rostova"
        })
        assert signoff_res.status_code == 200, f"Deployment signoff failed: {signoff_res.text}"

        # Run monitoring cycle
        mon_res = requests.post(f"{BASE_URL}/api/v1/monitoring/run", json={
            "project_id": cls.project_id,
            "current_dataset_name": "Live_Stream_Sample_v2.0.csv",
            "include_ground_truth": True
        })
        assert mon_res.status_code == 200, f"Monitoring run failed: {mon_res.text}"

    # -------------------------------------------------------------------------
    # 1. AEGIS COPILOT TESTS
    # -------------------------------------------------------------------------
    def test_copilot_data_quality_query(self):
        """Test Copilot answering data quality queries with evidence grounding"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "What are the biggest data-quality issues?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["evidence_grounded"])
        self.assertIn("data quality", data["reply"].lower())
        # Verify navigation actions are returned
        action_names = [a["action"] for a in data.get("actions", [])]
        self.assertTrue("view-data-studio" in action_names or "view-auditor" in action_names)

    def test_copilot_missing_features_query(self):
        """Test Copilot identifying missing value features"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "Which features have the most missing values?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(data["reply"]) > 20)

    def test_copilot_model_performance_query(self):
        """Test Copilot answering model performance with holdout accuracy"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "How did the model perform?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("accuracy", data["reply"].lower())

    def test_copilot_fairness_query(self):
        """Test Copilot answering fairness & EEOC 80% rule query"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "What fairness issues were detected?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("disparate impact", data["reply"].lower())

    def test_copilot_drift_query(self):
        """Test Copilot answering drift query with PSI evidence"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "What drift was detected?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("psi", data["reply"].lower())

    def test_copilot_stress_test_query(self):
        """Test Copilot answering stress test query"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "What happened during the stress test?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("resilience", data["reply"].lower())

    def test_copilot_deployment_investigation_query(self):
        """Test Copilot advising on deployment checklist before signoff"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "What should I investigate before deployment?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(data["reply"]) > 50)

    def test_copilot_model_version_query(self):
        """Test Copilot returning current model version"""
        res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
            "project_id": self.project_id,
            "question": "What model version is currently selected?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("version", data["reply"].lower())

    # -------------------------------------------------------------------------
    # 2. ASSESSMENT REPORT GENERATION (16 SECTIONS)
    # -------------------------------------------------------------------------
    def test_assessment_report_structure(self):
        """Test that the 16-section AEGIS AI Assessment Report has all required sections"""
        res = requests.get(f"{BASE_URL}/api/v1/reports/assessment/{self.project_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        rep = data["report"]

        # Check all 16 sections
        self.assertIn("executive_summary", rep)
        self.assertIn("project_information", rep)
        self.assertIn("dataset_information", rep)
        self.assertIn("data_quality", rep)
        self.assertIn("cleaning_actions", rep)
        self.assertIn("feature_analysis", rep)
        self.assertIn("model_configuration", rep)
        self.assertIn("model_performance", rep)
        self.assertIn("fairness", rep)
        self.assertIn("drift", rep)
        self.assertIn("robustness", rep)
        self.assertIn("risk_assessment", rep)
        self.assertIn("business_impact", rep)
        self.assertIn("deployment_review", rep)
        self.assertIn("monitoring_status", rep)
        self.assertIn("recommendations", rep)
        self.assertIn("audit_timeline", rep)

        # Check evidence values
        self.assertEqual(rep["project_information"]["domain"], "Manufacturing")
        self.assertIsNotNone(rep["executive_summary"]["trust_score"])
        self.assertIsNotNone(rep["model_performance"]["accuracy"])

    # -------------------------------------------------------------------------
    # 3. MODEL CARD GENERATION
    # -------------------------------------------------------------------------
    def test_model_card_structure(self):
        """Test that the Model Card contains all required governance specifications"""
        res = requests.get(f"{BASE_URL}/api/v1/reports/model-card/{self.project_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        mc = data["model_card"]

        self.assertIn("model_name", mc)
        self.assertIn("version", mc)
        self.assertIn("dataset", mc)
        self.assertIn("target", mc)
        self.assertIn("problem_type", mc)
        self.assertIn("features", mc)
        self.assertIn("training_date", mc)
        self.assertIn("evaluation_metrics", mc)
        self.assertIn("known_limitations", mc)
        self.assertIn("fairness_information", mc)
        self.assertIn("robustness_information", mc)
        self.assertIn("intended_use", mc)
        self.assertIn("out_of_scope_use", mc)

        self.assertEqual(mc["target"], "Machine_Failure")
        self.assertTrue(len(mc["features"]) > 0)
        self.assertTrue(len(mc["known_limitations"]) > 0)

    # -------------------------------------------------------------------------
    # 4. PROJECT TIMELINE
    # -------------------------------------------------------------------------
    def test_project_timeline(self):
        """Test that the Project Timeline returns chronological verifiable events"""
        res = requests.get(f"{BASE_URL}/api/v1/projects/{self.project_id}/timeline")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        timeline = data["timeline"]
        self.assertTrue(len(timeline) >= 4)
        
        event_types = [e["event_type"] for e in timeline]
        self.assertIn("PROJECT_CREATED", event_types)
        self.assertIn("MODEL_TRAINED", event_types)
        self.assertIn("DEPLOYMENT_SIGN_OFF", event_types)

    # -------------------------------------------------------------------------
    # 5. GLOBAL COMMAND CENTER SEARCH
    # -------------------------------------------------------------------------
    def test_command_center_search(self):
        """Test Command Center quick search endpoint"""
        res = requests.get(f"{BASE_URL}/api/v1/command-center/search?q=drift")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(data["results"]) > 0)
        actions = [r["action"] for r in data["results"]]
        self.assertTrue("view-auditor" in actions or "view-monitoring-center" in actions)

    # -------------------------------------------------------------------------
    # 6. MULTI-DOMAIN END-TO-END DEMONSTRATION ACROSS 5 SYNTHETIC DATASETS
    # -------------------------------------------------------------------------
    def test_multi_domain_e2e_adaptation(self):
        """Demonstrate that AEGIS dynamically adapts across 5 distinct industry domains"""
        domains = [
            {"domain": "Manufacturing", "dataset": "Manufacturing_Machine_Failure_100k.csv", "target": "Machine_Failure"},
            {"domain": "E-Commerce", "dataset": "Ecommerce_Product_Returns_100k.csv", "target": "Returned"},
            {"domain": "Banking", "dataset": "Banking_Loan_Defaults_100k.csv", "target": "Default"},
            {"domain": "Healthcare", "dataset": "Healthcare_Patient_Readmission_75k.csv", "target": "Readmitted"},
            {"domain": "Logistics", "dataset": "Logistics_Delivery_Delays_100k.csv", "target": "Delayed"}
        ]

        for item in domains:
            # 1. Create project
            p_res = requests.post(f"{BASE_URL}/api/v1/projects", json={
                "name": f"E2E {item['domain']} Validation",
                "domain": item["domain"],
                "dataset_name": item["dataset"]
            })
            self.assertEqual(p_res.status_code, 201)
            p_id = p_res.json()["project"]["id"]

            # 2. Train model with industry-specific target
            t_res = requests.post(f"{BASE_URL}/api/v1/models/train", json={
                "project_id": p_id,
                "algorithm": "gbm",
                "target_column": item["target"]
            })
            self.assertEqual(t_res.status_code, 200)
            self.assertEqual(t_res.json()["status"], "success")

            # 3. Ask copilot domain-specific question
            c_res = requests.post(f"{BASE_URL}/api/v1/copilot/chat", json={
                "project_id": p_id,
                "question": "What is the target column and how did the model perform?"
            })
            self.assertEqual(c_res.status_code, 200)
            self.assertIn(item["target"].lower(), c_res.json()["reply"].lower())

            # 4. Generate Model Card
            mc_res = requests.get(f"{BASE_URL}/api/v1/reports/model-card/{p_id}")
            self.assertEqual(mc_res.status_code, 200)
            self.assertEqual(mc_res.json()["model_card"]["target"], item["target"])


if __name__ == "__main__":
    unittest.main()
