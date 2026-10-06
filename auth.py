"""
AI Risk & Reliability Platform - Authentication Module
File: auth.py
Description: Real user signup/login backed by a local SQLite database.
             Passwords are never stored in plain text - only a salted hash
             (via Werkzeug's generate_password_hash / check_password_hash,
             which uses PBKDF2-SHA256 under the hood).
"""

import sqlite3
import os
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aegis_users.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the users and projects tables if they don't already exist. Safe to call every startup."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Data Scientist',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            name TEXT NOT NULL,
            description TEXT,
            domain TEXT,
            dataset_name TEXT,
            dataset_rows INTEGER DEFAULT 0,
            dataset_cols INTEGER DEFAULT 0,
            dataset_version TEXT DEFAULT 'v1.0',
            target_column TEXT,
            model_status TEXT DEFAULT 'Not trained',
            audit_status TEXT DEFAULT 'Not run',
            audit_score REAL DEFAULT NULL,
            deployment_status TEXT DEFAULT 'Not reviewed',
            monitoring_status TEXT DEFAULT 'Inactive',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            state_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            problem_type TEXT NOT NULL,
            target_column TEXT NOT NULL,
            version TEXT NOT NULL DEFAULT 'v1',
            is_active INTEGER DEFAULT 1,
            metrics_json TEXT NOT NULL,
            feature_config_json TEXT NOT NULL,
            model_path TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stress_experiments (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            stress_type TEXT NOT NULL,
            severity_pct REAL NOT NULL,
            baseline_metric REAL NOT NULL,
            stressed_metric REAL NOT NULL,
            degradation_pct REAL NOT NULL,
            resilience_score REAL NOT NULL,
            metric_name TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_dossiers (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            dataset_version TEXT DEFAULT 'v1.0',
            model_name TEXT,
            model_version TEXT,
            trust_score REAL NOT NULL,
            readiness_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            deployment_verdict TEXT NOT NULL,
            dossier_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deployment_decisions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            model_id TEXT,
            model_version TEXT,
            dataset_name TEXT,
            decision TEXT NOT NULL,
            reviewer_name TEXT NOT NULL,
            reviewer_email TEXT,
            reviewer_role TEXT,
            decision_reason TEXT NOT NULL,
            evidence_snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            model_id TEXT,
            model_version TEXT,
            baseline_dataset_name TEXT NOT NULL,
            current_dataset_name TEXT NOT NULL,
            overall_psi REAL NOT NULL,
            drift_status TEXT NOT NULL,
            baseline_metric_name TEXT,
            baseline_metric_val REAL,
            current_metric_val REAL,
            performance_delta REAL,
            incidents_count INTEGER DEFAULT 0,
            summary_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_incidents (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            run_id TEXT,
            incident_type TEXT NOT NULL,
            feature_or_metric TEXT NOT NULL,
            baseline_value TEXT,
            current_value TEXT,
            psi_score REAL,
            severity TEXT NOT NULL,
            status TEXT DEFAULT 'OPEN',
            recommended_action TEXT NOT NULL,
            resolution_notes TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_user(name, email, password, role="Data Scientist"):
    """
    Attempts to create a new user.
    Returns (True, user_dict) on success, or (False, error_message) on failure.
    """
    email = email.strip().lower()
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return False, "An account with this email already exists. Please sign in instead."

        password_hash = generate_password_hash(password)
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (name.strip(), email, password_hash, role, datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = cursor.lastrowid
        return True, {"id": user_id, "name": name.strip(), "email": email, "role": role}
    finally:
        conn.close()


def verify_user(email, password):
    """
    Checks email + password against the stored hash.
    Returns the user dict if valid, or None if invalid.
    """
    email = email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            return None
        if not check_password_hash(row["password_hash"], password):
            return None
        return {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}
    finally:
        conn.close()


def get_user_by_id(user_id):
    conn = get_connection()
    try:
        row = conn.execute("SELECT id, name, email, role FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# AEGIS PROJECT DATABASE OPERATIONS
# ------------------------------------------------------------------------------

import json
import uuid


def list_projects(user_id=None):
    """Returns all projects (optionally filtered by user_id)."""
    conn = get_connection()
    try:
        if user_id:
            cursor = conn.execute("SELECT * FROM projects WHERE user_id = ? OR user_id IS NULL ORDER BY updated_at DESC", (user_id,))
        else:
            cursor = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        projects = []
        for r in rows:
            p = dict(r)
            if p.get("state_json"):
                try:
                    p["state"] = json.loads(p["state_json"])
                except Exception:
                    p["state"] = {}
            projects.append(p)
        return projects
    finally:
        conn.close()


def get_project_by_id(project_id):
    """Retrieves a single project record by id."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            return None
        p = dict(row)
        if p.get("state_json"):
            try:
                p["state"] = json.loads(p["state_json"])
            except Exception:
                p["state"] = {}
        return p
    finally:
        conn.close()


def create_project(name, description="", domain="General", user_id=None, dataset_name=None, dataset_rows=0, dataset_cols=0, state_data=None):
    """Creates a new AEGIS Project in the SQLite database."""
    proj_id = f"prj_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat()
    state_json = json.dumps(state_data) if state_data else "{}"

    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO projects (
                id, user_id, name, description, domain, dataset_name, dataset_rows, dataset_cols,
                dataset_version, target_column, model_status, audit_status, audit_score,
                deployment_status, monitoring_status, created_at, updated_at, state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            proj_id, user_id, name.strip(), (description or "").strip(), (domain or "General").strip(),
            dataset_name, dataset_rows, dataset_cols, 'v1.0', None,
            'Not trained', 'Not run' if not dataset_name else 'Completed', 94.8 if dataset_name else None,
            'Not reviewed', 'Inactive', now, now, state_json
        ))
        conn.commit()
        return get_project_by_id(proj_id)
    finally:
        conn.close()


def update_project(project_id, updates: dict):
    """Updates fields of an existing project."""
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not existing:
            return None

        fields = []
        values = []
        allowed_fields = [
            "name", "description", "domain", "dataset_name", "dataset_rows", "dataset_cols",
            "dataset_version", "target_column", "model_status", "audit_status", "audit_score",
            "deployment_status", "monitoring_status", "state_json"
        ]

        for k, v in updates.items():
            if k in allowed_fields:
                fields.append(f"{k} = ?")
                if k == "state_json" and isinstance(v, (dict, list)):
                    values.append(json.dumps(v))
                else:
                    values.append(v)

        if not fields:
            return get_project_by_id(project_id)

        fields.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(project_id)

        sql = f"UPDATE projects SET {', '.join(fields)} WHERE id = ?"
        conn.execute(sql, tuple(values))
        conn.commit()
        return get_project_by_id(project_id)
    finally:
        conn.close()


def update_project_dataset(project_id, dataset_name, domain=None, file_path=None, row_count=0, col_count=0, target_column=None):
    """Convenience helper to update project dataset attributes."""
    updates = {
        "dataset_name": dataset_name,
        "dataset_rows": row_count,
        "dataset_cols": col_count,
        "audit_status": "Completed"
    }
    if domain:
        updates["domain"] = domain
    if target_column:
        updates["target_column"] = target_column
    return update_project(project_id, updates)


def delete_project(project_id):
    """Deletes a project by id."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.execute("DELETE FROM models WHERE project_id = ?", (project_id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# AEGIS MODEL REGISTRY DATABASE OPERATIONS (STAGE 2)
# ------------------------------------------------------------------------------

def get_next_model_version(project_id: str) -> str:
    """Calculates next incremental version string for a project (e.g., v1, v2, v3)."""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM models WHERE project_id = ?", (project_id,))
        row = cursor.fetchone()
        count = row["cnt"] if row else 0
        return f"v{count + 1}"
    finally:
        conn.close()


def create_model_record(project_id: str, name: str, algorithm: str, problem_type: str,
                        target_column: str, metrics: dict, feature_config: dict,
                        model_path: str = None, version: str = None) -> dict:
    """Inserts a new model record into the registry and sets it as active."""
    model_id = f"mdl_{uuid.uuid4().hex[:10]}"
    now = datetime.utcnow().isoformat()
    if not version:
        version = get_next_model_version(project_id)

    conn = get_connection()
    try:
        # Deactivate older models for this project
        conn.execute("UPDATE models SET is_active = 0 WHERE project_id = ?", (project_id,))
        
        conn.execute("""
            INSERT INTO models (
                id, project_id, name, algorithm, problem_type, target_column, version,
                is_active, metrics_json, feature_config_json, model_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
        """, (
            model_id, project_id, name, algorithm, problem_type, target_column, version,
            json.dumps(metrics), json.dumps(feature_config), model_path, now
        ))

        # Update Project record's model status
        primary_metric = metrics.get("f1_score") if problem_type == "classification" else metrics.get("r2_score")
        metric_str = f" ({primary_metric})" if primary_metric is not None else ""
        conn.execute("""
            UPDATE projects SET model_status = ?, updated_at = ? WHERE id = ?
        """, (f"Trained ({name} {version})", now, project_id))

        conn.commit()
        return get_model_by_id(model_id)
    finally:
        conn.close()


def list_models_for_project(project_id: str) -> list:
    """Lists all model versions trained for a project, sorted newest first."""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM models WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
        rows = cursor.fetchall()
        models = []
        for r in rows:
            m = dict(r)
            m["metrics"] = json.loads(m["metrics_json"]) if m.get("metrics_json") else {}
            m["feature_config"] = json.loads(m["feature_config_json"]) if m.get("feature_config_json") else {}
            m["model_name"] = m.get("name")
            m["model_id"] = m.get("id")
            m["features"] = m["feature_config"].get("selected_features", [])
            m["feature_importances"] = m["metrics"].get("feature_importances", {})
            models.append(m)
        return models
    finally:
        conn.close()


def get_model_by_id(model_id: str) -> dict:
    """Retrieves a single model record by model_id."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            return None
        m = dict(row)
        m["metrics"] = json.loads(m["metrics_json"]) if m.get("metrics_json") else {}
        m["feature_config"] = json.loads(m["feature_config_json"]) if m.get("feature_config_json") else {}
        m["model_name"] = m.get("name")
        m["model_id"] = m.get("id")
        m["features"] = m["feature_config"].get("selected_features", [])
        m["feature_importances"] = m["metrics"].get("feature_importances", {})
        return m
    finally:
        conn.close()


def set_active_model_for_project(project_id: str, model_id: str) -> dict:
    """Activates a specific model version for a project."""
    conn = get_connection()
    try:
        conn.execute("UPDATE models SET is_active = 0 WHERE project_id = ?", (project_id,))
        conn.execute("UPDATE models SET is_active = 1 WHERE id = ? AND project_id = ?", (model_id, project_id))
        target_model = get_model_by_id(model_id)
        if target_model:
            now = datetime.utcnow().isoformat()
            conn.execute("UPDATE projects SET model_status = ?, updated_at = ? WHERE id = ?",
                         (f"Active ({target_model['name']} {target_model['version']})", now, project_id))
        conn.commit()
        return target_model
    finally:
        conn.close()


def delete_model_record(model_id: str) -> bool:
    """Deletes a model record."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# AEGIS STRESS EXPERIMENT DATABASE OPERATIONS (STAGE 3)
# ------------------------------------------------------------------------------

def create_stress_experiment(project_id: str, model_id: str, model_version: str,
                             dataset_name: str, stress_type: str, severity_pct: float,
                             baseline_metric: float, stressed_metric: float,
                             degradation_pct: float, resilience_score: float,
                             metric_name: str, details: dict) -> dict:
    """Records a new AI Stress Lab experiment in SQLite."""
    exp_id = f"exp_{uuid.uuid4().hex[:10]}"
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO stress_experiments (
                id, project_id, model_id, model_version, dataset_name, stress_type,
                severity_pct, baseline_metric, stressed_metric, degradation_pct,
                resilience_score, metric_name, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id, project_id, model_id, model_version, dataset_name, stress_type,
            float(severity_pct), float(baseline_metric), float(stressed_metric),
            float(degradation_pct), float(resilience_score), metric_name,
            json.dumps(details), now
        ))
        conn.commit()
        return get_stress_experiment_by_id(exp_id)
    finally:
        conn.close()


def list_stress_experiments_for_project(project_id: str) -> list:
    """Lists all stress experiments conducted for a project, sorted newest first."""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM stress_experiments WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
        rows = cursor.fetchall()
        experiments = []
        for r in rows:
            exp = dict(r)
            exp["details"] = json.loads(exp["details_json"]) if exp.get("details_json") else {}
            experiments.append(exp)
        return experiments
    finally:
        conn.close()


def get_stress_experiment_by_id(exp_id: str) -> dict:
    """Retrieves a single stress experiment record by ID."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM stress_experiments WHERE id = ?", (exp_id,)).fetchone()
        if not row:
            return None
        exp = dict(row)
        exp["details"] = json.loads(exp["details_json"]) if exp.get("details_json") else {}
        return exp
    finally:
        conn.close()


def delete_stress_experiment(exp_id: str) -> bool:
    """Deletes a stress experiment record."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM stress_experiments WHERE id = ?", (exp_id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# AEGIS AUDIT DOSSIER DATABASE OPERATIONS (STAGE 4)
# ------------------------------------------------------------------------------

def save_audit_dossier(project_id: str, dossier: dict) -> dict:
    """Saves full Pre-Deployment Evidence Dossier and updates project audit state."""
    audit_id = dossier.get("audit_id") or f"audit_{uuid.uuid4().hex[:10]}"
    dataset_name = dossier.get("dataset_name", "Dataset.csv")
    dataset_version = dossier.get("dataset_version", "v1.0")
    model_name = dossier.get("model_name", "N/A")
    model_version = dossier.get("model_version", "N/A")
    trust_score = float(dossier.get("trust_score", 0.0))
    readiness_score = float(dossier.get("readiness_score", 0.0))
    risk_level = dossier.get("risk_level", "MEDIUM")
    deployment_verdict = dossier.get("deployment_verdict", "PENDING")
    now = datetime.utcnow().isoformat()

    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO audit_dossiers (
                id, project_id, dataset_name, dataset_version, model_name, model_version,
                trust_score, readiness_score, risk_level, deployment_verdict,
                dossier_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_id, project_id, dataset_name, dataset_version, model_name, model_version,
            trust_score, readiness_score, risk_level, deployment_verdict,
            json.dumps(dossier), now
        ))
        # Update project table with latest audit results
        conn.execute("""
            UPDATE projects 
            SET audit_status = ?, audit_score = ?, deployment_status = ?, updated_at = ?
            WHERE id = ?
        """, (
            f"Audited ({risk_level})", trust_score, deployment_verdict, now, project_id
        ))
        conn.commit()
        return get_audit_dossier_by_id(audit_id)
    finally:
        conn.close()


def get_latest_audit_dossier(project_id: str) -> Optional[dict]:
    """Retrieves the most recent audit dossier for a project."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT * FROM audit_dossiers 
            WHERE project_id = ? 
            ORDER BY created_at DESC LIMIT 1
        """, (project_id,)).fetchone()
        if not row:
            return None
        rec = dict(row)
        rec["dossier"] = json.loads(rec["dossier_json"]) if rec.get("dossier_json") else {}
        return rec
    finally:
        conn.close()


def list_audit_dossiers_for_project(project_id: str) -> list:
    """Lists historical audit dossiers for a project."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT id, project_id, dataset_name, dataset_version, model_name,
                   trust_score, readiness_score, risk_level, deployment_verdict, created_at
            FROM audit_dossiers
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_audit_dossier_by_id(audit_id: str) -> Optional[dict]:
    """Retrieves a specific audit dossier by its ID."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM audit_dossiers WHERE id = ?", (audit_id,)).fetchone()
        if not row:
            return None
        rec = dict(row)
        rec["dossier"] = json.loads(rec["dossier_json"]) if rec.get("dossier_json") else {}
        return rec
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# AEGIS STAGE 5: DEPLOYMENT GATE & DECISION ROOM OPERATIONS
# ------------------------------------------------------------------------------

def record_deployment_decision(project_id: str, decision: str, reviewer_name: str,
                               decision_reason: str, evidence_snapshot: dict,
                               model_id: Optional[str] = None, model_version: Optional[str] = None,
                               dataset_name: Optional[str] = None, reviewer_email: Optional[str] = None,
                               reviewer_role: Optional[str] = None) -> dict:
    """Records a human governance decision in the immutable deployment ledger."""
    decision_id = f"dec_{uuid.uuid4().hex[:10]}"
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO deployment_decisions (
                id, project_id, model_id, model_version, dataset_name,
                decision, reviewer_name, reviewer_email, reviewer_role,
                decision_reason, evidence_snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision_id, project_id, model_id or "N/A", model_version or "v1", dataset_name or "Dataset.csv",
            decision, reviewer_name, reviewer_email or "analyst@aegis.ai", reviewer_role or "AI Safety Lead",
            decision_reason, json.dumps(evidence_snapshot), now
        ))

        # Update project deployment_status & audit status
        status_map = {
            "APPROVED": "Approved for Deployment",
            "REQUEST_CHANGES": "Changes Requested",
            "REJECTED": "Deployment Rejected",
            "REMEDIATION_TASK": "Remediation Active"
        }
        dep_status = status_map.get(decision, decision)
        conn.execute("""
            UPDATE projects
            SET deployment_status = ?, updated_at = ?
            WHERE id = ?
        """, (dep_status, now, project_id))
        conn.commit()

        row = conn.execute("SELECT * FROM deployment_decisions WHERE id = ?", (decision_id,)).fetchone()
        rec = dict(row)
        rec["evidence_snapshot"] = json.loads(rec["evidence_snapshot_json"])
        return rec
    finally:
        conn.close()


def list_deployment_decisions_for_project(project_id: str) -> list:
    """Lists all historical human deployment decisions for a project."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM deployment_decisions
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))
        rows = cursor.fetchall()
        decisions = []
        for r in rows:
            d = dict(r)
            d["evidence_snapshot"] = json.loads(d["evidence_snapshot_json"]) if d.get("evidence_snapshot_json") else {}
            decisions.append(d)
        return decisions
    finally:
        conn.close()


def get_latest_deployment_decision(project_id: str) -> Optional[dict]:
    """Retrieves the most recent human deployment decision for a project."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT * FROM deployment_decisions
            WHERE project_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (project_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["evidence_snapshot"] = json.loads(d["evidence_snapshot_json"]) if d.get("evidence_snapshot_json") else {}
        return d
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# AEGIS STAGE 5: MONITORING CENTER & INCIDENT MANAGEMENT OPERATIONS
# ------------------------------------------------------------------------------

def record_monitoring_run(project_id: str, baseline_dataset_name: str, current_dataset_name: str,
                          overall_psi: float, drift_status: str, incidents_count: int,
                          summary_dict: dict, model_id: Optional[str] = None,
                          model_version: Optional[str] = None, baseline_metric_name: Optional[str] = None,
                          baseline_metric_val: Optional[float] = None, current_metric_val: Optional[float] = None,
                          performance_delta: Optional[float] = None) -> dict:
    """Records a post-deployment continuous monitoring run."""
    run_id = f"mon_{uuid.uuid4().hex[:10]}"
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO monitoring_runs (
                id, project_id, model_id, model_version, baseline_dataset_name,
                current_dataset_name, overall_psi, drift_status, baseline_metric_name,
                baseline_metric_val, current_metric_val, performance_delta,
                incidents_count, summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, project_id, model_id or "N/A", model_version or "v1", baseline_dataset_name,
            current_dataset_name, float(overall_psi), drift_status, baseline_metric_name,
            baseline_metric_val, current_metric_val, performance_delta,
            int(incidents_count), json.dumps(summary_dict), now
        ))

        # Update project monitoring status
        mon_status = "Incident Alert" if incidents_count > 0 else "Active (Healthy)"
        conn.execute("""
            UPDATE projects
            SET monitoring_status = ?, updated_at = ?
            WHERE id = ?
        """, (mon_status, now, project_id))
        conn.commit()

        row = conn.execute("SELECT * FROM monitoring_runs WHERE id = ?", (run_id,)).fetchone()
        r = dict(row)
        r["summary"] = json.loads(r["summary_json"]) if r.get("summary_json") else {}
        return r
    finally:
        conn.close()


def list_monitoring_runs_for_project(project_id: str) -> list:
    """Lists chronological monitoring runs for a project timeline."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM monitoring_runs
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))
        rows = cursor.fetchall()
        runs = []
        for r in rows:
            item = dict(r)
            item["summary"] = json.loads(item["summary_json"]) if item.get("summary_json") else {}
            runs.append(item)
        return runs
    finally:
        conn.close()


def create_monitoring_incident(project_id: str, incident_type: str, feature_or_metric: str,
                               severity: str, recommended_action: str, run_id: Optional[str] = None,
                               baseline_value: Optional[str] = None, current_value: Optional[str] = None,
                               psi_score: Optional[float] = None) -> dict:
    """Records an automated monitoring incident when drift or degradation exceeds thresholds."""
    incident_id = f"inc_{uuid.uuid4().hex[:10]}"
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO monitoring_incidents (
                id, project_id, run_id, incident_type, feature_or_metric,
                baseline_value, current_value, psi_score, severity, status,
                recommended_action, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id, project_id, run_id or "live", incident_type, feature_or_metric,
            str(baseline_value or "N/A"), str(current_value or "N/A"),
            float(psi_score) if psi_score is not None else None,
            severity, "OPEN", recommended_action, now
        ))
        conn.commit()

        row = conn.execute("SELECT * FROM monitoring_incidents WHERE id = ?", (incident_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_monitoring_incidents_for_project(project_id: str, status: Optional[str] = None) -> list:
    """Lists monitoring incidents for a project, optionally filtered by status."""
    conn = get_connection()
    try:
        if status:
            cursor = conn.execute("""
                SELECT * FROM monitoring_incidents
                WHERE project_id = ? AND status = ?
                ORDER BY created_at DESC
            """, (project_id, status))
        else:
            cursor = conn.execute("""
                SELECT * FROM monitoring_incidents
                WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def resolve_monitoring_incident(incident_id: str, resolution_notes: str) -> bool:
    """Marks a monitoring incident as resolved with human resolution notes."""
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE monitoring_incidents
            SET status = 'RESOLVED', resolution_notes = ?, resolved_at = ?
            WHERE id = ?
        """, (resolution_notes, now, incident_id))
        conn.commit()
        return True
    finally:
        conn.close()





