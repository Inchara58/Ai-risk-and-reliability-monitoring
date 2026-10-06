# AI Risk & Reliability Platform — Architecture Blueprint & Technical Specifications

**Tagline**: *"AI Powered Dataset Audit, Cleaning, Risk Assessment & Deployment Readiness Platform"*

---

## 1. System Architecture & Data Flow

```
                      [ User Web Browser / Client UI ]
                                     │
                        (HTTPS / REST APIs / WebSockets)
                                     ▼
                    [ Nginx Reverse Proxy & SSL Termination ]
                                     │
                   ┌─────────────────┴─────────────────┐
                   │                                   │
                   ▼                                   ▼
        [ Static Assets & CDN ]            [ Gunicorn WSGI Server ]
        (HTML5, CSS3, JS, Charts)                      │
                                                       ▼
                                            [ Flask App Container ]
                                                       │
         ┌───────────────────┬─────────────────────────┼─────────────────────────┐
         ▼                   ▼                         ▼                         ▼
 [ Validation Module ]  [ Cleaning Module ]   [ AI Risk & Trust Engine ] [ Bias & Drift Engine ]
 (Pandas, Schema)       (Mean/Median, Outlier) (Scikit-Learn, SciPy)     (PSI, Disparate Impact)
         │                   │                         │                         │
         └───────────────────┴─────────────────────────┼─────────────────────────┘
                                                       │
                                  ┌────────────────────┴────────────────────┐
                                  ▼                                         ▼
                       [ Redis Task Queue & Celery ]             [ MySQL 8.0 Database ]
                       (Async Heavy Audit Jobs)                (Users, Audits, Scores)
                                  │
                                  ▼
                       [ PDF Generator Module ]
                       (ReportLab / Print Engine)
```

---

## 2. Database Design & ER Diagram (MySQL DDL Schemas)

### Entity Relationship Structure
- `users` (1) ─── (N) `projects`
- `projects` (1) ─── (N) `datasets`
- `datasets` (1) ─── (1) `audit_reports`
- `audit_reports` (1) ─── (1) `risk_scores`
- `audit_reports` (1) ─── (N) `recommendations`
- `audit_reports` (1) ─── (1) `bias_metrics`
- `audit_reports` (1) ─── (1) `deployment_reports`
- `datasets` (1) ─── (N) `dataset_drift` (Current vs Baseline)

### Complete SQL DDL Script

```sql
CREATE DATABASE IF NOT EXISTS dataguard_ai_db;
USE dataguard_ai_db;

-- 1. Users Table
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('ML_ENGINEER', 'DATA_SCIENTIST', 'ARCHITECT', 'AUDITOR') DEFAULT 'DATA_SCIENTIST',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Projects Table
CREATE TABLE projects (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 3. Datasets Table
CREATE TABLE datasets (
    id VARCHAR(36) PRIMARY KEY,
    project_id VARCHAR(36) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    row_count INT NOT NULL,
    column_count INT NOT NULL,
    version VARCHAR(20) DEFAULT 'v1.0',
    storage_s3_path VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- 4. Audit Reports Table
CREATE TABLE audit_reports (
    id VARCHAR(36) PRIMARY KEY,
    dataset_id VARCHAR(36) NOT NULL UNIQUE,
    missing_count INT DEFAULT 0,
    missing_pct DECIMAL(5,2) DEFAULT 0.00,
    duplicate_count INT DEFAULT 0,
    duplicate_pct DECIMAL(5,2) DEFAULT 0.00,
    memory_usage_bytes BIGINT DEFAULT 0,
    completeness_score DECIMAL(5,2) DEFAULT 100.00,
    raw_summary_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

-- 5. Risk Scores Table
CREATE TABLE risk_scores (
    id VARCHAR(36) PRIMARY KEY,
    audit_report_id VARCHAR(36) NOT NULL UNIQUE,
    trust_score DECIMAL(5,2) NOT NULL,
    reliability_score DECIMAL(5,2) NOT NULL,
    readiness_score DECIMAL(5,2) NOT NULL,
    overall_risk_level ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    rationale_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audit_report_id) REFERENCES audit_reports(id) ON DELETE CASCADE
);

-- 6. Bias Metrics Table
CREATE TABLE bias_metrics (
    id VARCHAR(36) PRIMARY KEY,
    audit_report_id VARCHAR(36) NOT NULL,
    protected_attribute VARCHAR(100) NOT NULL,
    disparate_impact_ratio DECIMAL(5,3) NOT NULL,
    class_imbalance_ratio VARCHAR(50),
    fairness_score DECIMAL(5,2) NOT NULL,
    is_compliant BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (audit_report_id) REFERENCES audit_reports(id) ON DELETE CASCADE
);

-- 7. Dataset Drift Table
CREATE TABLE dataset_drift (
    id VARCHAR(36) PRIMARY KEY,
    current_dataset_id VARCHAR(36) NOT NULL,
    baseline_dataset_id VARCHAR(36) NOT NULL,
    psi_score DECIMAL(5,4) NOT NULL,
    ks_test_p_val DECIMAL(6,4),
    drift_detected BOOLEAN DEFAULT FALSE,
    drift_columns_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (current_dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (baseline_dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

-- 8. Recommendations Table
CREATE TABLE recommendations (
    id VARCHAR(36) PRIMARY KEY,
    audit_report_id VARCHAR(36) NOT NULL,
    priority ENUM('HIGH', 'MEDIUM', 'LOW') NOT NULL,
    category VARCHAR(100) NOT NULL,
    recommendation_text TEXT NOT NULL,
    expected_accuracy_boost_pct DECIMAL(4,2),
    FOREIGN KEY (audit_report_id) REFERENCES audit_reports(id) ON DELETE CASCADE
);

-- 9. Deployment Reports Table
CREATE TABLE deployment_reports (
    id VARCHAR(36) PRIMARY KEY,
    audit_report_id VARCHAR(36) NOT NULL UNIQUE,
    status_verdict ENUM('READY_FOR_DEPLOYMENT', 'NEEDS_CLEANING', 'HIGH_RISK_BLOCKED', 'CRITICAL_BIAS') NOT NULL,
    signoff_user_id VARCHAR(36),
    deployment_checklist_json JSON,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audit_report_id) REFERENCES audit_reports(id) ON DELETE CASCADE,
    FOREIGN KEY (signoff_user_id) REFERENCES users(id) ON DELETE SET NULL
);
```

---

## 3. RESTful API Specification

| Endpoint | Method | Description | Request Body Payload | Response Payload |
|---|---|---|---|---|
| `/api/v1/auth/login` | POST | Authenticate user & issue JWT | `{ "email": "...", "password": "..." }` | `{ "token": "jwt_...", "user": {...} }` |
| `/api/v1/datasets/upload` | POST | Upload CSV/Excel dataset | `multipart/form-data (file)` | `{ "dataset_id": "ds_101", "rows": 50000 }` |
| `/api/v1/datasets/:id/validate` | GET | Structural validation check | None | `{ "status": "VALID", "encoding": "UTF-8" }` |
| `/api/v1/datasets/:id/clean` | POST | Execute auto-cleaning | `{ "rules": ["impute", "dedupe"] }` | `{ "cleaned_dataset_id": "ds_102", "diff": {...} }` |
| `/api/v1/datasets/:id/audit` | GET | Comprehensive audit report | None | `{ "audit_report_id": "aud_88", "metrics": {...} }` |
| `/api/v1/datasets/:id/risk` | GET | Algorithmic risk scores | None | `{ "trust_score": 94.8, "verdict": "READY" }` |
| `/api/v1/datasets/:id/report/pdf` | GET | Download compliance PDF | None | PDF File Binary Stream |

---

## 4. Docker & AWS EC2 Deployment Blueprint

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  web:
    build: .
    command: gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
    environment:
      - DATABASE_URL=mysql+pymysql://user:pass@db:3306/dataguard_ai_db
      - REDIS_URL=redis://redis:6379/0
    expose:
      - "5000"
    depends_on:
      - db
      - redis

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web

  db:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: dataguard_ai_db
      MYSQL_ROOT_PASSWORD: secret_root_pass
    volumes:
      - mysql_data:/var/lib/mysql

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  mysql_data:
```

### GitHub Actions CI/CD Workflow (`.github/workflows/deploy.yml`)
```yaml
name: Enterprise MLOps Deploy to AWS EC2

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Run PyTest & Data Integrity Checks
        run: |
          pip install -r requirements.txt
          pytest tests/

      - name: Build & Push Docker Image to ECR
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}
          docker build -t dataguard-ai:latest .
          docker tag dataguard-ai:latest ${{ secrets.ECR_REGISTRY }}/dataguard-ai:latest
          docker push ${{ secrets.ECR_REGISTRY }}/dataguard-ai:latest

      - name: Deploy to AWS EC2 via SSH
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_PUBLIC_IP }}
          username: ec2-user
          key: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
          script: |
            cd /var/www/ai_risk_platform
            docker-compose pull
            docker-compose up -d --remove-orphans
```
