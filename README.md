<div align="center">

# SmartPayroll AI

### End-to-End HR Intelligence & Payroll Analytics System
### Built on Microsoft Azure AI Foundry

[![CI](https://github.com/Najam0786/smartpayroll-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Najam0786/smartpayroll-ai/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Azure](https://img.shields.io/badge/Azure-AI%20Foundry-0078D4?logo=microsoftazure)](https://ai.azure.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.6-green)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2?logo=mlflow)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-30%2F30%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*From raw HR data to production AI agents — demonstrating enterprise-grade data science and AI engineering on Azure*

[Architecture](#architecture) • [Results](#key-results) • [Quick Start](#quick-start) • [API Docs](#api-endpoints) • [Project Structure](#project-structure)

</div>

---

## Overview

SmartPayroll AI is a production-grade, end-to-end intelligent HR and payroll analytics system built on **Microsoft Azure AI Foundry**. It demonstrates the complete data science and AI engineering lifecycle — from raw data ingestion through classical machine learning, retrieval-augmented generation, multi-agent AI orchestration, and a deployed REST API.

### What It Solves

| Business Problem | AI Solution |
|-----------------|-------------|
| Which employees are at risk of leaving? | XGBoost attrition prediction (AUC 0.772) |
| What does our HR policy say about leave? | RAG pipeline with semantic search |
| Which payroll records are anomalous? | Two-layer detection (Recall 94.4%) |
| How do I investigate an employee situation? | LangGraph multi-agent audit system |
| How do I integrate this into existing systems? | FastAPI REST service |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SmartPayroll AI System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Raw Data (IBM HR Analytics CSV)                                  │
│       │                                                           │
│       ▼ ─────────────── Data Pipeline ──────────────────         │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │ Ingest  │───▶│ Validate │───▶│  Clean   │───▶│  Store   │    │
│  │ (Azure  │    │(5 checks)│    │(Bronze → │    │(Parquet) │    │
│  │  Blob)  │    │          │    │ Silver)  │    │          │    │
│  └─────────┘    └──────────┘    └──────────┘    └──────────┘    │
│       │                                                           │
│       ▼ ─────────── Intelligence Layer ─────────────────         │
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Classical ML   │    │   RAG Pipeline  │    │  Anomaly    │  │
│  │─────────────────│    │─────────────────│    │  Detection  │  │
│  │ • LR Baseline   │    │ • Chunking      │    │─────────────│  │
│  │ • XGBoost+SMOTE │    │ • Embeddings    │    │ • Rules     │  │
│  │ • MLflow Track  │    │ • Cosine Search │    │ • Isolation │  │
│  │ • AUC: 0.772    │    │ • Phi-4-mini   │    │   Forest    │  │
│  └────────┬────────┘    └────────┬────────┘    └──────┬──────┘  │
│           │                      │                     │         │
│           └──────────────────────┼─────────────────────┘         │
│                                  ▼                                │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              LangGraph Multi-Agent System                │     │
│  │─────────────────────────────────────────────────────────│     │
│  │                                                          │     │
│  │  START → Supervisor → Employee Agent → Risk Agent        │     │
│  │                              │                           │     │
│  │                    ┌─────────┴──────────┐                │     │
│  │                    │  Conditional Route  │                │     │
│  │                    └─────────┬──────────┘                │     │
│  │                 HIGH ◄───────┴──────► LOW/MEDIUM          │     │
│  │                   │                      │                │     │
│  │              HITL Node            Department Agent        │     │
│  │                   │                      │                │     │
│  │                   └──────────┬───────────┘                │     │
│  │                              ▼                            │     │
│  │                       Report Agent → END                  │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                  │                                │
│                                  ▼                                │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                   FastAPI REST Service                   │     │
│  │   GET /health  │  GET /employees/{id}                   │     │
│  │   POST /attrition/risk  │  POST /investigate            │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Cloud & AI Platform

| Component | Technology | Purpose |
|-----------|-----------|---------|
| AI Platform | Microsoft Azure AI Foundry | Model deployment, management |
| Chat Model | Phi-4-mini-instruct (Microsoft) | RAG generation, agent reasoning |
| Embedding Model | text-embedding-3-small (OpenAI) | 1,536-dim semantic vectors |
| Region | Sweden Central | EU data residency |

### Data Engineering

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Validation | Pandera | Schema contracts, quality gates |
| Data Processing | Pandas, PyArrow | Transformation, Parquet I/O |
| Architecture | Medallion (Bronze → Silver → Gold) | Layered data quality |
| Synthetic Data | Faker (es_ES) | Realistic test data generation |

### Machine Learning

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Baseline | Scikit-learn Logistic Regression | Interpretable benchmark |
| Production Model | XGBoost | Gradient boosted attrition prediction |
| Imbalance Handling | SMOTE (imblearn) | Synthetic minority oversampling |
| Experiment Tracking | MLflow 2.10 | Parameters, metrics, model registry |
| Anomaly Detection | Isolation Forest | Unsupervised payroll anomaly detection |
| Explainability | SHAP | Feature importance attribution |

### AI & Agents

| Component | Technology | Purpose |
|-----------|-----------|---------|
| RAG Pipeline | Custom cosine similarity | HR policy retrieval |
| Agent Framework | LangGraph 1.2.6 | Multi-agent state machine |
| Tool Calling | Custom Python tools | Employee lookup, risk scoring |
| HITL Pattern | LangGraph interrupt | Human approval for HIGH risk |
| LLM Client | OpenAI SDK (Azure) | Chat completions |

### Production Services

| Component | Technology | Purpose |
|-----------|-----------|---------|
| REST API | FastAPI 0.110 | HTTP service layer |
| Validation | Pydantic v2 | Request/response models |
| Server | Uvicorn | ASGI production server |
| CI/CD | GitHub Actions | Automated quality gates |
| Testing | Pytest | 30 unit tests, 100% passing |

---

## Key Results

### Machine Learning

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression (baseline) | 0.772 | 0.411 | 0.298 | 0.660 |
| **XGBoost + SMOTE** | **0.750** | **0.490** | **0.471** | **0.511** |

> **Note:** Logistic Regression achieves higher AUC while XGBoost achieves higher F1. For HR attrition, recall matters most — catching employees about to leave is more valuable than avoiding false positives.

### RAG Pipeline

| Metric | Result |
|--------|--------|
| Embedding dimensions | 1,536 |
| Top chunk similarity (relevant query) | 0.72+ |
| Documents indexed | 2 HR policies |
| Chunks generated | 7 |
| Chunk size | 500 characters, 50 overlap |

### Anomaly Detection

| Metric | Result |
|--------|--------|
| Recall | **94.4%** |
| Precision | 60.7% |
| True positives | 34 / 36 |
| Anomaly types detected | 5 |
| Records processed | 1,470 / month |

### Engineering

| Metric | Result |
|--------|--------|
| Unit tests | **30 / 30 passing** |
| Test execution time | 3.34 seconds |
| Pull requests merged | 13 |
| Feature branches | 13 |
| Commits on main | 28 |
| CI pipeline | GitHub Actions (lint + test) |

---

## EDA Key Findings

From the complete exploratory analysis of 1,470 IBM HR employees:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Class Imbalance** | 16.1% attrition — requires SMOTE or class_weight |
| 2 | **Overtime = #1 Risk** | 30.5% vs 10.4% attrition rate — **2.9x multiplier** |
| 3 | **Salary Gap** | Leavers earn **$2,002/month less** than stayers |
| 4 | **Experience Protects** | TotalWorkingYears: strongest negative correlation (−0.171) |
| 5 | **Job Level Matters** | Higher job level strongly associated with retention (−0.169) |
| 6 | **Distance Effect** | Distance from home positively correlated (+0.078) |

---

## Project Structure

```
smartpayroll-ai/
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions: lint + test on every PR
│
├── src/
│   ├── data/
│   │   ├── ingest.py                 # Load CSV (local or Azure Blob Storage)
│   │   ├── validate.py               # Pandera schema validation (5 checks)
│   │   ├── clean.py                  # Bronze → Silver transformation
│   │   ├── pipeline.py               # ETL orchestrator (ingest→validate→clean→save)
│   │   └── synthetic_generator.py    # Payroll data generator (8,820 records, 2% anomalies)
│   │
│   ├── features/
│   │   └── feature_engineering.py    # Encoding, scaling, stratified split (46 features)
│   │
│   ├── models/
│   │   ├── attrition/
│   │   │   └── train.py              # LR baseline + XGBoost + SMOTE + MLflow tracking
│   │   └── anomaly/
│   │       └── detect.py             # Rules + Isolation Forest (Recall 94.4%)
│   │
│   ├── rag/
│   │   ├── document_processor.py     # Chunk (500 chars) + embed (1,536-dim) + index
│   │   └── chain.py                  # Cosine similarity retrieval + Phi-4 generation
│   │
│   ├── agents/
│   │   ├── tools/
│   │   │   └── hr_tools.py           # 3 tools: employee details, risk, dept stats
│   │   ├── investigation_agent.py    # Single-agent investigation + batch risk scan
│   │   └── audit_graph.py            # LangGraph: 5 nodes, HITL, conditional routing
│   │
│   └── api/
│       └── main.py                   # FastAPI: 4 endpoints, Pydantic v2, Swagger UI
│
├── notebooks/
│   └── 01_data_exploration.ipynb     # Full EDA: 8 cells, 4 charts, 6 findings
│
├── tests/
│   └── unit/
│       └── test_data_pipeline.py     # 30 tests: validation, cleaning, features, tools
│
├── hr_policies/
│   ├── HR_Policy_Annual_Leave_ES.md  # Spanish annual leave policy (Article 38)
│   └── HR_Policy_Overtime.md         # EU overtime policy (Working Time Directive)
│
├── docs/
│   ├── eda_01_attrition.png          # Target variable distribution
│   ├── eda_02_overtime.png           # Overtime 2.9x attrition risk
│   ├── eda_03_salary.png             # $2,002/month salary gap
│   └── eda_04_correlation.png        # Feature correlation heatmap
│
├── data/
│   ├── raw/                          # Original CSV (gitignored)
│   ├── processed/                    # Silver layer Parquet (gitignored)
│   └── synthetic/                    # Generated payroll records (gitignored)
│
├── .env.example                      # Environment variable template
├── .gitignore                        # Protects secrets, data, and ML artifacts
├── requirements.txt                  # All dependencies pinned (Python 3.11)
├── pyproject.toml                    # ruff + black + pytest configuration
└── README.md                         # This file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- Microsoft Azure account (AI Foundry with Phi-4-mini-instruct and text-embedding-3-small deployed)
- Kaggle account (for dataset download)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Najam0786/smartpayroll-ai.git
cd smartpayroll-ai

# 2. Create virtual environment (Python 3.11 required)
py -3.11 -m venv smartpayroll
smartpayroll\Scripts\activate        # Windows
# source smartpayroll/bin/activate   # Mac/Linux

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# Edit .env and add your Azure credentials:
# AZURE_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com
# AZURE_API_KEY=your-api-key
# AZURE_CHAT_DEPLOYMENT=Phi-4-mini-instruct
# AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### Data Setup

```bash
# Download IBM HR Analytics dataset from Kaggle:
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Save as: data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

### Run the Complete Pipeline

```bash
# Step 1: ETL Pipeline (Bronze → Silver)
python -m src.data.pipeline

# Step 2: Generate synthetic payroll data
python -m src.data.synthetic_generator

# Step 3: Train attrition model (tracked in MLflow)
python -m src.models.attrition.train

# Step 4: Train anomaly detection model
python -m src.models.anomaly.detect

# Step 5: Index HR policy documents
python -m src.rag.document_processor

# Step 6: Run investigation agent
python -m src.agents.investigation_agent

# Step 7: Run LangGraph multi-agent audit
python -m src.agents.audit_graph

# Step 8: Start REST API
python -m src.api.main
# → Swagger UI: http://localhost:8000/docs
# → Health check: http://localhost:8000/health

# Step 9: Run all tests
pytest tests/unit/ -v

# Step 10: View MLflow experiments
mlflow ui
# → http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/health` | System health check | None |
| `GET` | `/api/v1/employees/{id}` | Employee profile lookup | API Key |
| `POST` | `/api/v1/attrition/risk` | Attrition risk assessment | API Key |
| `POST` | `/api/v1/investigate` | Full investigation report | API Key |

### Example: Attrition Risk Assessment

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/attrition/risk \
  -H "Content-Type: application/json" \
  -d '{"employee_id": 7}'
```

**Response:**
```json
{
  "employee_id": 7,
  "risk_level": "HIGH",
  "risk_score": 9,
  "risk_factors": [
    "Working overtime — 2.9x higher attrition risk",
    "Low job satisfaction: 1/4",
    "Short tenure: 1 years",
    "Below median salary: $2,670"
  ],
  "recommendation": "Immediate retention conversation recommended",
  "department": "Research & Development",
  "monthly_income": 2670.0
}
```

### Example: Full Investigation

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/investigate \
  -H "Content-Type: application/json" \
  -d '{"employee_id": 7}'
```

**Response:**
```json
{
  "employee_id": 7,
  "status": "complete",
  "risk_level": "HIGH",
  "summary": "Employee 7 — Laboratory Technician in R&D\nRisk: HIGH (9/10)\nSalary: $2,670/month ($-1,704 vs dept median)\nAction: Immediate retention conversation recommended"
}
```

---

## LangGraph Multi-Agent Architecture

The audit system implements a production-grade multi-agent pattern:

```python
# State machine with 5 specialist nodes
graph = StateGraph(AuditState)

graph.add_node("supervisor_node", supervisor_node)      # Validates input
graph.add_node("employee_agent_node", employee_agent)   # Fetches profile
graph.add_node("risk_agent_node", risk_agent)           # Scores risk
graph.add_node("hitl_node", hitl_node)                  # Human review
graph.add_node("department_agent_node", dept_agent)     # Benchmarks
graph.add_node("report_agent_node", report_agent)       # Final report

# Conditional routing — HIGH risk triggers HITL
graph.add_conditional_edges(
    "risk_agent_node",
    route_after_risk,    # Returns "hitl_node" or "department_agent_node"
)
```

**Routing logic:**
- `HIGH risk (score ≥ 5)` → HITL node → Department → Report
- `MEDIUM/LOW risk` → Department directly → Report

---

## Anomaly Detection System

Two-layer detection for payroll anomalies:

### Layer 1: Deterministic Rules (Zero ML cost)

| Rule | Severity | Description |
|------|----------|-------------|
| `ZERO_OR_NEGATIVE_NET_PAY` | CRITICAL | Net pay ≤ 0 |
| `NET_EXCEEDS_GROSS` | CRITICAL | Net pay > gross pay |
| `MISSING_PENSION` | CRITICAL | Pension = 0 on salary > €1,500 |
| `TAX_RATE_TOO_HIGH` | CRITICAL | Exceeds statutory maximum |
| `TAX_RATE_TOO_LOW` | WARNING | Below statutory minimum |
| `EXCESSIVE_DEDUCTIONS` | WARNING | Total deductions > 80% of gross |

### Layer 2: Isolation Forest (Catches subtle patterns)

```python
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,    # Expected 2% anomaly rate
    random_state=42,
)
```

Features: `gross_pay, income_tax, social_security, pension, net_pay, net_to_gross_ratio, tax_rate, deduction_rate`

**Evaluation on synthetic data (ground truth known):**
- Recall: **94.4%** — catches 34 of 36 real anomalies
- Precision: **60.7%** — acceptable for payroll (over-flagging preferred)

---

## Engineering Standards

### Git Workflow

```
Every feature = one branch = one PR = one merge = delete branch

Branch naming:  feature/SA-{number}-{description}
Commit style:   conventional commits (feat/fix/chore/docs/test/ci)
PR rules:       no direct push to main, CI must pass
```

### Security

```
✅ Azure credentials in .env only — never in code
✅ .env in .gitignore — never committed to GitHub
✅ Raw data in .gitignore — no PII on GitHub
✅ ML artifacts in .gitignore — models stay local
```

### Code Quality

```
✅ ruff linting — PEP8 compliance
✅ Type hints throughout (Python 3.11)
✅ Docstrings on every function
✅ Structured logging (not print statements)
✅ Error handling with meaningful messages
```

---

## Synthetic Data Generator

Generates realistic payroll data for testing anomaly detection:

```
Total records:  8,820 (1,470 employees × 6 months)
Anomaly rate:   2.0% (176 records with known labels)
Anomaly types:
  HIGH_TAX_RATE:      46 records
  LARGE_DEVIATION:    42 records
  MISSING_PENSION:    34 records
  ZERO_NET_PAY:       32 records
  NET_EXCEEDS_GROSS:  22 records

Locale: Faker es_ES (Spanish employee names)
Countries: ES (Spain), BE (Belgium), DE (Germany)
Tax rates: Country-specific statutory rates
```

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

| Property | Value |
|----------|-------|
| Source | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Rows | 1,470 employees |
| Features | 35 (reduced to 33 after cleaning) |
| Target | Attrition (16.1% positive — imbalanced) |
| Licence | Public domain |
| After feature engineering | 46 features |
| Train / Test split | 80% / 20% (stratified) |

---

## Azure Setup

```
Platform:   Microsoft Azure AI Foundry
Project:    smartpayroll-ai
Region:     Sweden Central

Deployed models:
  Phi-4-mini-instruct      → Chat completion (RAG + agents)
  text-embedding-3-small   → Embeddings (1,536 dimensions, 500K TPM)

Deployment type: Global Standard (serverless, pay-per-use)
Authentication:  API key via .env (managed identity for production)
```

---

## Roadmap

- [ ] Azure Container Apps deployment (Bicep IaC)
- [ ] OpenTelemetry + Azure Monitor dashboards
- [ ] Data drift detection (PSI + KS test)
- [ ] Model performance monitoring with retraining triggers
- [ ] Azure AI Search integration (replace cosine similarity)
- [ ] LangGraph PostgreSQL checkpointing for durable state
- [ ] EU AI Act compliance documentation

---

## Author

<div align="center">

**Nazmul Farooquee**
AI & Data Science Engineer
Barcelona, Spain

[![GitHub](https://img.shields.io/badge/GitHub-Najam0786-181717?logo=github)](https://github.com/Najam0786)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/nazmulfarooquee)

*Built end-to-end in one weekend as a portfolio demonstration of Azure AI engineering capabilities.*

</div>

---

<div align="center">

**SmartPayroll AI** — Demonstrating enterprise AI engineering on Azure

*Data Pipeline • Machine Learning • RAG • Multi-Agent AI • REST API • CI/CD*

</div>