# TRACE
### Hypothesis-Driven Fleet Intelligence System

> A vehicle health intelligence system that inverts the traditional
> monitoring paradigm — instead of reactive alerts, TRACE starts with
> your hypothesis and works backward through vehicle telemetry to
> confirm or disprove it with sensor-level evidence.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-AI-4285F4?logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-ff6b35)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b?logo=streamlit&logoColor=white)

---

## The Idea

Most fleet intelligence systems are monitoring-first: they watch
everything and surface alerts when something crosses a threshold.
The fleet manager is reactive — waiting for the system to tell
them what is wrong.

TRACE is built on a different premise. Most fleet problems do not
start with a dashboard alert. They start with a gut feeling, a
driver's complaint, or an unexplained repair bill. The manager
already has a partial picture — they need to complete it fast.

TRACE is question-first. The manager brings a hypothesis.
TRACE parses it, runs targeted ML analysis on the relevant
vehicle subset, retrieves the specific sensor events that
support or disprove it, and returns a structured verdict.

Same data. Different mental model. Investigative rather than reactive.

---

## What It Does

- **Module 1 — Data Foundation**: Ingests NASA CMAPSS FD001 telemetry, computes Remaining Useful Life per engine, maps RUL to a 0-100 health score, and classifies each vehicle into a risk tier (HEALTHY / WATCH / WARNING / CRITICAL)
- **Module 2 — LSTM Predictor**: Two-layer LSTM trained with early stopping and engine-level train/val split, predicts RUL for each vehicle from a 30-cycle sliding window of sensor readings
- **Module 3 — Hypothesis Reasoner**: Gemini-powered query parser converts natural-language questions into structured queries; 5 analytics handlers (threshold, ranking, anomaly, comparison, trend) run targeted pandas analysis; Gemini synthesises a natural-language verdict with confidence and recommended action — with full rule-based fallback when Gemini is unavailable
- **Module 4 — Evidence Layer**: ChromaDB vector store + sentence-transformers embeddings; automatically detects 4 health event types (health drop, critical entry, sustained decline, recovery) from the timeline; RAG retrieval returns the most semantically relevant events as evidence for each verdict

---

## Architecture

```
Fleet Manager's Question
         |
         v
 Hypothesis Reasoner
 (Gemini parses question
  into structured query)
         |
         v
 +----------------------------+
 |   Analysis Engine          |
 |   LSTM Health Scores       |
 |   5 Query Handlers         |
 +----------------------------+
         |
         v
 RAG Evidence Retrieval
 (ChromaDB semantic search
  over vehicle event logs)
         |
         v
 Structured Verdict
 + Evidence Trail
         |
         v
 Streamlit Dashboard
```

---

## Dataset

**NASA CMAPSS -- Turbofan Engine Degradation Simulation**

CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
is the industry-standard public benchmark for predictive
maintenance research. It provides multivariate sensor readings
from engines across their full operational lifetime until failure --
a valid proxy for any rotating machinery degradation problem
including vehicle drivetrains and powertrains.

- **FD001**: 100 training engines, 21 sensors, one fault mode
- **Download**: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

Place the three files in `data/cmapss/`:
```
data/cmapss/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
```

---

## Getting Started

### 1. Clone and install

```bash
git clone <repo-url>
cd trace
pip install -r requirements.txt
```

### 2. Set your Gemini API key (optional -- rule-based fallback works without it)

```bash
# Windows
set GEMINI_API_KEY=your_key_here

# macOS / Linux
export GEMINI_API_KEY=your_key_here
```

### 3. Run the data pipeline

```bash
python src/data_layer/loader.py
```

### 4. Train the LSTM model

```bash
python src/ml_layer/lstm_predictor.py
```

### 5. Populate the evidence store

```bash
python src/evidence_layer/event_logger.py
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

---

## Example Queries

| Question | Query Type |
|----------|------------|
| Which vehicles will need maintenance in the next 30 cycles? | threshold_query |
| Show me the 5 most at-risk vehicles right now | ranking_query |
| Are Group A vehicles degrading faster than Group B? | fleet_comparison |
| Which vehicles show unusual health patterns? | anomaly_hunt |
| Is fleet health improving or declining over the last 50 cycles? | trend_analysis |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data ingestion | pandas, NumPy, scikit-learn |
| ML model | PyTorch (two-layer LSTM) |
| NLP query parser | Google Gemini (gemini-1.5-flash) |
| Vector store | ChromaDB + sentence-transformers |
| Dashboard | Streamlit + Plotly |
| Config | PyYAML |
| Persistence | joblib, JSON |

---

## Module Status

| Module | Status |
|--------|--------|
| Module 1 -- Data Foundation | Complete |
| Module 2 -- LSTM Predictor | Complete |
| Module 3 -- Hypothesis Reasoner | Complete |
| Module 4 -- Evidence Layer | Complete |
| Module 5 -- Dashboard and Polish | Complete |

---

## Running Tests

```bash
pytest tests/ -v
```

Integration test covers: full pipeline, predictor loading, ChromaDB population,
all 5 query types returning complete verdicts, evidence event retrieval, and
graceful unknown-query handling.

---

## Author

**Siddharth**
[GitHub](https://github.com/Siddharth-G06)

Built as a demonstration of hypothesis-driven fleet intelligence --
the investigative alternative to threshold-based monitoring.