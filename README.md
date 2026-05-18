# Agentic AutoML POC

This repository contains a Phase 1 starter setup for an Agentic AutoML Proof of Concept.

## Structure
- `backend/` FastAPI backend
- `frontend/` Streamlit frontend

## Setup
1. Create and activate virtual environment.
2. Install backend dependencies:
   - `pip install -r backend/requirements.txt`
3. Install frontend dependencies:
   - `pip install -r frontend/requirements.txt`

## Run Backend
- `uvicorn backend.main:app --reload`

## Run Frontend
- `streamlit run frontend/app.py`

## Quick Check
- Open Streamlit UI and click **Check Backend Health**.
