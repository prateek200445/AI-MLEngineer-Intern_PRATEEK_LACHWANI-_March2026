# Minerva Catalog RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) assistant for Minerva course-catalog use cases.

It supports:
- Eligibility and prerequisite checking
- General catalog and policy Q&A
- Academic term planning

## Repository

- GitHub: https://github.com/prateek200445/AI-MLEngineer-Intern_PRATEEK_LACHWANI-_March2026

## Tech Stack

- Backend: Python, FastAPI, Uvicorn
- Retrieval: Qdrant vector database
- Embeddings: `BAAI/bge-base-en-v1.5`
- LLM providers supported: OpenRouter, Google, Ollama (OpenAI path in config)
- Frontend: React + TypeScript

## Project Structure

- `collections/` backend API, ingestion, rule engine
- `frontend/` web UI
- `eval/` evaluation scripts/results

## Prerequisites

- Python 3.10+
- Node.js 18+
- Qdrant running on `localhost:6333`

## Setup

1. Clone repository

```bash
git clone https://github.com/prateek200445/AI-MLEngineer-Intern_PRATEEK_LACHWANI-_March2026.git
cd AI-MLEngineer-Intern_PRATEEK_LACHWANI-_March2026/collections
```

2. Create Python virtual environment and install backend dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment

- Copy `.env.example` to `.env`
- Fill your API keys (`OPENROUTER_API_KEY` and/or `GOOGLE_API_KEY`)

4. Install frontend dependencies

```bash
cd ..\frontend
npm install
```

## Run

### 1) Start backend

```bash
cd ..\collections
.venv\Scripts\activate
uvicorn api:api --host 0.0.0.0 --port 8000 --reload --app-dir d:\qdrant_ping\collections
```

### 2) Start frontend

```bash
cd ..\frontend
npm run dev
```

Frontend connects to backend via `VITE_API_URL` (default `http://localhost:8000`).

## Ingestion (Build Index + Rules)

Place catalog PDFs in `collections/policy_docs/`, then run:

```bash
cd ..\collections
.venv\Scripts\activate
python ingest.py
```

This updates:
- Qdrant vector collection (`policy_db`)
- `catalog_rules.json`

## API Endpoints

- `GET /` health check
- `POST /query` main eligibility + general catalog Q&A endpoint
- `POST /extract-prerequisite` structured prerequisite extraction
- `POST /plan-term` deterministic term planner

## Evaluation

```bash
cd ..\eval
python check_25_cases.py
```

Output is saved to `eval/check_25_results.json`.

## Security Notes

- Do not commit `.env`
- Use `.env.example` as template
- Keep API keys private
