# CrimeVision AI: Multimodal Forensic Analysis Platform

React frontend + FastAPI backend. Analyzes crime-scene images, video (sampled frames) and witness audio
(Whisper transcription) with Claude, builds an event timeline, classifies the incident, answers questions
about the case, and exports a PDF forensic report. Evidence files and results are stored in `backend/data/`.

## Run locally
1. Install Python 3.10+ and get an Anthropic API key.
2. `cd backend && python -m venv venv && venv\Scripts\activate` (Mac/Linux: `source venv/bin/activate`)
3. `pip install -r requirements.txt`
4. Set the key: `set ANTHROPIC_API_KEY=your_key` (Mac/Linux: `export ANTHROPIC_API_KEY=your_key`)
5. `uvicorn main:app --reload`, then open http://localhost:8000

API docs: http://localhost:8000/docs

## Endpoints
`POST /analyze/image|video|audio` · `POST /chat` · `GET /report/{case_id}`
