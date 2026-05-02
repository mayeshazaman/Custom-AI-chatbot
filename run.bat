@echo off
start cmd /k "cd /d G:\custom-ai-chatbot && venv\Scripts\activate && uvicorn app:app --reload --port 8000"
start cmd /k "cd /d G:\custom-ai-chatbot\frontend && npm run dev"