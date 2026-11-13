"""
Lancement local :
    uvicorn main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import PROJECT_ROOT
from logging_setup import setup_logging

setup_logging() 

from agent.graph import ask  

logger = logging.getLogger("graph")

app = FastAPI(title="Pharma RAG agent")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    details: dict  


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    
    try:
        return AskResponse(**ask(request.question))
    except Exception as e:
        logger.exception("Erreur pendant le traitement de la question")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")



FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")