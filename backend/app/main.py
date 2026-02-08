from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models import AnalyzeResponse, ScenarioRequest
from .rag import generate_corep_assessment

logger = logging.getLogger(__name__)


def create_app():
    app = FastAPI(
        title="PRA COREP Own Funds LLM Assistant",
        description=(
            "Prototype LLM-assisted PRA COREP Own Funds reporting assistant, "
            "combining RAG over synthetic regulatory text with deterministic validation."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health():
        settings = get_settings()
        if settings.llm_provider == "groq":
            model_name = settings.groq_model
        else:
            model_name = settings.openai_model
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "model": model_name,
        }

    @app.post("/api/analyze_scenario", response_model=AnalyzeResponse, tags=["corep"])
    def analyze_scenario(payload: ScenarioRequest):
        try:
            corep_result, retrieved_docs, raw_model_output = generate_corep_assessment(
                scenario=payload.scenario
            )

            return AnalyzeResponse(
                scenario=payload.scenario,
                retrieved_context=retrieved_docs,
                corep_result=corep_result,
                raw_model_output=raw_model_output,
            )
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logger.error(f"Error in analyze_scenario: {error_msg}\n{error_trace}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {error_msg}. Check server logs for details."
            )

    return app


app = create_app()
