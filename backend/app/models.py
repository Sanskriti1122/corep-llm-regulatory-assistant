from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):

    scenario: str = Field(
        ...,
        description="Natural-language description of the bank's capital position and scenario.",
        min_length=5,
    )


class RetrievedDoc(BaseModel):

    id: str = Field(..., description="Identifier within the vector store.")
    source: str = Field(..., description="Logical source of the text (e.g. 'PRA Rulebook').")
    citation: str = Field(
        ..., description="Textual citation, e.g. rule or paragraph reference."
    )
    text: str = Field(..., description="Content of the retrieved regulatory snippet.")


class CorepResult(BaseModel):

    template_name: str = Field(
        ...,
        description="Name or code of the COREP template, e.g. C 01.00.",
    )
    CET1: Optional[float] = Field(
        None, description="Common Equity Tier 1 capital amount."
    )
    AT1: Optional[float] = Field(None, description="Additional Tier 1 capital amount.")
    Tier2: Optional[float] = Field(None, description="Tier 2 capital amount.")
    RWA: Optional[float] = Field(
        None, description="Risk-weighted assets used to compute capital ratios."
    )
    CET1_ratio: Optional[float] = Field(
        None,
        description="CET1 capital ratio computed as CET1 / RWA (in decimal, e.g. 0.12 for 12%).",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of numeric fields (CET1, AT1, Tier2, RWA) that are missing or could not be inferred.",
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Human-readable warnings produced by deterministic Python validation.",
    )
    rules_used: List[str] = Field(
        default_factory=list,
        description="List of regulatory paragraphs or rules cited for this assessment.",
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of how the numbers and conclusions were derived.",
    )


class AnalyzeResponse(BaseModel):

    scenario: str = Field(..., description="Original user-provided scenario text.")
    retrieved_context: List[RetrievedDoc] = Field(
        default_factory=list,
        description="Regulatory text snippets retrieved via RAG and provided to the LLM.",
    )
    corep_result: CorepResult = Field(
        ..., description="Structured and validated COREP Own Funds output."
    )
    raw_model_output: Optional[dict] = Field(
        default=None,
        description=(
            "Raw JSON structure as returned by the LLM prior to deterministic validation. "
            "Useful for debugging and audit."
        ),
    )
