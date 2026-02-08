from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import get_settings
from .models import CorepResult, RetrievedDoc
from .validation import validate_corep_result

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


def _build_seed_corpus():
    docs = []

    docs.append(
        Document(
            page_content=dedent(
                """
                Under the PRA's implementation of the CRR, institutions must maintain
                a minimum Common Equity Tier 1 (CET1) capital ratio of at least 4.5%
                of their total risk-weighted exposure amount (RWA). CET1 capital is
                composed primarily of common shares, share premium, retained earnings,
                accumulated other comprehensive income and certain regulatory
                adjustments.
                """
            ).strip(),
            metadata={
                "source": "PRA Rulebook (synthetic excerpt)",
                "citation": "CRR Art. 92(1)(a) – minimum CET1 capital ratio (illustrative)",
            },
        )
    )

    docs.append(
        Document(
            page_content=dedent(
                """
                Additional Tier 1 (AT1) instruments are perpetual subordinated
                instruments that meet the relevant eligibility criteria. Tier 2
                capital consists of subordinated instruments with limited
                maturity and certain loan loss provisions. Total capital is the
                sum of Tier 1 capital (CET1 + AT1) and Tier 2 capital.
                """
            ).strip(),
            metadata={
                "source": "PRA Rulebook (synthetic excerpt)",
                "citation": "CRR Part Two – Own Funds (illustrative)",
            },
        )
    )

    docs.append(
        Document(
            page_content=dedent(
                """
                Risk-weighted assets (RWA) represent the total of exposure values
                multiplied by applicable risk weights under the standardised or
                internal ratings based approaches. Capital ratios are expressed as
                capital amounts divided by total RWA. Institutions must monitor
                their CET1, Tier 1 and total capital ratios against minimum and
                buffer requirements at all times.
                """
            ).strip(),
            metadata={
                "source": "PRA Rulebook (synthetic excerpt)",
                "citation": "CRR Part Three – Capital Requirements (illustrative)",
            },
        )
    )

    docs.append(
        Document(
            page_content=dedent(
                """
                The COREP Own Funds templates require firms to report CET1, AT1,
                Tier 2 capital and total risk exposure amount in the relevant
                reporting currency. Ratios should typically be reported to at
                least four decimal places when expressed as decimals.
                """
            ).strip(),
            metadata={
                "source": "COREP Implementing Technical Standards (synthetic excerpt)",
                "citation": "ITS on Supervisory Reporting – COREP Own Funds (illustrative)",
            },
        )
    )

    return docs


def _get_vector_store():
    settings = get_settings()
    
    if HuggingFaceEmbeddings is not None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif settings.openai_api_key:
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    else:
        raise RuntimeError(
            "No embedding model available. Please install sentence-transformers: "
            "pip install sentence-transformers langchain-huggingface"
        )

    vector_store = Chroma(
        collection_name="corep_own_funds_rules",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_directory,
    )

    try:
        count = vector_store._collection.count()
    except:
        count = 0

    if count == 0:
        docs = _build_seed_corpus()
        vector_store.add_documents(docs)
        vector_store.persist()

    return vector_store


def _build_prompt():
    system_prompt = dedent(
        """
        You are an expert UK PRA / COREP regulatory reporting analyst focusing on
        capital requirements and Own Funds templates.

        You will be given:
        - A natural-language banking scenario.
        - Extracts from PRA Rulebook / COREP Own Funds instructions (synthetic).

        Your task is to propose a *STRICT* structured JSON object capturing the
        firm's COREP Own Funds position. You must:

        1. Use the regulatory context to interpret the scenario conservatively.
        2. Only infer amounts when they are clearly implied; otherwise treat them
           as missing.
        3. Never hallucinate regulatory references – only cite rules explicitly
           supported by the provided context.

        OUTPUT FORMAT (JSON ONLY, NO PROSE OUTSIDE JSON):
        {{
          "template_name": "C 01.00 - Own Funds (illustrative)",
          "CET1": <number or null>,
          "AT1": <number or null>,
          "Tier2": <number or null>,
          "RWA": <number or null>,
          "CET1_ratio": <number or null>,   // decimal, e.g. 0.125 for 12.5
          "missing_fields": [<list of missing numeric fields>],
          "validation_warnings": [<list of checks the *model* thinks might fail>],
          "rules_used": [<list of strings with rule / paragraph references>],
          "explanation": "<short human-readable explanation>"
        }}

        - Use numbers, not strings with units.
        - If you cannot determine a numeric field, set it to null and include
          the field name in missing_fields.
        - For CET1_ratio, you may propose a value, but it will be recomputed by
          deterministic validation.
        - Respond with **valid JSON only**, with double quotes, and no comments.
        """
    ).strip()

    human_prompt = dedent(
        """
        Banking scenario:
        -----------------
        {scenario}

        Relevant regulatory context:
        ----------------------------
        {context}

        Now produce the JSON object as specified, with no additional commentary.
        """
    ).strip()

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])


def _call_llm(scenario: str, context_text: str):
    settings = get_settings()

    if settings.llm_provider == "groq":
        if ChatGroq is None:
            raise RuntimeError(
                "langchain-groq is not installed. Please install it with: "
                "pip install langchain-groq"
            )
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )
    else:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )

    prompt = _build_prompt()
    chain = prompt | llm

    response = chain.invoke({"scenario": scenario, "context": context_text})
    content = response.content

    def _parse_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                return json.loads(snippet)
            raise

    return _parse_json(content)


def generate_corep_assessment(scenario: str, top_k: int = 4):
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieved = retriever.invoke(scenario)

    retrieved_docs = []
    context_chunks = []

    for idx, doc in enumerate(retrieved):
        metadata = doc.metadata or {}
        source = str(metadata.get("source", "unknown"))
        citation = str(metadata.get("citation", f"synthetic-{idx}"))
        text = str(doc.page_content)

        retrieved_docs.append(
            RetrievedDoc(
                id=str(idx),
                source=source,
                citation=citation,
                text=text,
            )
        )

        context_chunks.append(f"[{citation}] ({source})\n{text}")

    context_text = "\n\n---\n\n".join(context_chunks)

    raw_model_output = _call_llm(scenario=scenario, context_text=context_text)
    corep_result = validate_corep_result(raw_model_output)

    citations_from_docs = {doc.citation for doc in retrieved_docs}
    for citation in sorted(citations_from_docs):
        if citation not in corep_result.rules_used:
            corep_result.rules_used.append(citation)

    return corep_result, retrieved_docs, raw_model_output
