"""AgentLedger – Probabilistic Matcher Agent.

Two-stage matching pipeline:
  A. **Vector Search Retrieval** – embeds open invoices (customer_name + invoice_id)
     into an in-memory FAISS index and retrieves the top-k most relevant candidates
     for a given bank transaction description.
  B. **LLM Reasoning** – passes only the shortlisted candidates to an LLM that
     outputs a structured `ReconciliationResult` using LangChain's
     `with_structured_output`.

This design minimises hallucination risk and context-token spend by never
sending the entire AR ledger to the LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.models.schemas import ERPInvoice, ReconciliationResult

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TOP_K_CANDIDATES = 10  # max invoices returned by the vector search

SYSTEM_PROMPT = """\
You are an expert Treasury Reconciliation Agent. Your task is to match a single \
incoming Bank Transaction against a provided shortlist of Open ERP Invoices.

**Instructions & Business Rules:**

1. **Short Pays:** It is standard industry practice for intermediary banks to \
deduct wire fees. If the Bank Transaction amount is exactly $15, $20, or $25 \
less than a single invoice amount, you may declare a match and note \
"Standard Wire Fee Deducted" in your reasoning.

2. **Bulk Pays:** A customer may pay multiple invoices with a single wire. \
You must evaluate if the sum of any combination of the provided candidate \
invoices exactly equals the Bank Transaction amount (with or without standard \
wire fees).

3. **Typos:** Ignore minor formatting differences in invoice prefixes \
(e.g., "IN-" vs "INV-") if the numerical digits and customer names align.

**Output Format:**
You must output a JSON object containing:
  • match_status:  "Exact", "Partial" (for short pays), or "Unmatched"
  • confidence_score:  A float between 0.0 and 1.0.
  • matched_invoice_ids:  An array of the matched invoice IDs.
  • reasoning:  A step-by-step explanation of the math or logic used to \
determine the match.
"""

HUMAN_TEMPLATE = """\
## Bank Transaction
- **Transaction ID:** {txn_id}
- **Date:** {txn_date}
- **Amount:** ${txn_amount}
- **Description:** {txn_description}

## Candidate Open Invoices
{candidate_invoices}

Analyse the transaction against the candidates and return your reconciliation \
result.
"""


# ── A. Vector Search Retrieval ───────────────────────────────────────────────


def build_invoice_index(
    invoices: List[ERPInvoice],
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """Embed open invoices into an in-memory FAISS vector store.

    Each document's page_content is a composite string of customer_name +
    invoice_id so that similarity search can match on either dimension.
    Metadata carries the full invoice payload for later retrieval.
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    documents: list[Document] = []
    for inv in invoices:
        text = f"{inv.customer_name} {inv.invoice_id}"
        metadata = inv.model_dump()
        # Dates aren't JSON-serialisable by default in some backends
        metadata["due_date"] = str(metadata["due_date"])
        documents.append(Document(page_content=text, metadata=metadata))

    index = FAISS.from_documents(documents, embeddings)
    logger.info("Built FAISS index with %d invoice documents.", len(documents))
    return index


def retrieve_candidates(
    index: FAISS,
    bank_description: str,
    top_k: int = TOP_K_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Return the top-k most similar invoice records for a given bank description."""
    results = index.similarity_search(bank_description, k=top_k)
    candidates = [doc.metadata for doc in results]
    logger.info(
        "Retrieved %d candidate invoices for description: '%s'",
        len(candidates),
        bank_description[:80],
    )
    return candidates


# ── B. LLM Reasoning ────────────────────────────────────────────────────────


def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    """Pretty-print candidate invoices as a numbered markdown list for the prompt."""
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. **{c['invoice_id']}** | Customer: {c['customer_name']} "
            f"| Amount Due: ${c['amount_due']:.2f} "
            f"| Due Date: {c['due_date']}"
        )
    return "\n".join(lines)


def build_matching_chain(
    llm: ChatOpenAI | None = None,
) -> Any:
    """Build a LangChain chain that returns a structured `ReconciliationResult`.

    Uses `with_structured_output` so the LLM is forced to return valid JSON
    conforming to the Pydantic schema — no manual parsing needed.
    """
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_TOKEN")
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE),
        ]
    )

    # Enforce structured output matching our Pydantic model
    structured_llm = llm.with_structured_output(ReconciliationResult)
    chain = prompt | structured_llm
    return chain


# ── Public entry-point (called from graph node) ─────────────────────────────


def run_probabilistic_match(
    txn_id: str,
    txn_date: str,
    txn_amount: float,
    txn_description: str,
    invoice_index: FAISS,
    chain: Any | None = None,
) -> ReconciliationResult:
    """End-to-end probabilistic matching for a single bank transaction.

    1. Vector-retrieve the top-k candidate invoices.
    2. Pass candidates + transaction to the LLM chain.
    3. Return the structured `ReconciliationResult`.

    Parameters
    ----------
    txn_id : str
        Bank transaction identifier.
    txn_date : str
        Transaction posting date.
    txn_amount : float
        Deposit amount.
    txn_description : str
        Free-text description / memo on the bank statement.
    invoice_index : FAISS
        Pre-built vector index of open invoices.
    chain : optional
        Pre-built LangChain chain. If *None* one is created on the fly.

    Returns
    -------
    ReconciliationResult
        Structured reconciliation output with match status, confidence,
        matched invoice IDs, and step-by-step reasoning.
    """
    # ── A. Retrieve ──────────────────────────────────────────────────────
    candidates = retrieve_candidates(invoice_index, txn_description)

    if not candidates:
        logger.warning("No candidates found for txn %s – marking Unmatched.", txn_id)
        return ReconciliationResult(
            transaction_id=txn_id,
            matched_invoice_ids=[],
            match_status="Unmatched",
            confidence_score=0.0,
            reasoning="No candidate invoices were retrieved from the vector store.",
        )

    # ── B. Reason ────────────────────────────────────────────────────────
    if chain is None:
        chain = build_matching_chain()

    result: ReconciliationResult = chain.invoke(
        {
            "txn_id": txn_id,
            "txn_date": txn_date,
            "txn_amount": f"{txn_amount:.2f}",
            "txn_description": txn_description,
            "candidate_invoices": _format_candidates(candidates),
        }
    )

    # Ensure the transaction_id is set correctly (LLM may echo back a
    # slightly different value).
    result.transaction_id = txn_id

    logger.info(
        "Probabilistic match for %s → %s (confidence %.2f)",
        txn_id,
        result.match_status,
        result.confidence_score,
    )
    return result
