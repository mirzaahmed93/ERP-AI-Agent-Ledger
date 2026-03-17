"""AgentLedger – LangGraph reconciliation workflow.

Three-pass pipeline:
  1. Deterministic Match  – fast Python exact-match (no LLM cost)
  2. Probabilistic Match  – FAISS retrieval → LLM reasoning
  3. Human Review         – terminal-based approval for low-confidence results
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.models.schemas import (
    BankTransaction,
    ERPInvoice,
    MatchStatus,
    ReconciliationResult,
)
from src.agents.matcher_agent import (
    build_invoice_index,
    build_matching_chain,
    run_probabilistic_match,
)

logger = logging.getLogger(__name__)

# ── ANSI colour helpers (for terminal output) ────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------
# 1. Define the Graph State
# ---------------------------------------------------------
class ReconciliationState(TypedDict):
    current_transaction: BankTransaction
    open_invoices: List[ERPInvoice]
    match_hypotheses: List[Dict[str, Any]]
    final_result: Optional[ReconciliationResult]
    messages: List[Any]
    # Pre-built resources passed in to avoid re-creation per txn
    _invoice_index: Any  # FAISS index
    _chain: Any  # LangChain chain


# ---------------------------------------------------------
# 2. Node Implementations
# ---------------------------------------------------------
def deterministic_match_node(state: ReconciliationState) -> Dict:
    """Pass 1: Fast, cheap Python logic for exact 1:1 matches.

    Checks whether *any* open invoice satisfies BOTH conditions:
      • amount_due == transaction amount  (exact penny match)
      • invoice_id appears exactly as in the bank description
    """
    txn: BankTransaction = state["current_transaction"]
    invoices: List[ERPInvoice] = state["open_invoices"]

    # The bank description is stored in reference_note (mapped from CSV)
    txn_description = txn.reference_note or ""

    for inv in invoices:
        amount_match = abs(inv.amount_due - txn.amount) < 0.005  # float tolerance
        id_match = inv.invoice_id in txn_description

        if amount_match and id_match:
            result = ReconciliationResult(
                transaction_id=txn.transaction_id,
                matched_invoice_ids=[inv.invoice_id],
                match_status=MatchStatus.EXACT,
                confidence_score=1.0,
                amount_matched=inv.amount_due,
                amount_discrepancy=round(txn.amount - inv.amount_due, 2),
                reasoning=(
                    f"Deterministic exact match: invoice {inv.invoice_id} "
                    f"(${inv.amount_due:.2f}) found verbatim in bank description "
                    f"and amount matches the transaction (${txn.amount:.2f})."
                ),
            )
            logger.info(
                "%s✓ Deterministic match for %s → %s%s",
                GREEN, txn.transaction_id, inv.invoice_id, RESET,
            )
            return {"final_result": result}

    logger.info(
        "  No deterministic match for %s – forwarding to LLM.", txn.transaction_id
    )
    return {"final_result": None}


def probabilistic_match_node(state: ReconciliationState) -> Dict:
    """Pass 2: LLM agent logic for bulk pays, short pays, and typos."""
    txn: BankTransaction = state["current_transaction"]
    invoice_index = state.get("_invoice_index")
    chain = state.get("_chain")

    txn_description = txn.reference_note or ""

    import time
    time.sleep(4)  # Rate limiting for free-tier GitHub Models API (15 RPM)
    
    result = run_probabilistic_match(
        txn_id=txn.transaction_id,
        txn_date=str(txn.date),
        txn_amount=txn.amount,
        txn_description=txn_description,
        invoice_index=invoice_index,
        chain=chain,
    )

    return {"final_result": result}


def human_review_node(state: ReconciliationState) -> Dict:
    """Pass 3: Terminal-based Human-in-the-Loop approval for low-confidence matches."""
    txn: BankTransaction = state["current_transaction"]
    result: ReconciliationResult = state["final_result"]

    print(f"\n{'=' * 60}")
    print(f"{BOLD}{YELLOW}⚠  HUMAN REVIEW REQUIRED{RESET}")
    print(f"{'=' * 60}")
    print(f"  {BOLD}Transaction:{RESET}  {txn.transaction_id}")
    print(f"  {BOLD}Amount:{RESET}       ${txn.amount:.2f}")
    print(f"  {BOLD}Description:{RESET}  {txn.reference_note or 'N/A'}")
    print(f"  {BOLD}Date:{RESET}         {txn.date}")
    print(f"{'─' * 60}")
    print(f"  {BOLD}LLM Proposed Match:{RESET}")
    print(f"    Status:     {result.match_status.value}")
    print(f"    Invoices:   {', '.join(result.matched_invoice_ids) or 'None'}")
    print(f"    Confidence: {result.confidence_score:.0%}")
    print(f"    Reasoning:  {result.reasoning}")
    print(f"{'─' * 60}")

    print(f"  {RED}⚠ Escalated to human reviewer.{RESET}\n")
    
    result = ReconciliationResult(
        transaction_id=txn.transaction_id,
        matched_invoice_ids=[],
        match_status=MatchStatus.UNMATCHED,
        confidence_score=0.0,
        amount_matched=0.0,
        amount_discrepancy=txn.amount,
        reasoning="Match escalated for human review.",
    )

    return {"final_result": result}


# ---------------------------------------------------------
# 3. Routing Logic
# ---------------------------------------------------------
def route_after_deterministic(state: ReconciliationState) -> str:
    """If we found a perfect 1:1 match, stop. Otherwise, call the LLM."""
    result = state.get("final_result")
    if result and result.match_status == MatchStatus.EXACT:
        return END
    return "probabilistic_match"


def route_after_probabilistic(state: ReconciliationState) -> str:
    """If the LLM is confident (≥ 85%), approve it. Otherwise, flag for human."""
    result = state.get("final_result")
    if result and result.confidence_score >= 0.85:
        return END
    return "human_review"


# ---------------------------------------------------------
# 4. Build and Compile the Graph
# ---------------------------------------------------------
workflow = StateGraph(ReconciliationState)

workflow.add_node("deterministic_match", deterministic_match_node)
workflow.add_node("probabilistic_match", probabilistic_match_node)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("deterministic_match")

workflow.add_conditional_edges("deterministic_match", route_after_deterministic)
workflow.add_conditional_edges("probabilistic_match", route_after_probabilistic)
workflow.add_edge("human_review", END)

app = workflow.compile()
