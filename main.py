"""AgentLedger – End-to-end execution script.

Usage:
    1. Generate mock data:   python data/generate_mock_data.py
    2. Run the pipeline:     python main.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()  # reads .env → sets GITHUB_TOKEN

import pandas as pd

from src.models.schemas import BankTransaction, ERPInvoice, MatchStatus
from src.agents.graph import app as reconciliation_graph
from src.agents.matcher_agent import build_invoice_index, build_matching_chain

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentledger.main")

# ── ANSI Colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ERP_CSV = DATA_DIR / "erp_accounts_receivable.csv"
BANK_CSV = DATA_DIR / "bank_statement.csv"
NUM_TRANSACTIONS = 5  # run the first N bank transactions


def load_erp_invoices(path: Path) -> list[ERPInvoice]:
    """Load the ERP Accounts-Receivable CSV into Pydantic models."""
    df = pd.read_csv(path)
    invoices = []
    for _, row in df.iterrows():
        invoices.append(
            ERPInvoice(
                invoice_id=row["invoice_id"],
                customer_name=row["customer_name"],
                amount_due=float(row["amount_due"]),
                due_date=row["due_date"],
            )
        )
    return invoices


def load_bank_transactions(path: Path) -> list[BankTransaction]:
    """Load the Bank Statement CSV into Pydantic models.

    Maps CSV columns → Pydantic fields:
        txn_id      → transaction_id
        description → reference_note (used as the free-text memo)
    """
    df = pd.read_csv(path)
    transactions = []
    for _, row in df.iterrows():
        transactions.append(
            BankTransaction(
                transaction_id=row["txn_id"],
                date=row["date"],
                amount=float(row["amount"]),
                reference_note=row["description"],
            )
        )
    return transactions


def status_color(status: MatchStatus) -> str:
    """Return the ANSI colour code for a given match status."""
    return {
        MatchStatus.EXACT: GREEN,
        MatchStatus.PARTIAL: YELLOW,
        MatchStatus.UNMATCHED: RED,
    }.get(status, RESET)


def status_icon(status: MatchStatus) -> str:
    """Return an icon for a given match status."""
    return {
        MatchStatus.EXACT: "✅",
        MatchStatus.PARTIAL: "⚡",
        MatchStatus.UNMATCHED: "❌",
    }.get(status, "❓")


def print_banner() -> None:
    print(f"\n{BOLD}{CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║            🏦  A G E N T   L E D G E R             ║")
    print("  ║        AI-Powered Payment Reconciliation            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")


def print_result_row(idx: int, result) -> None:
    """Print one colour-coded result row."""
    sc = status_color(result.match_status)
    icon = status_icon(result.match_status)
    invoices = ", ".join(result.matched_invoice_ids) or "—"

    print(
        f"  {BOLD}{idx}.{RESET}  "
        f"{DIM}{result.transaction_id}{RESET}  │  "
        f"{sc}{icon} {result.match_status.value:<10}{RESET}  │  "
        f"Confidence: {BOLD}{result.confidence_score:.0%}{RESET}  │  "
        f"Invoices: {BOLD}{invoices}{RESET}"
    )
    print(f"      {DIM}↳ {result.reasoning}{RESET}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print_banner()

    # 1. Load data ────────────────────────────────────────────────────────
    if not ERP_CSV.exists() or not BANK_CSV.exists():
        print(
            f"{RED}⚠  Data files not found. Run the generator first:{RESET}\n"
            f"   python data/generate_mock_data.py\n"
        )
        sys.exit(1)

    logger.info("Loading ERP invoices from %s", ERP_CSV)
    invoices = load_erp_invoices(ERP_CSV)
    logger.info("Loaded %d open invoices.", len(invoices))

    logger.info("Loading bank transactions from %s", BANK_CSV)
    transactions = load_bank_transactions(BANK_CSV)
    logger.info("Loaded %d bank transactions.", len(transactions))

    # 2. Pre-build shared resources (avoids re-embedding per txn) ────────
    logger.info("Building FAISS invoice index...")
    invoice_index = build_invoice_index(invoices)

    logger.info("Building LLM matching chain...")
    chain = build_matching_chain()

    # 3. Run the graph for each transaction ──────────────────────────────
    selected = transactions[:NUM_TRANSACTIONS]
    print(
        f"  {BOLD}Processing {len(selected)} of {len(transactions)} "
        f"bank transactions...{RESET}\n"
    )
    print(f"  {'─' * 65}\n")

    results = []
    for i, txn in enumerate(selected, 1):
        logger.info("─── Transaction %d / %d: %s ───", i, len(selected), txn.transaction_id)

        state = {
            "current_transaction": txn,
            "open_invoices": invoices,
            "match_hypotheses": [],
            "final_result": None,
            "messages": [],
            "_invoice_index": invoice_index,
            "_chain": chain,
        }

        output = reconciliation_graph.invoke(state)
        result = output["final_result"]
        results.append(result)
        print_result_row(i, result)

    # 4. Summary ─────────────────────────────────────────────────────────
    exact = sum(1 for r in results if r.match_status == MatchStatus.EXACT)
    partial = sum(1 for r in results if r.match_status == MatchStatus.PARTIAL)
    unmatched = sum(1 for r in results if r.match_status == MatchStatus.UNMATCHED)

    print(f"  {'─' * 65}")
    print(f"  {BOLD}Summary:{RESET}  "
          f"{GREEN}✅ Exact: {exact}{RESET}  │  "
          f"{YELLOW}⚡ Partial: {partial}{RESET}  │  "
          f"{RED}❌ Unmatched: {unmatched}{RESET}")
    print()


if __name__ == "__main__":
    main()
