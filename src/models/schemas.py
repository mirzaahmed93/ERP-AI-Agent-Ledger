"""AgentLedger – Pydantic data models for reconciliation pipeline."""

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MatchStatus(str, Enum):
    """Possible reconciliation outcomes."""

    EXACT = "Exact"
    PARTIAL = "Partial"
    UNMATCHED = "Unmatched"


# ---------------------------------------------------------------------------
# ERP Invoice (Accounts Receivable record)
# ---------------------------------------------------------------------------

class ERPInvoice(BaseModel):
    """Represents a single open invoice from the ERP / Accounts Receivable."""

    invoice_id: str = Field(..., description="Unique invoice identifier (e.g. INV-20240101)")
    customer_name: str = Field(..., description="Name of the paying customer")
    amount_due: float = Field(..., ge=0, description="Outstanding amount on the invoice")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    due_date: datetime.date = Field(..., description="Payment due date")
    reference_note: Optional[str] = Field(
        default=None,
        description="Optional PO number or reference the customer should quote",
    )


# ---------------------------------------------------------------------------
# Bank Transaction (Bank Statement line)
# ---------------------------------------------------------------------------

class BankTransaction(BaseModel):
    """Represents one line item from a bank statement."""

    transaction_id: str = Field(..., description="Bank-assigned transaction ID")
    date: datetime.date = Field(..., description="Value / posting date")
    amount: float = Field(..., description="Deposit amount (positive = credit)")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    payer_name: Optional[str] = Field(
        default=None,
        description="Remitter / payer name as reported by the bank",
    )
    reference_note: Optional[str] = Field(
        default=None,
        description="Free-text reference / memo field on the bank statement",
    )


# ---------------------------------------------------------------------------
# Reconciliation Result
# ---------------------------------------------------------------------------

class ReconciliationResult(BaseModel):
    """The output of the matching pipeline for a single bank transaction."""

    transaction_id: str = Field(
        ..., description="ID of the bank transaction that was evaluated"
    )
    matched_invoice_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Invoice IDs matched to this transaction. "
            "Multiple entries indicate a bulk-payment scenario."
        ),
    )
    match_status: MatchStatus = Field(
        ..., description="Overall match quality"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the proposed match (0–1)",
    )
    amount_matched: float = Field(
        default=0.0,
        description="Total invoice amount covered by this transaction",
    )
    amount_discrepancy: float = Field(
        default=0.0,
        description=(
            "Difference between transaction amount and matched invoice total. "
            "Positive = overpayment, negative = short pay / wire fee deduction."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of why this match was proposed",
    )
