import os
import time
from pathlib import Path

# Apple Silicon FAISS workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.models.schemas import MatchStatus
from src.agents.graph import app as reconciliation_graph
from src.agents.matcher_agent import build_invoice_index, build_matching_chain
from main import load_erp_invoices, load_bank_transactions

# Load env variables (for GITHUB_TOKEN)
load_dotenv()

# --- Config & Paths ---
DATA_DIR = Path("data")
ERP_CSV = DATA_DIR / "erp_accounts_receivable.csv"
BANK_CSV = DATA_DIR / "bank_statement.csv"

st.set_page_config(
    page_title="AgentLedger Dashboard",
    page_icon="🏦",
    layout="wide",
)

# --- Define Caching Functions ---
def get_data():
    invoices = load_erp_invoices(ERP_CSV)
    transactions = load_bank_transactions(BANK_CSV)
    return invoices, transactions

@st.cache_resource
def get_ml_components(_invoices):
    """Caches the FAISS index and LangChain setup so it doesn't rebuild every click."""
    index = build_invoice_index(_invoices)
    chain = build_matching_chain()
    return index, chain

def status_color(status: MatchStatus) -> str:
    if status == MatchStatus.EXACT:
        return "green"
    elif status == MatchStatus.PARTIAL:
        return "orange"
    else:
        return "red"

def status_icon(status: MatchStatus) -> str:
    if status == MatchStatus.EXACT:
        return "✅"
    elif status == MatchStatus.PARTIAL:
        return "⚡"
    else:
        return "❌"

# --- Main Application UI ---
st.title(" AI AgentLedger Dashboard")
st.markdown("**AI-Powered Payment Reconciliation System**")
st.markdown("Automates standard matching and intelligently reasons about short-pays/wire fees using GitHub Models (GPT-4o-mini).")

# Check data existence
if not ERP_CSV.exists() or not BANK_CSV.exists():
    st.error("Data files not found. Please run `python data/generate_mock_data.py` first.")
    st.stop()

# Load Data
with st.spinner("Loading ledger data..."):
    invoices, transactions = get_data()

# Load Models
with st.spinner("Waking up AI agents and building FAISS vector index..."):
    invoice_index, chain = get_ml_components(invoices)

# --- Sidebar ---
with st.sidebar:
    st.header("Control Panel")
    st.metric(label="Open ERP Invoices", value=len(invoices))
    st.metric(label="Pending Bank Txns", value=len(transactions))
    
    st.divider()
    num_to_process = st.slider("Transactions to Process", min_value=1, max_value=len(transactions), value=5)
    
    run_button = st.button("Run Reconciliation", type="primary", use_container_width=True)

# --- Dashboard Layout ---
tab1, tab2 = st.tabs(["Reconciliation Hub", "Raw Data View"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ERP Invoices")
        st.dataframe(pd.read_csv(ERP_CSV))
    with col2:
        st.subheader("Bank Transactions")
        st.dataframe(pd.read_csv(BANK_CSV))

with tab1:
    if not run_button:
        st.info("👈 Select how many transactions to process in the sidebar and hit **Run Reconciliation**.")
    else:
        selected_txns = transactions[:num_to_process]
        
        # Placeholders for live metrics
        metrics_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        st.subheader("Live Processing Feed")
        
        # Process transactions one by one
        for i, txn in enumerate(selected_txns, 1):
            status_text.text(f"Processing transaction {i} of {num_to_process}: {txn.transaction_id}")
            
            state = {
                "current_transaction": txn,
                "open_invoices": invoices,
                "match_hypotheses": [],
                "final_result": None,
                "messages": [],
                "_invoice_index": invoice_index,
                "_chain": chain,
            }
            
            start_time = time.time()
            output = reconciliation_graph.invoke(state)
            duration = time.time() - start_time
            
            result = output["final_result"]
            results.append(result)
            
            # Draw an expander for this transaction
            with st.expander(f"{status_icon(result.match_status)} **{txn.transaction_id}** — {result.match_status.value} (Confidence: {result.confidence_score:.0%})", expanded=False):
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Bank Transaction:**")
                    st.write(f"- Amount: **${txn.amount:.2f}**")
                    st.write(f"- Note: `{txn.reference_note}`")
                with colB:
                    st.markdown("**Matched Invoices:**")
                    if result.matched_invoice_ids:
                        for inv_id in result.matched_invoice_ids:
                            st.write(f"- `{inv_id}`")
                    else:
                        st.write("`- None -`")
                
                st.markdown("---")
                st.markdown("**Agent Reasoning:**")
                st.info(result.reasoning.replace("$", r"\$"))
                st.caption(f"Processed in {duration:.2f}s")
                
            progress_bar.progress(i / num_to_process)
            
        status_text.text("Reconciliation Complete!")
        
        # Compute final metrics
        exact_count = sum(1 for r in results if r.match_status == MatchStatus.EXACT)
        partial_count = sum(1 for r in results if r.match_status == MatchStatus.PARTIAL)
        unmatched_count = sum(1 for r in results if r.match_status == MatchStatus.UNMATCHED)
        
        with metrics_container:
            st.subheader("Run Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("✅ Exact Matches", exact_count)
            m2.metric("⚡ Partial Matches (LLM)", partial_count)
            m3.metric("❌ Unmatched (Human Review)", unmatched_count)
