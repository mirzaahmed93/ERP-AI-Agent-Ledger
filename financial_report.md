# Agent Ledger: Modernising Financial Reconciliation with AI Agents
**An Architectural Walkthrough**

---

## i) Executive Summary
As digital payment volumes surge, finance departments remain bottlenecked by legacy Enterprise Resource Planning (ERP) systems that lack the intelligence to automate complex reconciliation tasks. **AgentLedger** is a proof-of-concept AI application designed to bridge this gap. By orchestrating deterministic logic with Large Language Model (LLM) reasoning via LangGraph and Streamlit, AgentLedger automates the tedious matching of bank statements to open accounts receivable, cleanly handling edge cases like short-pays and missing references that traditionally require human intervention.

## ii) Problem Statement: ERP Reconciliation Challenges (2026)
Despite continuous digital transformation efforts, financial reconciliation remains a highly manual, error-prone process. Recent industry analyses highlight several critical challenges with traditional ERP systems:

*   **The Burden of Manual Resolution**: Despite the adoption of modern ERPs, reconciliation remains heavily manual. **PwC** research indicates that finance teams still spend approximately 30% of their time on manual reconciliation, with 56% of CFOs relying on spreadsheets for these tasks (PwC, 2025). Similarly, an **EY** survey found that up to 59% of a financial department's work involves manual reconciliation tasks (EY,2025). 
*   **Data Silos and Fragmentation**: Disconnected systems across multinational corporations result in data silos. **Deloitte** highlights that 54% of companies lack sufficient visibility for intercompany reconciliation due to fragmented systems and poor integration between ERPs and external banking platforms (Deloitte, 2025). *This depicts an issue with having a standardised golden record set present within companies*.
*   **Ineffective ERP Modernization**: Organisations often fail to realize the benefits of new ERP implementations. **McKinsey & Company** reports a 70% failure rate for ERP modernization projects when businesses simply transfer outdated, broken manual processes to new systems rather than redesigning them (McKinsey & Company, 2025).
*   **The AI Imperative**: To combat these bottlenecks, financial institutions are turning to Generative AI. **McKinsey** projects that GenAI could add between $200 billion USD and $340 billion USD in annual value to the global banking sector by 2025, largely achieved by automating complex middle office tasks like reconciliation and exception handling (McKinsey & Company, 2025).

## iii) The AgentLedger Solution (Proposed Fix)
AgentLedger tackles these challenges by introducing an intelligent, middle-tier matching engine. Instead of replacing the ERP, AgentLedger acts as a highly capable assistant that pre-processes incoming bank feeds before they are committed to the ledger.

### Core Architecture and Library Requirements
The system is built on a modern, open-source Python data stack, ensuring it remains *100% free-tier capable leveraging localised models* where possible:

*   **LangGraph (`langgraph`)**: Orchestrates the state machine, defining the workflow between deterministic matching, LLM reasoning, and `human-in-the-loop` review.
*   **LangChain (`langchain-openai`, `langchain-huggingface`)**: Provides the framework to interact with LLMs. The GitHub Models API are utilised here (`gpt-4o-mini`) for cost-effective reasoning and local HuggingFace embeddings (`all-MiniLM-L6-v2`) to preserve API rate limits.
*   **FAISS (`faiss-cpu`)**: Refers here to Facebook AI Similarity Search and is an open-source library from Meta for efficient similarity search and clustering of dense vectors. It powers the high-speed, in-memory vector database. It embeds open ERP invoices, allowing the system to rapidly retrieve relevant candidate invoices based on messy bank descriptions.
*   **Pydantic (`pydantic`)**: Enforces strict runtime data validation, ensuring the LLM's outputs conform to perfectly structured JSON schemas before hitting downstream financial systems.
*   **Streamlit (`streamlit`)**: Delivers a rich, interactive frontend GUI (grpahical user interface), allowing financial stakeholders to monitor the AI's real-time decisions and review its reasoning for its matching decisions.
*   **Pandas (`pandas`)**: Handles the initial ingestion and manipulation of the CSV data extracts.

## iv) System Walkthrough & Workflow
The AgentLedger pipeline operates as a multi-stage funnel, optimising for both speed and accuracy.

### Stage 1: Ingestion & Vectorization
1. **Data Load**: The system ingests the internal `erp_accounts_receivable.csv` (Open Invoices) and the external `bank_statement.csv` (Incoming Cash).
2. **Knowledge Retrieval Setup**: To avoid overwhelming the LLM with thousands of invoices (and bankrupting the API quota), AgentLedger converts all open invoices into vector embeddings using a local HuggingFace model and stores them in FAISS.

### Stage 2: The Fast-Path (Deterministic Matching)
When a bank transaction arrives, AgentLedger first attempts a purely heuristic matching. 
*   *Example*: If the bank description contains exactly "INV-2679" and the payment matches the balance of $1100.04 perfectly, it is instantly marked as an **Exact Match**. This bypasses AI entirely, *saving both time and money*.

### Stage 3: The AI-Path (Probabilistic Reasoning)
*If the deterministic check fails (e.g., due to a short-pay), the transaction enters the LLM Node*.
*   **Retrieval**: FAISS searches the vector space for the top 10 most contextually similar invoices based on the bank memo (e.g., finding the right customer despite misspelled names) similar to a semantic fuzzy matching check.
*   **Reasoning**: These candidates are passed to `gpt-4o-mini` with a strict prompt defining business rules (like standard wire fees).
*   *Real-World Example*: In this demo pipeline, `TXN-10001` arrived as a payment for $1867.84. The closest invoice was $1892.84. The deterministic logic failed. However, the LLM agent reasoned: *"The difference is exactly $25. This matches the standard wire fee deduction criteria."* It successfully linked the payment, grading it a **Partial Match** with 90% confidence.

### Stage 4: Human-in-the-Loop & The Streamlit GUI
For unresolvable transactions (0% confidence), the system halts and escalates. The Streamlit dashboard visualizes this entire flow. Stakeholders can watch transactions turn Green (Exact), Yellow (Partial - AI Reasoned), or Red (Failed - Human Review Required). *By clicking on a Yellow transaction, the user can read the exact mathematical justification the LLM used to make the match, ensuring full auditability*.

## v) Conclusion
AgentLedger demonstrates how augmenting traditional ERP systems with targeted AI agents can drastically reduce manual reconciliation workloads. By blending lightning fast vector search with LLM reasoning, finance teams can resolve complex short-pays and messy data in real-time, achieving higher accuracy and freeing analysts to focus on truly strategic anomalies.

---

## vi) References
**PwC** (2025). *Finance Transformation and the Burden of Manual Reconciliation in Modern Workflows.* 

**EY** (2025). *Streamlining the Financial Close: The Cost of Manual Account Reconciliation.*

**Deloitte** (2025). *Intercompany Accounting and Reconciliation: Overcoming Fragmentation and Visibility Gaps.*

**McKinsey & Company** (2025). *Why ERP Implementations Fail: The Risk of Transferring Outdated Processes.*

**McKinsey & Company** (2025). *The Economic Potential of Generative AI: The Next Productivity Frontier in Banking.*
