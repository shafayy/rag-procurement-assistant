#!/usr/bin/env python3
"""
main.py — CLI for the RAG Procurement Assistant.

Commands
--------
  python main.py ingest          # index everything in /docs
  python main.py chat            # interactive Q&A session
  python main.py ask "question"  # single one-shot question
  python main.py demo            # run a self-contained demo (no docs needed)
"""

import sys
import os
from rag import RAGAssistant


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — injects synthetic procurement docs so the project works
# immediately without any user-provided documents
# ─────────────────────────────────────────────────────────────────────────────

DEMO_DOCS = {
    "vendor_policy.txt": """
VENDOR MANAGEMENT POLICY — Rafed UAE (PureHealth Group)

All vendors supplying medical consumables must hold a valid ISO 13485 certification.
Vendor onboarding requires submission of: trade licence, VAT registration certificate,
product catalogue, and a minimum of two reference contracts from healthcare clients.

Preferred vendors are reviewed quarterly by the Procurement Committee. Any vendor
scoring below 70% on the Vendor Scorecard (covering delivery reliability, invoice
accuracy, and quality compliance) is placed on a Performance Improvement Plan (PIP).
Vendors on PIP for two consecutive quarters are suspended from the Approved Vendor List.

New vendor requests must be submitted via the Fairmarkit sourcing portal and approved
by the Category Manager and the Head of Procurement.
""",
    "item_master_guidelines.txt": """
ITEM MASTER DATA GOVERNANCE GUIDELINES

The Item Master is the single source of truth for all healthcare SKUs procured across
the PureHealth Group, including SEHA hospitals and SSMC.

Naming Convention:
  [Category] - [Brand/Generic Name] - [Specification] - [UOM]
  Example: CONSUMABLE - GLOVES NITRILE POWDER FREE - SIZE M - BOX/100

Duplicate Detection:
  Items sharing the same UNSPSC code, unit of measure, and specification string
  (normalised) are flagged as potential duplicates and routed to the Data Governance
  team for review. Automated deduplication scripts run every Sunday at 02:00 GST.

Classification Standards:
  All items must be mapped to a Level-4 UNSPSC code. Items without a valid UNSPSC
  code are quarantined and cannot be raised on a Purchase Order until resolved.

Data Owners:
  - Category Managers own their category classifications.
  - The Procurement Analytics team owns the deduplication pipeline.
  - PureCS AI division oversees the ML-assisted normalisation layer.
""",
    "kpi_definitions.txt": """
KPI DEFINITIONS — SUPPLY CHAIN & PROCUREMENT (Unified Framework)

The following KPIs are standardised across SCM, Procurement, and facility-level
reporting (SEHA network, SSMC) to ensure consistency.

1. On-Time Delivery Rate (OTDR)
   Definition: % of PO lines delivered on or before the confirmed delivery date.
   Formula: (Lines delivered on time / Total lines delivered) × 100
   Target: ≥ 95%

2. Purchase Order Cycle Time
   Definition: Elapsed time in business days from PR approval to PO issuance.
   Target: ≤ 3 business days for catalogue items; ≤ 10 for non-catalogue.

3. Invoice Match Rate
   Definition: % of invoices matching the PO and GRN without manual intervention.
   Target: ≥ 90%

4. Vendor Scorecard Composite Score
   Weighted average of: Delivery Reliability (40%), Quality Compliance (35%),
   Invoice Accuracy (25%).

5. Item Master Data Quality Score
   Definition: % of active SKUs that are fully classified, deduplicated, and
   carry a valid UNSPSC code.
   Target: ≥ 98%

All KPIs are reported monthly via the automated BI dashboard surfaced to the
senior leadership team. Source data is pulled from the ERP system nightly.
""",
}


def inject_demo_docs():
    os.makedirs("docs", exist_ok=True)
    for fname, content in DEMO_DOCS.items():
        path = f"docs/{fname}"
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)
    print("  Demo documents written to /docs")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoints
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ingest():
    rag = RAGAssistant()
    rag.ingest()
    print("✓ Ingestion complete.")


def cmd_chat():
    rag = RAGAssistant()
    if os.path.exists("index.json"):
        rag.load_index()
    else:
        print("No index found — ingesting docs first …")
        rag.ingest()

    print("\n─────────────────────────────────────────")
    print("  RAG Procurement Assistant  (type 'quit' to exit, 'reset' to clear history)")
    print("─────────────────────────────────────────\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if q.lower() == "reset":
            rag.reset_history()
            print("  [History cleared]")
            continue
        rag.ask(q)


def cmd_ask(question: str):
    rag = RAGAssistant()
    if os.path.exists("index.json"):
        rag.load_index()
    else:
        rag.ingest()
    rag.ask(question)


def cmd_demo():
    print("\n── RAG Demo Mode ──────────────────────────────────────────────────")
    inject_demo_docs()
    rag = RAGAssistant()
    rag.ingest()
    questions = [
        "What certifications do vendors need to supply medical consumables?",
        "How is the Item Master data quality score defined?",
        "What happens to a vendor who scores below 70% on the scorecard?",
        "What is the KPI target for On-Time Delivery Rate?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        rag.ask(q)
    print("\n── Demo complete ──────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "chat":
        cmd_chat()
    elif args[0] == "ingest":
        cmd_ingest()
    elif args[0] == "ask":
        if len(args) < 2:
            print("Usage: python main.py ask \"your question here\"")
        else:
            cmd_ask(" ".join(args[1:]))
    elif args[0] == "demo":
        cmd_demo()
    else:
        print(__doc__)
