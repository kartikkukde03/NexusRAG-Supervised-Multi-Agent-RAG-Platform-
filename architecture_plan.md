# Multi-Agent RAG System Architecture

A clean, supervised multi-agent system designed for accurate question answering and whole-document summarization over large documents (PDFs, DOCX, CSVs).

---

## 1. Core Idea

Instead of using one big prompt, the system divides the work among **specialized agents** coordinated by **LangGraph**.

```mermaid
flowchart LR
    User["User Question"] --> Orchestrator["LangGraph Orchestrator"]
    Orchestrator <--> Agents["Specialized Agents\n(Planner, Search, Table, Writer, Citation)"]
    Orchestrator --> Output["Verified Answer with Page Citations [S1]"]
```

- **Shared Blackboard**: Agents do not chat with each other. They read from and write to one shared dictionary (`WorkflowState`).
- **Exact Math**: Tables are calculated with Python math (0% LLM guessing).
- **Verifiable Sources**: Every statement must cite its source (e.g. `[S1]`, `[S2]`).

---

## 2. The Agents & Their Roles

| Agent | What It Does | Tools Used |
| :--- | :--- | :--- |
| **`PlannerAgent`** | Understands user intent and rewrites follow-up questions | LLM planning prompt |
| **`ResearchAgent`** | Finds relevant text and tables across the document | `search_documents`, `get_page_range`, `get_section` |
| **`DocumentSummaryAgent`** | Gathers strategic anchors for whole-document summaries | `get_document_outline`, `query_by_pages` |
| **`TableAgent`** | Performs exact math on markdown tables | `calculate_table_aggregation`, `lookup_table_cell` |
| **`WriterAgent`** | Writes the final grounded answer or executive summary | System prompts + token streamer |
| **`CitationAgent`** | Audits citations (`[S1]`) and fixes any missing references | `verify_citations`, `format_source_references` |

---

## 3. How a Question is Answered (Execution Flow)

```mermaid
flowchart TD
    Q(["1. User Question"]) --> Planner["2. PlannerAgent\n(Finds intent & fixes follow-up questions)"]
    
    Planner -- "Full Document Summary" --> Summary["3A. DocumentSummaryAgent\n(Pulls Head + Milestones + Tail)"]
    Planner -- "Specific Question" --> Search["3B. ResearchAgent\n(Hybrid Vector + Keyword Search)"]
    
    Search --> TableCheck{"4. Needs Table Math?"}
    Summary --> TableCheck
    
    TableCheck -- "Yes" --> TableMath["TableAgent: Exact Python Math"]
    TableCheck -- "No" --> Writer["5. WriterAgent: Drafts Answer"]
    
    Writer --> CitationCheck{"6. CitationAgent: Valid [S1] tags?"}
    CitationCheck -- "Missing" --> Repair["Auto-Repair Loop"]
    Repair --> Done
    CitationCheck -- "Valid" --> Done(["7. Deliver Verified Answer"])
    TableMath --> Done
```

---

## 4. How Memory is Shared

Agents share data through a single shared state called **`WorkflowState`**.

```mermaid
flowchart LR
    subgraph MemoryRelay ["How Data Moves from Agent to Agent"]
        direction LR
        P["Planner\nadds search query"] --> R["Research\nadds text chunks"]
        R --> W["Writer\nadds draft answer"]
        W --> C["Citation\nadds verified answer"]
    end
```

### The 3 Memory Levels

1. **Working Memory (`WorkflowState`)**:
   - Temporary dictionary used during a single question.
   - Passes retrieved chunks and draft answers from one agent to the next.

2. **Conversation Memory (`MemorySaver`)**:
   - Remembers past chat turns for the user session.
   - Allows follow-ups like *"What about for zone 2?"* without re-reading the entire file.

3. **Document Storage (ChromaDB + SQLite)**:
   - Permanent storage of document text, embeddings, and tables.

---

## 5. How Whole-Document Summarization Works

Simple RAG fails on 50–400 page PDFs because it only retrieves the first few pages. This system uses **Milestone Sampling**:

```mermaid
flowchart TD
    PDF["Uploaded Document (1 to 400+ Pages)"] --> Sampler["DocumentSummaryAgent"]
    
    Sampler --> Head["Head (Pages 1-3): Title, Preamble & Table of Contents"]
    Sampler --> Body["Body Milestones: Sampled across percentiles (25%, 50%, 75%)"]
    Sampler --> Tail["Tail (Last Pages): Directives, Exemptions & Next Steps"]
    
    Head & Body & Tail --> Writer["WriterAgent: 5-Part Executive Briefing\n(Executive Overview, Section Breakdown, Core Rules, Exceptions, Directives)"]
    
    Writer --> Citations["CitationAgent: Verifies [S1], [S2] page tags"]
```

- **Fast**: Generates in **one single LLM pass (2–4 seconds)**.
- **Complete**: Covers the beginning, middle, and end of the document.
- **Accurate**: Every claim links to an exact page number citation.

