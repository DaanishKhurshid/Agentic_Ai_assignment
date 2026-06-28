## Adaptive Multi-Agent Support Network with LangGraph & Streamlit

An enterprise-grade multi-agent technical support network built to interpret customer queries, retrieve local context, utilize live web fallback engines, and execute automated quality assurance (QA) self-correction loops.


## System Architecture & Node Topology

The system is engineered as a cyclic State Machine using LangGraph to handle multi-agent collaboration and shared state memory data. The execution path flows exactly as follows:

1. **`rewrite_query` (Support Specialist Node):** Translates ambiguous or messy human inputs into optimized, keyword-driven search queries.
2. **`retrieve_and_grade` (QA Gatekeeper Node):** Extracts top document chunks from the local `Chroma` database and scores their relevance.
   * **Path A (Relevant Docs):** Routes directly to the answer generator.
   * **Path B (Irrelevant Docs):** Triggers the live web search fallback.
3. **`web_search` (Fallback Node):** Uses Google Gemini's live production search engine to fetch missing real-time operational or plant updates.
4. **`generate_answer` (Synthesis Node):** Generates a formal technical response bounded strictly by the collected text history.
5. **`qa_hallucination` (Auditing Node):** Cross-examines the output text against the sources. If it fails, it increments the loop counter and triggers an automated self-correction retry loop back to Node 1.

---

## Architectural Tool Justification: Gemini Native Grounding

Instead of utilizing standard third-party text scrapers (such as Tavily or DuckDuckGo API) which return messy, unformatted raw HTML/text, this architecture relies on **Google Gemini's Native Search Grounding** (`"google_search_enabled": True`). 

This represents an optimized enterprise design pattern. By using the LLM's built-in live production index, the fallback node avoids external network latency and structural token bloat. The system receives a cleanly synthesized, factually authoritative context summary, maximizing the accuracy of the downstream generator node.

---

## Project Directory Structure

```text
Multi_Agent_Assignment/
│
├── app_code.py         # Main backend containing LangGraph topology & node logic
├── app.py              # Frontend script running the interactive Streamlit user interface
├── requirements.txt    # Frozen library versions for environment installation
└── README.md           # Project configuration and architectural guide
```

---

## Installation & Local Execution

Follow these steps to configure your local runtime environment:

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed on your system. Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Credentials:**
   Provide your Google Gemini system key before launching the server pipeline:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. **Launch the User Interface:**
   Boot up the live interactive chat dashboard using Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## ⚠️ Runtime Trace Simulation Notice

During the final system validation and stress-testing phases, the development environment exhausted its maximum **Google GenAI API token quota limit**. 

