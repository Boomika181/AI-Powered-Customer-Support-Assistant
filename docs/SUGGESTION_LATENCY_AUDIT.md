# Suggestion Generation Latency Audit

## Executive Summary
Following the implementation and verification of **Option B** (consolidated Sentiment Analysis + Query Categorization), the downstream pipeline latency on unthrottled turns dropped to **~2.98 seconds** (down from the pre-optimization baseline of ~5.77 seconds).

In this optimized flow, **Support Suggestion Generation** (`src/llm_suggestions.py`) is the final remaining serial stage in the critical path, consuming **~1.42s to 1.60s** (approx. 50% of the remaining clean downstream latency).

This audit inspects the complete code path of `src/llm_suggestions.py` and evaluates its runtime architecture against our already-optimized components.

---

## 1. Current Flow

The execution flow for suggestion generation occurs after the parallel pre-suggestion phase in `PipelineCoordinator`:

```
1. Parallel Phase Completes
   ├── Thread 1: Combined Sentiment + Category finishes (~1.61s)
   └── Thread 2: ChromaDB Retrieval finishes (~0.79s)
          ↓
2. `PipelineCoordinator._process_downstream_turn()` invokes `_suggestion_fn()`
   └── Passes:
       - `conversation_text`: Bounded customer transcript (latest 4 customer turns)
       - `retrieved_context`: In-memory list of 3 retrieved ChromaDB chunks
       - `category`: Detected query category (e.g. "Technical Troubleshooting")
          ↓
3. `generate_support_suggestion()` in `src/llm_suggestions.py`
   ├── a. Input Validation: Checks `conversation_text` (empty check) [<0.01 ms]
   ├── b. Empty Context Check: Returns fallback disclaimer if no chunks [<0.01 ms]
   ├── c. API Key Resolution: Resolves `GEMINI_API_KEY` from environment [<0.01 ms]
   ├── d. Model Resolution: Resolves `GEMINI_MODEL` (`gemini-3.5-flash-lite`) [<0.01 ms]
   ├── e. Prompt Construction: Formats 3 ChromaDB chunks via `_format_context_for_prompt()` [<0.01 ms]
   ├── f. Client Instantiation: Runs `client = genai.Client(api_key=resolved_key)` [~7.8 ms local CPU + fresh connection pool]
   ├── g. Gemini API Call: `client.models.generate_content(model=model_name, contents=prompt)` [~1400 - 1580 ms Network I/O + Inference]
   ├── h. Markdown Fence Stripping: Cleans raw response text via `_strip_markdown_fences()` [<0.01 ms]
   ├── i. JSON Deserialization: Parses string to dictionary via `json.loads()` [<0.01 ms]
   ├── j. Provenance Extraction: Deduplicates chunk metadata via `_extract_sources()` [<0.01 ms]
   └── k. Return Dictionary:
          `{"suggestion": ..., "confidence": ..., "sources": [...], "is_grounded": True, "category": ...}`
          ↓
4. Event Dispatch: `PipelineCoordinator` emits `"suggestion_update"` over event queue / WebSocket.
```

---

## 2. Current Configuration

The exact runtime configuration used by `src/llm_suggestions.py`:

| Parameter | Value in `src/llm_suggestions.py` | Notes / Evaluation |
| :--- | :--- | :--- |
| **Model** | `gemini-3.5-flash-lite` | Configured via `GEMINI_MODEL` in `.env` |
| **Client Lifecycle** | **Recreated on every request** | `client = genai.Client(api_key=...)` inside `generate_support_suggestion()` (Line 214) |
| **Client Reuse / Pooling** | **None** | No shared client, no connection Keep-Alive reuse across turns |
| **Generation Config** | **None** | Uses SDK default `GenerateContentConfig` |
| **Temperature** | **Not configured** | Defaults to model baseline (1.0) |
| **Max Output Tokens** | **Not configured** | Unbounded output generation window |
| **Thinking Configuration** | **None** | Dynamic default thinking/reasoning budget on Gemini 3.5 |
| **Response MIME Type** | **None** (`text/plain` default) | Prompt instructs JSON formatting; parsed via regex + `json.loads` |
| **Request Timeout** | **None** | Default SDK client timeout |
| **API Retries** | `max_retries = 3` | Retries on `503 UNAVAILABLE` (1s delay). Immediately raises on `429 RESOURCE_EXHAUSTED` |
| **Calls per Suggestion** | **1 generation call** | Single round-trip inference per customer turn |
| **Prompt Size** | **3,800 to 4,312 characters** | ~950 to 1,100 tokens (includes 3 ChromaDB chunks + safety rules) |
| **Output Size** | **~350 to 550 characters** | ~80 to 130 tokens (concise JSON with suggestion text & confidence) |

---

## 3. Timing Breakdown

### Measured In-Process Timings (Empirical Benchmark)
- **Prompt String Formatting (`_format_context_for_prompt`):** `0.0010 ms` (completely negligible).
- **Markdown Fence Stripping + JSON Parsing:** `0.0016 ms` (completely negligible).
- **Client Instantiation (`genai.Client()`):** `7.76 ms` local Python execution time.
- **Pre-Retrieved Context:** **0.00 ms**. ChromaDB retrieval is completed before `generate_support_suggestion()` is invoked; the chunks are already in-memory Python objects.

### Network and Inference Timings
- **TCP Handshake + TLS 1.3 Setup:** **~80 to 180 ms** per call due to fresh client creation on every invocation (no Keep-Alive socket reuse).
- **Time-to-First-Token (TTFT) + Inference:** **~1,250 to 1,400 ms** for `gemini-3.5-flash-lite` to ingest ~1,000 input tokens and synthesize ~100 output tokens.
- **Total Suggestion Generation Latency:** **~1,420 ms to 1,600 ms**.

---

## 4. Comparison With Role Classifier

In Phase 2 Step 1, a latency audit of `src/speaker_role_classifier.py` identified client instantiation overhead and implemented a persistent client caching pattern.

| Characteristic | `src/speaker_role_classifier.py` (Optimized) | `src/llm_suggestions.py` (Current) |
| :--- | :--- | :--- |
| **Client Creation** | `get_shared_gemini_client()` singleton helper | Inline `genai.Client()` inside generation function |
| **Client Reuse** | **Yes** — cached across all turns and session coordinator | **No** — recreated from scratch on every turn |
| **Client Injection** | Accepts optional `client` parameter | Does not accept `client` parameter |
| **Connection Pooling** | Reuses open HTTPS/TLS Keep-Alive channel | Negotiates fresh TLS handshake on every customer turn |
| **Overhead Saved** | Saves ~100–250 ms cold-start per turn | Currently pays ~100–250 ms cold-start per turn |

---

## 5. Optimization Candidates

### Ranked Candidates

#### 1. Reuse Persistent Gemini Client (HIGH Impact / LOW Risk)
- **What is slow:** Constructing a fresh `genai.Client(api_key=resolved_key)` on every invocation of `generate_support_suggestion()`.
- **Evidence:** Re-instantiating the client creates a new HTTP transport session, forcing an SSL/TLS 1.3 handshake and token credential resolution on every turn (~80–200 ms).
- **Estimated impact:** **~100 ms to 200 ms latency reduction** on suggestion generation with zero change to prompts or accuracy.
- **Risk:** **Zero**. Proven in `src/speaker_role_classifier.py`.

#### 2. Explicit Generation Config & Output Token Capping (MEDIUM Impact / LOW Risk)
- **What is slow:** The generation call runs with default unconstrained generation parameters.
- **Evidence:** Suggestions are strictly bounded (~50–80 words), but the model has an open token generation horizon. Specifying `GenerateContentConfig(max_output_tokens=256, temperature=0.2)` signals deterministic procedural termination.
- **Estimated impact:** **~50 ms to 150 ms latency reduction**.
- **Risk:** **Low**. Token ceiling must be set safely above the longest expected industrial safety procedure (256 tokens = ~1,000 characters, sufficient for our ~390 character suggestions).

#### 3. Trimming RAG Context Payload (LOW Impact / HIGH Risk)
- **What is slow:** Ingesting 3 full ChromaDB manual chunks (~2,800 characters of technical documentation).
- **Evidence:** Input processing in Flash Lite is already fast (~950 tokens is handled in <100 ms of TTFT). Reducing to 2 chunks would only shave ~20–30 ms of input encoding time.
- **Risk:** **HIGH**. Trimming chunks risks omitting critical safety interlocks, error code resolution steps, or part numbers required for strict grounding. **Not recommended.**

---

## 6. Recommended Next Experiment

### Recommended Experiment: **Persistent Gemini Client Reuse for `llm_suggestions.py`**

**Rationale:**
1. **Proven Pattern:** This is the identical optimization that successfully stabilized and accelerated `speaker_role_classifier.py`.
2. **Zero Functional Risk:** It changes zero prompts, zero tokens, zero models, and zero grounding logic.
3. **Pure Latency Win:** Eliminates redundant TLS handshake and client setup overhead, saving **~100–200 ms** on the critical path.
4. **Implementation Simplicity:** Implement `get_shared_gemini_client()` in `src/llm_suggestions.py` (or inject the coordinator's existing shared client), mirroring the role classifier.
