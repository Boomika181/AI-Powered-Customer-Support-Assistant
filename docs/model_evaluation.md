# Model Comparison & Evaluation

This document preserves the evaluation results of candidate models for the AI-Powered Customer Support Assistant POC, comparing **Google Gemini 3.8 Flash**, **DeepSeek V4.1 Flash**, and **Qwen 3.7 Plus**.

## 1. Sentiment Analysis Evaluation

**Task:** Classify customer tone into:
- Positive
- Neutral
- Negative / Agitated

| Model | Correct (out of 10) | Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **Gemini 3.8 Flash** | **10 / 10** | **100%** | **PASS** |
| DeepSeek V4.1 Flash | 10 / 10 | 100% | PASS |
| Qwen 3.7 Plus (slow) | 9 / 10 | 90% | PASS |

---

## 2. Category Classification Evaluation

**Task:** Classify customer queries into one of three core support categories:
1. Machine Operation Issues
2. Maintenance & Parts
3. Technical Troubleshooting

| Model | Correct (out of 10) | Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **Gemini 3.8 Flash** | **10 / 10** | **100%** | **PASS** |
| DeepSeek V4.1 Flash | 8 / 10 | 80% | FAIL |
| Qwen 3.7 Plus (slow) | 10 / 10 | 100% | PASS |

---

## 3. RAG Suggestions Evaluation

**Task:** Provide grounded support suggestions using technical user manuals:
- 2–3 relevant suggestions per question.
- Strict grounding: Use only the manual (no outside knowledge).
- Include document reference code and page number.
- Explicit fallback: Say *"Not found in the provided manual"* if information is missing.

| Model | Correct (out of 10) | Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **Gemini 3.8 Flash** | **10 / 10** | **100%** | **PASS** |
| DeepSeek V4.1 Flash | 10 / 10 | 100% | PASS |
| Qwen 3.7 Plus (slow) | 10 / 10 | 100% | PASS |

---

## Overall Results Summary

| Model | Sentiment (10) | Category (10) | Suggestions (10) | Average Accuracy | Meets SOW Target (≥85%) | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini 3.8 Flash** | **100%** | **100%** | **100%** | **100%** | **Yes** | **Yes (Primary Stack)** |
| DeepSeek V4.1 Flash | 100% | 80% | 100% | 93.3% | Yes (sub-target failed) | No (Alternative) |
| Qwen 3.7 Plus (slow) | 90% | 100% | 100% | 96.7% | Yes | No (Latency too high) |

### Key Takeaways
1. **Gemini 3.8 Flash** is the only model achieving a perfect 100% score across all three tasks while maintaining fast streaming response times.
2. **DeepSeek V4.1 Flash** dropped below the 85% requirement in query categorization (80%), leading to classification errors on complex multi-part queries.
3. **Qwen 3.7 Plus** met accuracy thresholds but demonstrated unacceptable latency for real-time customer support call assistance (<2s target).
