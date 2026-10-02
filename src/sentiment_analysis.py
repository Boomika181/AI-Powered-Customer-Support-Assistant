"""
src/sentiment_analysis.py

Sentiment Analysis Module using Google Gemini Flash.
Scheduled for Phase 1 implementation.

Phase 1 Responsibilities:
- Analyze customer tone from dialogue history.
- Classify tone into: Positive, Neutral, Negative / Agitated.
- Return structured JSON output for dashboard consumption.
"""

from typing import Dict, Any, Optional


def analyze_sentiment(conversation_text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    TODO (Phase 1): Call Gemini Flash to classify customer sentiment.
    
    Expected return structure:
    {
        "sentiment": "Positive" | "Neutral" | "Negative" | "Agitated",
        "confidence": float,
        "rationale": str
    }
    """
    raise NotImplementedError("Phase 1 Feature: Gemini sentiment analysis.")
