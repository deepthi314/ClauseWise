"""
backup_features.py
-------------------
Supplementary AI utility functions for ClauseWise Legal AI Assistant.
These can be imported into app.py for future feature upgrades.
"""

import streamlit as st
import re
import random

# -------------------------------------------------------------------
# 🧠 Named Entity Recognition
# -------------------------------------------------------------------
def named_entity_recognition(text):
    """
    Extract named entities such as Parties, Dates, Amounts, etc.
    Currently returns mock data for demonstration.
    """
    return {
        "Parties": ["Alice", "Bob"],
        "Dates": ["2009-01-15"],
        "Amounts": ["$138,708.00"]
    }

# -------------------------------------------------------------------
# 📜 Clause Extraction
# -------------------------------------------------------------------
def clause_extraction(text):
    """
    Identify major clauses and sections from legal text.
    """
    return [
        "Section 1.F: Base Rent",
        "Change Orders 1–8",
        "Amendment effective date"
    ]

# -------------------------------------------------------------------
# 🏷️ Document Classification
# -------------------------------------------------------------------
def document_classification(text):
    """
    Classify the uploaded legal document into common categories.
    """
    return "First Amendment to Lease Agreement"

# -------------------------------------------------------------------
# ⚠️ Risky Clause Flagging
# -------------------------------------------------------------------
def flag_risky_clauses(text):
    """
    Identify potentially risky or biased clauses using keyword patterns.
    """
    risky = []
    clauses = re.split(r"\n|\. ", text)
    for clause in clauses:
        if re.search(r"penalty|termination|sole|exclusive|arbitration", clause, re.I):
            risky.append(f"⚠️ Risky Clause Detected: {clause.strip()}")
    return risky or ["No high-risk clauses detected."]

# -------------------------------------------------------------------
# 📅 Timeline Visualization Placeholder
# -------------------------------------------------------------------
def timeline_visualization(text):
    """
    Generate a timeline of key events from the contract.
    To be replaced later with Plotly timeline chart.
    """
    st.info("📅 Timeline visualization placeholder (to be implemented with Plotly or Streamlit chart).")

# -------------------------------------------------------------------
# ⚖️ Fairness Assessment
# -------------------------------------------------------------------
def fairness_assessment(text):
    """
    Compute a basic fairness score using positive and negative keywords.
    """
    pos = len(re.findall(r"\b(mutual|both|equal|shared|balanced|fair)\b", text, re.I))
    neg = len(re.findall(r"\b(sole|exclusive|unilateral|one-sided|penalty)\b", text, re.I))
    score = max(0, min(100, 50 + (pos * 5) - (neg * 5)))

    if score >= 75:
        label = "Highly Fair"
    elif score >= 50:
        label = "Moderately Balanced"
    else:
        label = "Needs Review"

    return f"Fairness Score: {score}% ({label})"

# -------------------------------------------------------------------
# 🤖 AI Contract Assistant (Negotiation Suggestions)
# -------------------------------------------------------------------
def ai_contract_assistant(text):
    """
    Suggest negotiation points or improvements based on clause content.
    """
    suggestions = [
        "Consider reducing penalty duration to improve fairness.",
        "Add a mutual indemnification clause.",
        "Clarify termination notice period.",
        "Specify dispute resolution process clearly."
    ]
    return random.choice(suggestions)

# -------------------------------------------------------------------
# ⚖️ Contract Comparison
# -------------------------------------------------------------------
def contract_comparison(text1, text2):
    """
    Compare two contracts and identify key differences.
    """
    if len(text1) > len(text2):
        return "Contract A has more extensive clauses; Contract B is more concise."
    elif len(text1) < len(text2):
        return "Contract B includes additional terms; Contract A is shorter."
    else:
        return "Both contracts are similar in length and complexity."

# -------------------------------------------------------------------
# 🌐 Multilingual Support Placeholder
# -------------------------------------------------------------------
def multilingual_support(text, target_language):
    """
    Placeholder for translation system; integrated in app.py currently.
    """
    return f"Translated document into {target_language} (placeholder)."

# -------------------------------------------------------------------
# 🔊 Text-to-Audio Placeholder
# -------------------------------------------------------------------
def text_to_audio(text):
    """
    Placeholder for text-to-speech (implemented in app.py).
    """
    st.info("🔊 Text-to-audio placeholder. Add TTS module (gTTS/pyttsx3) to enable.")

# -------------------------------------------------------------------
# 🧩 Summary Function (to test all features)
# -------------------------------------------------------------------
def demo_all_features(sample_text):
    """
    Run all backup modules on sample text for testing.
    """
    st.subheader("🔍 Named Entities")
    st.json(named_entity_recognition(sample_text))

    st.subheader("📜 Extracted Clauses")
    st.write(clause_extraction(sample_text))

    st.subheader("🏷️ Document Type")
    st.success(document_classification(sample_text))

    st.subheader("⚠️ Risky Clauses")
    for c in flag_risky_clauses(sample_text):
        st.warning(c)

    st.subheader("⚖️ Fairness Assessment")
    st.info(fairness_assessment(sample_text))

    st.subheader("🤖 AI Contract Assistant Suggestion")
    st.write(ai_contract_assistant(sample_text))

    st.subheader("📅 Timeline Visualization")
    timeline_visualization(sample_text)
