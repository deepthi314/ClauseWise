# app.py
"""
ClauseWise - Multilingual Legal AI Assistant
Final corrected version (no syntax errors)
"""

import os
import re
import io
import uuid
import tempfile
from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.express as px
from PyPDF2 import PdfReader
from docx import Document
from gtts import gTTS
import json
from backup_features import (
    named_entity_recognition,
    clause_extraction,
    flag_risky_clauses,
    fairness_assessment,
    ai_contract_assistant
)

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="ClauseWise", page_icon="⚖️", layout="wide")

# Language map
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "Hindi": "hi",
    "French": "fr",
    "German": "de"
}

# ---------------------------
# TRANSLATIONS (simplified)
# ---------------------------
TRANSLATIONS = {
    "en": {
        "title": "ClauseWise: Multilingual Legal AI Assistant",
        "subtitle": "Simplify, analyze, and discuss legal documents with AI — in your language.",
        "select_language": "Select Language:",
        "tabs": {
            "analyzer": "Analyzer",
            "translate_audio": "Translate & Audio",
            "advanced": "Advanced Analysis",
            "balance": "Balance Analysis",
            "alternatives": "Alternative Clauses",
            "chat": "AI Legal Chat"
        },
        "upload": {
            "label": "Upload NDA Document",
            "drop_placeholder": "Drag and drop file here",
            "or_paste": "Or paste text:",
            "invalid_doc": "Uploaded document doesn’t look like an NDA. Please upload a valid NDA."
        },
        "footer": "Important: ClauseWise provides educational information only. This is not legal advice. Always consult a licensed attorney for legal guidance on your specific situation."
    },
    "es": {"title": "ClauseWise: Asistente Legal Multilingüe", "subtitle": "Analiza y simplifica documentos legales con IA.", "select_language": "Seleccionar idioma:"},
    "hi": {"title": "ClauseWise: बहुभाषी कानूनी AI सहायक", "subtitle": "AI के साथ कानूनी दस्तावेज़ों को सरल बनाएं और विश्लेषण करें।", "select_language": "भाषा चुनें:"},
    "fr": {"title": "ClauseWise : Assistant juridique IA", "subtitle": "Analysez et simplifiez vos documents juridiques grâce à l'IA.", "select_language": "Choisir la langue:"},
    "de": {"title": "ClauseWise: Mehrsprachiger KI-Rechtsassistent", "subtitle": "Analysieren und vereinfachen Sie juristische Dokumente mit KI.", "select_language": "Sprache auswählen:"}
}


def t(path, lang="en"):
    """Access translation key via dot notation"""
    keys = path.split(".")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    for k in keys:
        if isinstance(text, dict):
            text = text.get(k, None)
    return text or TRANSLATIONS["en"].get(keys[-1], path)


# ---------------------------
# FILE EXTRACTORS
# ---------------------------
def extract_text(file):
    if not file:
        return ""
    name = file.name.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
    tmp.write(file.read())
    tmp.close()
    text = ""
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(tmp.name)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif name.endswith(".docx"):
            doc = Document(tmp.name)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(tmp.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
    finally:
        os.remove(tmp.name)
    return text.strip()


# ---------------------------
# FIXED NDA DETECTION FUNCTION ✅
# ---------------------------
def is_nda_text_like_nda(text: str) -> bool:
    """Simple keyword-based NDA detector"""
    if not text:
        return False
    keywords = [
        "non-disclosure", "confidential", "receiving party",
        "disclosing party", "nondisclosure", "confidential information"
    ]
    text_low = text.lower()
    return any(k in text_low for k in keywords)


# ---------------------------
# TEXT TO SPEECH
# ---------------------------
def text_to_speech_bytes(text, lang="en"):
    try:
        gtts_lang = {"en": "en", "es": "es", "hi": "hi", "fr": "fr", "de": "de"}[lang[:2]]
        tts = gTTS(text[:1500], lang=gtts_lang)
        bio = BytesIO()
        tts.write_to_fp(bio)
        bio.seek(0)
        return bio
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None


# ---------------------------
# LIGHTWEIGHT SIMPLIFIER
# ---------------------------
def simplify_text(text, mode="Simple"):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if mode.lower().startswith("eli"):
        return sentences[0] + " — In short, this means something important you should know."
    elif mode.lower().startswith("pro"):
        return " ".join(sentences[:2]) + " (professional summary)."
    else:
        return " ".join(sentences[:2])


# ---------------------------
# FAIRNESS METER
# ---------------------------
def show_fairness_meter(text, lang="en"):
    pos = len(re.findall(r"\b(mutual|both parties|equal|balanced)\b", text, re.I))
    neg = len(re.findall(r"\b(sole|unilateral|exclusive|only)\b", text, re.I))
    score = max(0, min(100, 50 + (pos * 5) - (neg * 5)))
    df = pd.DataFrame({"Aspect": ["Company", "Balanced", "You"], "Score": [100 - score, score, score]})
    fig = px.bar(df, x="Score", y="Aspect", orientation="h", text="Score", title="Fairness Meter")
    st.plotly_chart(fig, use_container_width=True, key=f"fairness-{uuid.uuid4().hex}")


# ---------------------------
# CHATBOT (local logic)
# ---------------------------
def init_chat():
    if "chat" not in st.session_state:
        st.session_state.chat = []


def chatbot_reply(user_msg, lang="en"):
    msg = user_msg.lower()
    if "compet" in msg:
        reply = "This clause may restrict competition. Review duration, scope, and geography."
    elif "duration" in msg or "how long" in msg:
        reply = "NDA duration varies. Check the clause for exact terms or termination conditions."
    elif "penalt" in msg or "breach" in msg:
        reply = "Usually, NDAs define remedies or penalties for breach. Look for monetary caps and resolution process."
    else:
        reply = ai_contract_assistant(user_msg)

    reminder = "Always consult a licensed attorney for legal advice."
    return reply + "\n\n" + reminder


# ---------------------------
# MAIN APP UI
# ---------------------------
def main():
    # Language selector
    st.title(t("title"))
    st.markdown(f"_{t('subtitle')}_")
    lang_sel = st.selectbox(t("select_language"), list(LANG_MAP.keys()))
    lang_code = LANG_MAP[lang_sel]

    tabs = st.tabs([
        t("tabs.analyzer", lang_code),
        t("tabs.translate_audio", lang_code),
        t("tabs.advanced", lang_code),
        t("tabs.balance", lang_code),
        t("tabs.alternatives", lang_code),
        t("tabs.chat", lang_code)
    ])
    analyzer, translate, advanced, balance, alt, chat = tabs

    # 1️⃣ ANALYZER TAB
    with analyzer:
        st.subheader(t("tabs.analyzer", lang_code))
        file = st.file_uploader(t("upload.label", lang_code), type=["pdf", "docx", "txt"])
        text_input = st.text_area(t("upload.or_paste", lang_code), height=200)

        text = ""
        if file:
            text = extract_text(file)
        elif text_input:
            text = text_input

        if text:
            if is_nda_text_like_nda(text):
                st.success("✅ NDA detected — analyzing...")
            else:
                st.error(t("upload.invalid_doc", lang_code))
                text = ""

        mode = st.radio("Mode", ["ELI5", "Simplified", "Professional"], horizontal=True)

        if st.button("🧾 Simplify Clauses"):
            if not text:
                st.warning("Please upload a valid NDA.")
            else:
                result = simplify_text(text, mode)
                st.write(result)
                audio = text_to_speech_bytes(result, lang_code)
                if audio:
                    st.audio(audio.read(), format="audio/mp3")

        if st.button("⚖️ Fairness Analysis"):
            if text:
                show_fairness_meter(text, lang_code)
            else:
                st.warning("Please upload an NDA to analyze.")

    # 2️⃣ TRANSLATE TAB
    with translate:
        st.subheader(t("tabs.translate_audio", lang_code))
        content = st.text_area("Enter text to translate", height=200)
        if st.button("🎧 Generate Audio"):
            audio = text_to_speech_bytes(content, lang_code)
            if audio:
                st.audio(audio.read(), format="audio/mp3")

    # 3️⃣ ADVANCED ANALYSIS TAB
    with advanced:
        st.subheader("🧠 Advanced Analysis")
        text_advanced = st.text_area("Paste document text or load sample", height=200)
        if st.button("Run Advanced Analysis"):
            if not text_advanced.strip():
                st.warning("Please enter text.")
            else:
                ents = named_entity_recognition(text_advanced)
                st.write("**Named Entities**", ents)
                clauses = clause_extraction(text_advanced)
                st.write("**Extracted Clauses**")
                for c in clauses:
                    st.write("-", c)
                risks = flag_risky_clauses(text_advanced)
                st.write("**Top 5 Risks**")
                for r in risks[:5]:
                    st.warning(r)
                fair = fairness_assessment(text_advanced)
                st.info(f"Fairness Score: {fair}")
                suggestion = ai_contract_assistant(text_advanced)
                st.success(suggestion)

    # 4️⃣ BALANCE TAB
    with balance:
        st.subheader("⚖️ Balance Analysis")
        data = pd.DataFrame({
            "Category": ["Termination", "Liability", "IP", "Restrictions", "Obligations"],
            "You": [30, 10, 0, 40, 40],
            "Company": [70, 90, 100, 60, 60]
        })
        for i, row in data.iterrows():
            fig = px.bar(pd.DataFrame({
                "Side": ["You", "Company"],
                "Score": [row["You"], row["Company"]]
            }), x="Score", y="Side", orientation="h", title=row["Category"])
            st.plotly_chart(fig, use_container_width=True, key=f"bal-{uuid.uuid4().hex}")

    # 5️⃣ ALTERNATIVE CLAUSES
    with alt:
        st.subheader("📋 Alternative Clauses")
        examples = [
            {"title": "Common Alternative", "text": "Employee liability capped at $50,000.", "freq": 78},
            {"title": "Moderate Alternative", "text": "Mutual confidentiality clause — both parties equal.", "freq": 45},
            {"title": "Aggressive Alternative", "text": "No cap on liability for intentional acts.", "freq": 12}
        ]
        for i, ex in enumerate(examples):
            st.markdown(f"**{ex['title']} ({ex['freq']}%)**")
            st.write(ex["text"])
            if st.button(f"Copy {i}", key=f"copy-{i}"):
                st.experimental_set_clipboard(ex["text"])
                st.toast("Copied!")

    # 6️⃣ CHAT TAB
    with chat:
        st.subheader("💬 AI Legal Chat")
        init_chat()
        for role, msg in st.session_state.chat:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Assistant:** {msg}")
        msg = st.text_area("Ask a legal question", key="chat_msg")
        if st.button("Send"):
            if msg.strip():
                st.session_state.chat.append(("user", msg))
                reply = chatbot_reply(msg, lang_code)
                st.session_state.chat.append(("assistant", reply))
                st.experimental_rerun()

    # Footer Disclaimer
    st.write("---")
    st.info(t("footer", lang_code))


if __name__ == "__main__":
    main()
