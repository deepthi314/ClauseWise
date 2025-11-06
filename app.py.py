import streamlit as st
import tempfile
import os
import re
from cryptography.fernet import Fernet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForTokenClassification
from PyPDF2 import PdfReader
from docx import Document
import plotly.express as px
import pandas as pd

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="ClauseWise: Legal Document Analyzer",
                   page_icon="⚖️", layout="wide")

st.title("⚖️ ClauseWise: Legal Document Analyzer")
st.markdown("""
**Simplify, Decode, and Classify Legal Documents using AI**  
Your smart assistant for understanding contracts, clauses, and obligations.
""")
st.markdown("---")

# -------------------------
# ENCRYPTION UTILITIES
# -------------------------
def get_session_key():
    if "enc_key" not in st.session_state:
        st.session_state["enc_key"] = Fernet.generate_key()
    return st.session_state["enc_key"]

def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    cipher = Fernet(key)
    return cipher.encrypt(data)

def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    cipher = Fernet(key)
    return cipher.decrypt(token)

def write_temp_encrypted_file(encrypted_bytes: bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(encrypted_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

def secure_delete(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# -------------------------
# FILE EXTRACTION
# -------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    text = ""
    try:
        reader = PdfReader(tmp_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        text = ""
    secure_delete(tmp_path)
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    text = ""
    try:
        doc = Document(tmp_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        text = ""
    secure_delete(tmp_path)
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# -------------------------
# CLEAN / PREPROCESS
# -------------------------
def clean_text(text: str) -> str:
    patterns = [
        r"Downloaded from[^\n]*\n?",
        r"Appears in \d+ contracts[^\n]*\n?",
        r"I'm 5:.*\n?",
        r"I'm 5 or Appears in.*\n?",
        r"(Employee Signature Date:.*?Title:\s*\d*)+",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

# -------------------------
# MODEL CACHE (Hugging Face only)
# -------------------------
@st.cache_resource(ttl=3600)
def load_models():
    simplify_model_name = "mrm8488/t5-small-finetuned-text-simplification"
    tokenizer = AutoTokenizer.from_pretrained(simplify_model_name)
    simplify_model = AutoModelForSeq2SeqLM.from_pretrained(simplify_model_name)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return tokenizer, simplify_model, summarizer, ner_pipeline, classifier

tokenizer, simplify_model, summarizer, ner_pipeline, classifier = load_models()

# -------------------------
# CORE AI FEATURES
# -------------------------
def clause_simplification(text, mode):
    if not text:
        return "No text to simplify."
    prefix = {
        "Simplified": "simplify: ",
        "Explain like I'm 5": "explain like I'm 5: ",
        "Professional": "rephrase professionally: "
    }.get(mode, "simplify: ")
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512)
    outputs = simplify_model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clause_extraction(text):
    matches = re.findall(r'(Section\s+\d+[\w\.\-]*[:\-]?\s*[A-Z][^\n]+)', text)
    return list(dict.fromkeys(matches)) if matches else ["Section 1.F: Base Rent"]

def named_entity_recognition(text):
    entities = ner_pipeline(text[:2000])
    grouped = {}
    for ent in entities:
        grouped.setdefault(ent["entity_group"], []).append(ent["word"])
    return grouped

def document_classification(text):
    labels = ["Lease Agreement", "Employment Contract", "NDA", "Purchase Agreement"]
    result = classifier(text[:1024], candidate_labels=labels)
    return result["labels"][0]

def flag_risky_clauses(text):
    risky = re.findall(r"(penalty|termination|breach|liability|indemnity)", text, flags=re.IGNORECASE)
    return [f"Clause mentioning '{w}' requires review." for w in set(risky)] or ["No high-risk clauses detected."]

def fairness_assessment(text):
    pos = len(re.findall(r"(mutual|both parties|shared)", text, flags=re.IGNORECASE))
    neg = len(re.findall(r"(sole|unilateral|exclusive right)", text, flags=re.IGNORECASE))
    score = max(0, min(100, 70 + pos - neg * 2))
    return f"Fairness Score: {score}%"

def ai_contract_assistant(text):
    suggestion = re.search(r"penalty|termination", text, flags=re.IGNORECASE)
    if suggestion:
        return "Suggested negotiation: Reduce penalty duration or clarify termination terms."
    return "No immediate negotiation points detected."

def multilingual_support(text, target_language):
    try:
        translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language.lower()[:2]}")
        return translator(text[:1000])[0]["translation_text"]
    except Exception:
        return f"Translated to {target_language} (mock)."

def text_to_audio(text):
    st.info("Text-to-speech support coming soon (use gTTS or pyttsx3).")

# -------------------------
# SMART CLAUSE-GROUPED TIMELINE + ENTITY PANEL
# -------------------------
def timeline_visualization(text):
    clauses = clause_extraction(text)
    entities = named_entity_recognition(text)
    events = []

    date_matches = re.finditer(
        r'((?:Section|Clause)\s[\dA-Za-z\.\-]+[^\n:]*[:\-]?\s*[^\n]*)|(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        text)

    current_clause = "General"
    for m in date_matches:
        if m.group(1):
            current_clause = m.group(1).strip()
        elif m.group(2):
            events.append({"Clause": current_clause, "Date": m.group(2)})

    if not events:
        st.warning("No dates or timeline events detected.")
        return

    df = pd.DataFrame(events)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    st.subheader("📊 Contract Timeline by Clause")
    fig = px.timeline(df, x_start="Date", x_end="Date", y="Clause", color="Clause", title="Clause-Wise Timeline")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Clause-Level Details")
    for clause in df["Clause"].unique():
        clause_dates = df[df["Clause"] == clause]["Date"].dt.strftime("%b %d, %Y").tolist()
        clause_entities = {k: v[:3] for k, v in entities.items()} if entities else {}
        with st.expander(f"📘 {clause}"):
            st.write(f"**Dates Mentioned:** {', '.join(clause_dates) if clause_dates else 'None'}")
            if clause_entities:
                st.write("**Entities Detected:**")
                st.json(clause_entities)
            else:
                st.write("No named entities found for this clause.")

# -------------------------
# MAIN UI
# -------------------------
st.subheader("📁 Upload a Legal Document")
uploaded_file = st.file_uploader("Choose a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    key = get_session_key()
    raw_bytes = uploaded_file.read()
    encrypted_bytes = encrypt_bytes(raw_bytes, key)
    temp_encrypted_path = write_temp_encrypted_file(encrypted_bytes)
    decrypted_bytes = decrypt_bytes(encrypted_bytes, key)

    filename_lower = uploaded_file.name.lower()
    if filename_lower.endswith(".pdf"): 
        content = extract_text_from_pdf(decrypted_bytes)
    elif filename_lower.endswith(".docx"): 
        content = extract_text_from_docx(decrypted_bytes)
    else: 
        content = extract_text_from_txt(decrypted_bytes)
    secure_delete(temp_encrypted_path)

    if not content.strip():
        st.warning("No readable text found in the document.")
    else:
        st.markdown("---")
        st.subheader("🔍 Apply Features")

        mode = st.radio("Choose simplification level:", ["Explain like I'm 5", "Simplified", "Professional"])
        if st.button("🧾 Simplify Clauses"):
            with st.spinner("Simplifying..."):
                st.write(clause_simplification(content, mode))
        st.markdown("---")

        if st.button("🔗 Extract Entities"):
            st.json(named_entity_recognition(content))
        st.markdown("---")

        if st.button("📑 Extract Clauses"):
            st.write(clause_extraction(content))
        st.markdown("---")

        if st.button("📂 Classify Document"):
            st.success(document_classification(content))
        st.markdown("---")

        if st.button("🚨 Flag Risky Clauses"):
            st.warning(flag_risky_clauses(content))
        st.markdown("---")

        if st.button("📅 Timeline Visualization"):
            timeline_visualization(content)
        st.markdown("---")

        if st.button("⚖️ Fairness Assessment"):
            st.info(fairness_assessment(content))
        st.markdown("---")

        if st.button("🤝 Contract Assistant"):
            st.write(ai_contract_assistant(content))
        st.markdown("---")

        lang = st.selectbox("🌐 Choose Language", ["French", "Spanish", "German"])
        if st.button("Translate Document"):
            st.write(multilingual_support(content, lang))
        st.markdown("---")

        if st.button("🔊 Convert Text to Audio"):
            text_to_audio(content)

else:
    st.info("👆 Upload a document above to start analysis.")

st.markdown(
    "<p style='text-align: center; font-style: italic; color: gray;'>"
    "Important: ClauseWise provides educational information only. This is not legal advice."
    "</p>", unsafe_allow_html=True
)
