import streamlit as st
import tempfile
import os
import re
import io
import json
from typing import List, Dict, Tuple, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader
import docx
import spacy
import math
import time

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="ClauseWise – Granite 3.2 (2B) Legal Assistant", page_icon="⚖️", layout="wide")

# -------------------------
# MODEL SETUP WITH OPTIMIZATIONS
# -------------------------
MODEL_ID = "ibm-granite/granite-3.2-2b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

@st.cache_resource
def load_llm_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )
        if DEVICE != "cuda":
            model.to(DEVICE)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_llm_model()

try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("spaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
    nlp = None

# -------------------------
# OPTIMIZED HELPER FUNCTIONS
# -------------------------
def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys = f"<|system|>\n{system_prompt}\n" if system_prompt else ""
        usr = f"<|user|>\n{user_prompt}\n<|assistant|>\n"
        return sys + usr

def llm_generate_optimized(system_prompt: str, user_prompt: str, max_new_tokens=256, temperature=0.3, top_p=0.9) -> str:
    if model is None or tokenizer is None:
        return "Model not available. Please check model loading."
    
    try:
        prompt = build_chat_prompt(system_prompt, user_prompt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract assistant response more efficiently
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].strip()
        elif full_text.startswith(prompt):
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()
            
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# -------------------------
# DOCUMENT LOADING
# -------------------------
def load_text_from_pdf(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def load_text_from_docx(file_obj) -> str:
    try:
        data = file_obj.read()
        file_obj.seek(0)
        f = io.BytesIO(data)
        doc = docx.Document(f)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paras).strip()
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def load_text_from_txt(file_obj) -> str:
    try:
        data = file_obj.read()
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8", errors="ignore")
            except:
                data = data.decode("latin-1", errors="ignore")
        return str(data).strip()
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

def load_document(file) -> str:
    if not file:
        return ""
    name = (file.name or "").lower()
    
    if name.endswith(".pdf"):
        return load_text_from_pdf(file)
    elif name.endswith(".docx"):
        return load_text_from_docx(file)
    elif name.endswith(".txt"):
        return load_text_from_txt(file)
    else:
        return "Unsupported file format"

def get_text_from_inputs(file, text):
    file_text = load_document(file) if file else ""
    user_text = (text or "").strip()
    
    if file_text and not user_text:
        return file_text
    elif user_text and not file_text:
        return user_text
    elif file_text and user_text:
        return file_text if len(file_text) > len(user_text) else user_text
    else:
        return ""

# -------------------------
# CLAUSE PROCESSING
# -------------------------
CLAUSE_SPLIT_REGEX = re.compile(r"(?:(?:^\s*\d+(?:\.\d+)*[.)]\s+)|(?:^\s*[•\-*]\s+)|(?:\n\s*\n))", re.MULTILINE)

def split_into_clauses(text: str, min_len: int = 20) -> List[str]:
    if not text or not text.strip():
        return []
    
    # First try splitting by common clause patterns
    parts = re.split(CLAUSE_SPLIT_REGEX, text)
    
    # If that doesn't work well, try sentence splitting
    if len(parts) < 2:
        parts = re.split(r"(?<=[.;!?])\s+(?=[A-Z])", text)
    
    clauses = [p.strip() for p in parts if p and len(p.strip()) >= min_len]
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in clauses:
        # Simple normalization for comparison
        key = re.sub(r'\s+', ' ', c.lower()).strip()
        if key and key not in seen and len(c) >= min_len:
            seen.add(key)
            unique.append(c)
    
    return unique

# -------------------------
# FAST CLAUSE SIMPLIFICATION
# -------------------------
def simplify_clause_fast(clause: str) -> str:
    if not clause.strip():
        return "Please provide a clause to simplify."
    
    # Quick validation for very short clauses
    if len(clause.strip()) < 10:
        return "Clause is too short for meaningful simplification."
    
    # Limit clause length for faster processing
    processed_clause = clause[:1500]  # Process only first 1500 chars
    
    system_prompt = """You are a legal assistant that rewrites complex legal clauses into plain, understandable English. 
    Be concise and focus on the main points. Keep responses under 200 words."""
    
    user_prompt = f"""Rewrite this legal clause in simple English. Focus on the key obligations and rights:

{processed_clause}

Provide a clear, simple explanation:"""

    start_time = time.time()
    result = llm_generate_optimized(
        system_prompt, 
        user_prompt, 
        max_new_tokens=200,  # Reduced from 400
        temperature=0.4
    )
    end_time = time.time()
    
    st.sidebar.info(f"Simplification took {end_time - start_time:.1f} seconds")
    
    return result

def simplify_clause_with_progress(clause: str) -> str:
    """Simplification with progress indicators"""
    if not clause.strip():
        return "Please provide a clause to simplify."
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing simplification...")
    progress_bar.progress(10)
    time.sleep(0.5)
    
    # Check if model is available
    if model is None:
        progress_bar.progress(100)
        status_text.text("Using basic simplification (model not available)")
        return "Model not available. Please check if the model loaded correctly."
    
    status_text.text("Analyzing legal language...")
    progress_bar.progress(30)
    time.sleep(0.5)
    
    status_text.text("Generating plain English version...")
    progress_bar.progress(60)
    
    # Use the optimized LLM call
    result = simplify_clause_fast(clause)
    
    progress_bar.progress(90)
    status_text.text("Finalizing output...")
    time.sleep(0.5)
    
    progress_bar.progress(100)
    status_text.text("Simplification complete!")
    time.sleep(1)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return result

def simplify_clause(clause: str) -> str:
    """Main simplification function"""
    return simplify_clause_with_progress(clause)

def ner_entities(text: str) -> Dict[str, List[str]]:
    if not text or not text.strip():
        return {}
    
    if nlp is None:
        return {"ERROR": ["spaCy model not available. Please install en_core_web_sm"]}
    
    try:
        # Process in chunks if text is too long
        if len(text) > 1000000:  # ~1MB limit
            text = text[:1000000]
            
        doc = nlp(text)
        out: Dict[str, List[str]] = {}
        
        for ent in doc.ents:
            out.setdefault(ent.label_, []).append(ent.text)
        
        # Remove duplicates and sort
        out = {k: sorted(set(v)) for k, v in out.items()}
        return out
    except Exception as e:
        return {"ERROR": [f"NER processing error: {str(e)}"]}

def extract_clauses(text: str) -> List[str]:
    return split_into_clauses(text)

# -------------------------
# DOCUMENT CLASSIFICATION
# -------------------------
DOC_TYPES = [
    "Non-Disclosure Agreement (NDA)",
    "Lease Agreement",
    "Employment Contract",
    "Service Agreement",
    "Sales Agreement",
    "Consulting Agreement",
    "End User License Agreement (EULA)",
    "Terms of Service",
    "Partnership Agreement",
    "Loan Agreement"
]

def classify_document(text: str) -> str:
    if not text or not text.strip():
        return "No text provided for classification"
    
    system_prompt = """You are a legal document classification expert. Analyze the provided text and determine the most appropriate document type from the given list."""
    
    labels = "\n".join(f"- {t}" for t in DOC_TYPES)
    user_prompt = f"""Classify the following legal document into one of these types:

Available types:
{labels}

Document text (first 3000 characters):
{text[:3000]}

Provide only the most appropriate document type from the list above."""

    resp = llm_generate_optimized(system_prompt, user_prompt, max_new_tokens=100)
    
    # Simple matching as fallback
    resp_lower = resp.lower()
    text_lower = text.lower()
    
    for doc_type in DOC_TYPES:
        if any(keyword in resp_lower for keyword in doc_type.lower().split()):
            return doc_type
    
    # If no match from LLM, try keyword matching
    if "confidential" in text_lower or "non-disclosure" in text_lower or "nda" in text_lower:
        return "Non-Disclosure Agreement (NDA)"
    elif "lease" in text_lower or "tenant" in text_lower or "landlord" in text_lower:
        return "Lease Agreement"
    elif "employment" in text_lower or "employee" in text_lower or "employer" in text_lower:
        return "Employment Contract"
    elif "service" in text_lower and "agreement" in text_lower:
        return "Service Agreement"
    elif "sale" in text_lower or "purchase" in text_lower:
        return "Sales Agreement"
    elif "consulting" in text_lower:
        return "Consulting Agreement"
    elif "eula" in text_lower or "end user" in text_lower:
        return "End User License Agreement (EULA)"
    elif "terms of service" in text_lower or "terms and conditions" in text_lower:
        return "Terms of Service"
    
    return "Unknown Document Type"

# -------------------------
# OPTIMIZED UI
# -------------------------

st.title("ClauseWise – Granite 3.2 (2B) Legal Assistant")
st.markdown("Upload a PDF/DOCX/TXT or paste text below. Tabs provide different legal analysis tools.")

with st.sidebar:
    st.header("Document Input")
    uploaded_file = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf","docx","txt"])
    pasted_text = st.text_area("Or paste text here", height=200, placeholder="Paste your legal text here...")
    
    # Performance info
    st.header("Performance Tips")
    st.info("""
    - Keep clauses under 1500 characters for faster processing
    - Use specific clauses rather than entire documents
    - Model loads faster on GPU (CUDA)
    """)
    
    if uploaded_file:
        st.info(f"Uploaded: {uploaded_file.name}")
    if pasted_text:
        st.info("Text input received")

# Get text data
text_data = get_text_from_inputs(uploaded_file, pasted_text)

# Show text preview with length info
if text_data and text_data not in ["", "Unsupported file format"]:
    with st.expander(f"Preview Extracted Text ({len(text_data)} characters)", expanded=False):
        st.text_area("Text Preview", text_data[:1500] + ("..." if len(text_data) > 1500 else ""), height=200, key="preview")
        if len(text_data) > 1500:
            st.warning(f"Document is large ({len(text_data)} characters). For faster processing, consider analyzing specific clauses.")
else:
    st.warning("Please upload a document or paste text to get started")

# Create only the core working tabs
tabs = st.tabs([
    "🚀 Clause Simplification", 
    "🔍 Named Entity Recognition", 
    "📑 Clause Extraction",
    "📊 Document Classification"
])

# Tab 1: OPTIMIZED Clause Simplification
with tabs[0]:
    st.header("Clause Simplification")
    st.markdown("Convert complex legal language into plain English")
    
    # Smart input selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        clause_input = st.text_area(
            "Enter specific clause to simplify:", 
            height=120, 
            placeholder="Paste a complex legal clause here (recommended: under 1500 characters)...",
            key="simplify_input"
        )
    
    with col2:
        st.markdown("### Options")
        use_document_text = st.checkbox(
            "Use uploaded document", 
            value=not bool(clause_input.strip()),
            help="Use the entire uploaded document for simplification"
        )
    
    # Character count and warnings
    if clause_input.strip():
        char_count = len(clause_input)
        if char_count > 1500:
            st.warning(f"Clause is long ({char_count} characters). This may take longer to process.")
        else:
            st.info(f"Clause length: {char_count} characters")
    
    if st.button("Simplify Clause", key="simplify", type="primary", use_container_width=True):
        if use_document_text and text_data and text_data not in ["", "Unsupported file format"]:
            if len(text_data) > 2000:
                st.warning("Document is large. Simplifying first 1500 characters for speed.")
                target = text_data[:1500]
            else:
                target = text_data
            source = "uploaded document"
        elif clause_input.strip():
            target = clause_input
            source = "text input"
        else:
            st.error("Please provide a clause to simplify either through text input or document upload")
            target = None
            
        if target:
            result = simplify_clause_with_progress(target)
            
            st.subheader("Simplified Output")
            
            # Display result in a nice container
            with st.container():
                st.success("✅ Simplification Complete")
                st.text_area(
                    "Plain English Version", 
                    result, 
                    height=300,
                    key="result_output"
                )
                
                # Add some metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", f"{len(target)} chars")
                with col2:
                    st.metric("Simplified Length", f"{len(result)} chars")
                with col3:
                    reduction = max(0, len(target) - len(result))
                    st.metric("Reduction", f"{reduction} chars")

# Tab 2: Named Entity Recognition
with tabs[1]:
    st.header("Named Entity Recognition")
    st.markdown("Identify people, organizations, dates, and other entities in your legal documents")
    
    if st.button("Extract Entities", key="ner", type="primary"):
        if text_data and text_data not in ["", "Unsupported file format"]:
            with st.spinner("Analyzing entities..."):
                entities = ner_entities(text_data)
                st.subheader("Extracted Entities")
                st.json(entities)
        else:
            st.error("Please upload a document or paste text first")

# Tab 3: Clause Extraction
with tabs[2]:
    st.header("Clause Extraction")
    st.markdown("Automatically identify and extract individual clauses from legal documents")
    
    if st.button("Extract Clauses", key="extract", type="primary"):
        if text_data and text_data not in ["", "Unsupported file format"]:
            with st.spinner("Extracting clauses..."):
                clauses = extract_clauses(text_data)
                st.subheader(f"Found {len(clauses)} Clauses")
                
                if clauses:
                    for i, clause in enumerate(clauses, 1):
                        with st.expander(f"Clause {i} (Length: {len(clause)} chars)"):
                            st.text(clause)
                else:
                    st.info("No clauses could be automatically extracted. Try using the full text in other analysis tools.")
        else:
            st.error("Please upload a document or paste text first")

# Tab 4: Document Classification
with tabs[3]:
    st.header("Document Classification")
    st.markdown("Automatically identify the type of legal document")
    
    if st.button("Classify Document", key="classify", type="primary"):
        if text_data and text_data not in ["", "Unsupported file format"]:
            with st.spinner("Analyzing document type..."):
                doc_type = classify_document(text_data)
                st.subheader("Document Classification")
                st.info(f"**Predicted Document Type:** {doc_type}")
        else:
            st.error("Please upload a document or paste text first")

st.markdown("---")
st.caption("ClauseWise Legal Assistant - Powered by Granite 3.2 2B Model | Core Features Only")
