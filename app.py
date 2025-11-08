import re
from pypdf import PdfReader
import docx
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------------------------------------------
# ✅ Extract text from files
# -------------------------------------------------------------
def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])

    return ""

# -------------------------------------------------------------
# ✅ Clause splitting (very stable)
# -------------------------------------------------------------
def split_into_clauses(text):
    parts = re.split(r"\n\s*\d+\.\s+|\n\s*-\s+|\n{2,}", text)
    clauses = [p.strip() for p in parts if len(p.strip()) > 40]
    return clauses[:15]  # keep app fast for HF cpu

# -------------------------------------------------------------
# ✅ Clause simplifier (dummy logic)
# -------------------------------------------------------------
def simplify_clause(clause, mode):
    clause = clause.strip()

    if mode == "eli5":
        return f"This clause basically means: {clause[:120]}..."

    if mode == "simple":
        return f"Simplified meaning: {clause[:150]}..."

    if mode == "pro":
        return f"Professional interpretation: {clause}"

    return clause

# -------------------------------------------------------------
# ✅ Chat with DistilGPT2 (working HF CPU-safe chat)
# -------------------------------------------------------------
def chat_with_model(model, tokenizer, prompt, history):
    full_prompt = ""

    # Build few-shot conversation context (last 6 messages)
    for role, text in history[-6:]:
        full_prompt += f"{role}: {text}\n"

    full_prompt += f"User: {prompt}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=200,
            num_beams=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean only last assistant message
    if "AI:" in result:
        result = result.split("AI:")[-1].strip()

    return result
