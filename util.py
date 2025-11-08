
from pypdf import PdfReader
import re

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        text += p.extract_text() + "\n"
    return text

def split_into_clauses(text):
    return [c.strip() for c in text.split("\n") if len(c.strip()) > 20]

def simplify_clause(text, mode):
    if mode == "ELI5":
        return "This clause means in simple child language: " + text[:150]
    if mode == "Simple":
        return "In simple terms: " + text[:200]
    return "Professional explanation: " + text[:300]

def extract_entities(text):
    persons = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
    companies = re.findall(r"[A-Z][A-Za-z]+ Pvt Ltd", text)
    return {"persons": list(set(persons)), "companies": list(set(companies))}

def generate_risks(text):
    return [
        "Confidentiality period unclear.",
        "No mention of exclusions.",
        "No termination conditions.",
        "Overly broad definitions.",
        "No liability limitations."
    ]

def fairness_score(text):
    return (40, 60)

def alternative_clauses(original):
    return "A more fair version of this clause is recommended here."
