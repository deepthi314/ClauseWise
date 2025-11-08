def extract_entities(text):
    return {
        "parties": ["Party A", "Party B"],
        "dates": ["January 1, 2024"],
        "amounts": ["$10,000"]
    }

def extract_clauses(text):
    return ["Confidentiality", "Term", "IP Ownership"]

def risk_analysis(text):
    return [
        "Very broad confidentiality scope.",
        "No defined liability cap.",
        "Unilateral termination clause.",
        "Unlimited data retention.",
        "Missing dispute resolution mechanism."
    ]

def fairness_score(text):
    user = 35
    company = 65
    return (50, user, company)

def alternative_clauses():
    return [
        "A mutual confidentiality version is commonly used.",
        "Liability caps between both parties are standard.",
        "Shorter NDA duration (1–2 years) is common."
    ]
