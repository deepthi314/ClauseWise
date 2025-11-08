import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from multilingual import UI_TEXT, translate_text
from util import extract_text, split_into_clauses, simplify_clause, chat_with_model

# ---------------------------------------------------
# ✅ PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ClauseWise – NDA Assistant",
    layout="wide"
)

st.markdown(
    "<h2 style='text-align:center;'>ClauseWise – Multilingual NDA Legal Assistant</h2>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# ✅ LANGUAGE HANDLING
# ---------------------------------------------------
LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "ಕನ್ನಡ (Kannada)": "kn"
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

selected_label = st.selectbox("🌐 Language", list(LANGUAGES.keys()))
st.session_state.lang = LANGUAGES[selected_label]
T = {k: v[st.session_state.lang] for k, v in UI_TEXT.items()}

# ---------------------------------------------------
# ✅ LOAD CHAT MODEL (DistilGPT2 – HF SAFE)
# ---------------------------------------------------
@st.cache_resource
def load_chat_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


model, tokenizer = load_chat_model()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------
# ✅ FILE UPLOAD
# ---------------------------------------------------
st.subheader(T["upload_title"])
uploaded = st.file_uploader(T["upload_instruction"], type=["pdf", "txt", "docx"])

if uploaded:
    st.info("⏳ Reading file...")
    text = extract_text(uploaded)

    # ---------------------------------------------------
    # ✅ STRICT NDA DETECTION
    # ---------------------------------------------------
    NDA_KEYWORDS = [
        "non-disclosure", "non disclosure", "nda",
        "confidential information", "disclosing party",
        "receiving party", "confidentiality",
        "confidential materials", "protected information"
    ]

    if len(text) < 50 or not any(k.lower() in text.lower() for k in NDA_KEYWORDS):
        st.error(T["error_not_nda"])
        st.stop()

    st.success(T["success_nda"])

    # ---------------------------------------------------
    # ✅ ANALYSIS TABS
    # ---------------------------------------------------
    st.subheader(T["analysis_title"])
    tabs = st.tabs([
        T["tab_clauses"],
        T["tab_risks"],
        T["tab_fairness"],
        T["tab_entities"],
        T["tab_alternatives"],
        T["tab_chat"],
    ])

    # ===================================================
    # ✅ TAB 1 — CLAUSE SIMPLIFICATION
    # ===================================================
    with tabs[0]:
        st.markdown(f"### {T['clause_simplify']}")

        mode = st.radio(
            T["choose_mode"],
            [("eli5", T["eli5"]), ("simple", T["simple"]), ("pro", T["pro"])],
            format_func=lambda x: x[1]
        )[0]

        clauses = split_into_clauses(text)

        for i, c in enumerate(clauses):
            with st.expander(f"Clause {i+1}"):
                st.write("**Original:**")
                st.write(c)

                st.write("**Explanation:**")
                st.write(simplify_clause(c, mode))

    # ===================================================
    # ✅ TAB 2 — RISK ANALYSIS
    # ===================================================
    with tabs[1]:
        st.markdown(f"### {T['risk_title']}")

        # Simple risk detector
        RISK_PATTERNS = {
            "Broad confidentiality definition": ["broad", "all information", "any information"],
            "Unlimited liability": ["unlimited", "full liability", "all damages"],
            "One-sided obligations": ["shall not", "only the receiving party"],
            "Long duration (>5 years)": ["5 years", "7 years", "perpetual"],
            "No termination rights": ["cannot terminate", "no termination"]
        }

        risks_found = []

        for clause in clauses:
            lower_c = clause.lower()
            for risk_label, kws in RISK_PATTERNS.items():
                if any(k in lower_c for k in kws):
                    risks_found.append(risk_label)

        risks_found = list(dict.fromkeys(risks_found))[:5]  # top 5

        if not risks_found:
            st.success("✅ No major risks detected.")
        else:
            for r in risks_found:
                st.error("⚠️ " + r)

    # ===================================================
    # ✅ TAB 3 — FAIRNESS METER
    # ===================================================
    with tabs[2]:
        st.markdown(f"### {T['fairness_title']}")

        fairness_score = max(20, min(90, 50 - len(risks_found) * 7))

        st.write(f"**{T['your_position']}:** {fairness_score}%")
        st.write(f"**{T['company_position']}:** {100 - fairness_score}%")

        st.progress(fairness_score / 100)

    # ===================================================
    # ✅ TAB 4 — ENTITIES
    # ===================================================
    with tabs[3]:
        st.markdown(f"### {T['entities_title']}")

        parties = []
        dates = []
        money = []

        import re

        for clause in clauses:
            if "party" in clause.lower():
                parties.append(clause[:80] + "...")

            money.extend(re.findall(r"\$[\d,]+", clause))
            dates.extend(re.findall(r"\b(?:\d{1,2}\/\d{1,2}\/\d{2,4}|20\d{2})\b", clause))

        st.write("**Parties:**", list(set(parties)))
        st.write("**Dates:**", list(set(dates)))
        st.write("**Amounts:**", list(set(money)))

    # ===================================================
    # ✅ TAB 5 — ALTERNATIVE CLAUSES
    # ===================================================
    with tabs[4]:
        st.markdown(f"### {T['alt_title']}")

        ALTS = [
            "A mutual confidentiality clause where both parties share equal protection.",
            "A time-limited confidentiality period of 2–3 years.",
            "Liability capped at a fixed reasonable amount."
        ]

        for alt in ALTS:
            st.info(alt)

    # ===================================================
    # ✅ TAB 6 — LEGAL CHAT ASSISTANT
    # ===================================================
    with tabs[5]:
        st.markdown(f"### {T['chat_title']}")

        user_input = st.text_input(T["chat_placeholder"])

        if user_input:
            reply = chat_with_model(model, tokenizer, user_input, st.session_state.chat_history)

            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("AI", reply))

        for role, msg in st.session_state.chat_history[-10:]:
            if role == "User":
                st.markdown(f"🧑 **You:** {msg}")
            else:
                st.markdown(f"🤖 **ClauseWise:** {msg}")
