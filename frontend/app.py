import json
import os
import sys
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels

st.set_page_config(page_title="Nyaya Lens", page_icon="⚖️", layout="wide")

API_URL = os.environ.get("NYAYA_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = 120  # the free hosted API can take about a minute to wake up


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path):
    r = requests.get(f"{API_URL}{path}", timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


STATE_LABELS = {"delhi": "Delhi", "odisha": "Odisha", "bihar": "Bihar"}
FIELD_LABELS = {"type_name_normalized": "Case Type", "court_tier": "Court Tier", "district_name": "District",
                "female_defendant": "Defendant", "female_petitioner": "Petitioner",
                "female_adv_def": "Defendant's advocate", "female_adv_pet": "Petitioner's advocate"}
GENDER_FIELDS = ["female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet"]
FIELD_LABELS.update({"female_defendant": "Defendant: gender", "female_petitioner": "Petitioner: gender",
                     "female_adv_def": "Defendant's lawyer: gender", "female_adv_pet": "Petitioner's lawyer: gender"})

CASE_TYPE_LABELS = {
    "arbtn": "Arbitration", "arbtn cases": "Arbitration Cases", "arb": "Arbitration",
    "bail matters": "Bail Matters", "c c": "Criminal Complaint", "cc": "Criminal Complaint",
    "c r act 1954": "C.R. Act 1954 (act unconfirmed)", "c s": "Civil Suit", "cs": "Civil Suit",
    "ca": "Civil Appeal", "cbi": "CBI Case", "civ dj": "Civil matter (District Judge)",
    "civ scj": "Civil matter (Senior Civil Judge)", "civ suit": "Civil Suit",
    "clo_r": "CLO-R matter (unconfirmed)", "clor": "CLOR matter (unconfirmed)",
    "contempt petition": "Contempt Petition", "corruption cases": "Corruption Cases",
    "counter claim": "Counter Claim", "court complaint": "Court Complaint",
    "cr": "Civil Revision", "cr case": "Criminal Case", "cr cases": "Criminal Cases",
    "cr criminal revision": "Criminal Revision", "cr rev": "Criminal Revision",
    "crc": "Criminal Revision Case (best-effort)", "cs cj": "Civil Suit (Civil Judge)",
    "cs dj": "Civil Suit (District Judge)", "cs dj adj": "Civil Suit (District Judge, adjourned)",
    "cs scj": "Civil Suit (Senior Civil Judge)", "ct cases": "Court Case",
    "dar": "DAR matter (unconfirmed)", "dd": "DD matter (unconfirmed)",
    "de": "DE matter (unconfirmed)", "de crl": "DE Criminal matter (unconfirmed)",
    "dpt eq": "Deposit/Equity matter (best-effort)", "dpt eq cr": "Deposit/Equity, Civil Revision",
    "dpt eq crl": "Deposit/Equity, Criminal", "dpteq": "Deposit/Equity matter",
    "e x": "Execution Application", "ex": "Execution Application", "ep": "Election Petition",
    "esic": "ESIC Case", "ex civil": "Execution Application (Civil)",
    "ex criminal": "Execution Application (Criminal)", "ex crl": "Execution Application (Criminal)",
    "execution civil": "Execution Application (Civil)", "fact f": "FACT-F matter (unconfirmed)",
    "fsat": "FSAT matter (unconfirmed)", "gp": "Guardianship Petition",
    "ham": "Hindu Adoption & Maintenance Act case", "hindu adp": "Hindu Adoption Case",
    "hma": "Hindu Marriage Act Case", "hta": "HTA matter (unconfirmed)",
    "ida": "Industrial Disputes Act Case", "insolvency": "Insolvency Case",
    "l i d": "L.I.D. matter (unconfirmed)", "l i r": "L.I.R. matter (unconfirmed)",
    "lac": "Land Acquisition Case", "lc": "Land Acquisition Case", "lca": "Land Acquisition Case",
    "m": "Miscellaneous matter", "m c a": "Miscellaneous Civil Application",
    "m crl": "Miscellaneous Criminal matter", "m ex": "Miscellaneous Execution matter",
    "m-ex": "Miscellaneous Execution matter", "mact": "Motor Accident Claims Tribunal Case",
    "mc": "Miscellaneous Case", "mca dj": "Misc. Civil Application (District Judge)",
    "mca dj adj": "Misc. Civil Application (District Judge, adjourned)",
    "mca scj": "Misc. Civil Application (Senior Civil Judge)",
    "mcd appl": "MCD Application", "mgp": "Misc. Guardianship Petition",
    "mha": "MHA matter (unconfirmed)", "misc": "Miscellaneous", "misc cases": "Miscellaneous Cases",
    "misc civ": "Miscellaneous Civil matter", "misc cj": "Misc. matter (Civil Judge)",
    "misc crl": "Miscellaneous Criminal matter", "misc dj": "Misc. matter (District Judge)",
    "misc dj adj": "Misc. matter (District Judge, adjourned)",
    "misc dj asj": "Misc. matter (District/Addl. Sessions Judge)",
    "misc ex": "Miscellaneous Execution matter", "misc rc arc": "Misc. Rent Control matter",
    "misc scj": "Misc. matter (Senior Civil Judge)", "ml": "ML matter (unconfirmed)",
    "mpc": "MPC matter (unconfirmed)", "mt case": "Miscellaneous Tribunal Case",
    "mt cases": "Miscellaneous Tribunal Cases", "muslim law": "Muslim Personal Law Matter",
    "nia": "National Investigation Agency Case", "pc": "Probate Case (best-effort)",
    "poit": "POIT matter (unconfirmed)", "ppa": "PPA matter (unconfirmed)",
    "rc arc": "Rent Control matter (Addl. Rent Controller)", "rca": "Rent Control Appeal",
    "rca civil dj adj": "Rent Control Appeal (District Judge, adjourned)",
    "rca dj": "Rent Control Appeal (District Judge)", "rca scj": "Rent Control Appeal (Senior Civil Judge)",
    "rct arct": "Rent Control Tribunal (Additional)", "revocation": "Revocation Matter",
    "rp": "Review Petition", "rti appeal": "RTI Appeal", "s c court": "Sessions Court",
    "s cause": "Small Cause Court matter", "sc": "Sessions Case", "sma": "SMA matter (unconfirmed)",
    "small cause": "Small Cause Court matter", "succ court": "Succession Court matter",
    "succession court": "Succession Court matter", "t p (c)": "Transfer Petition (Civil)",
    "t p (crl)": "Transfer Petition (Criminal)", "t p civ": "Transfer Petition (Civil)",
    "t p crl": "Transfer Petition (Criminal)", "ta": "Transfer Application",
    "tm": "Testamentary Matter (best-effort)", "tp c": "Transfer Petition (Civil)",
    "tp civ": "Transfer Petition (Civil)", "tp crl": "Transfer Petition (Criminal)",
    "tp(c)": "Transfer Petition (Civil)", "unknown": "Not specified",
}

PURPOSE_LABELS = {
    "air-applications": "Application (type unconfirmed)", "argument": "Arguments",
    "bail": "Bail Hearing", "charge": "Framing of Charge", "compromise": "Compromise / Settlement",
    "conciliatio": "Conciliation", "conciliation": "Conciliation",
    "consideration": "Consideration of Application", "evidence": "Evidence",
    "for cancellation": "For Cancellation", "for reduction": "For Reduction",
    "formality": "Procedural Formality", "frames of notice": "Framing of Notice",
    "fresh application": "Fresh Application", "issue": "Framing of Issues",
    "judgement": "Judgement", "misc": "Miscellaneous / General Hearing", "notice": "Notice",
    "old application": "Pending Application", "order": "For Orders",
    "proper o": "Proper Order (truncated)", "proper or": "Proper Order (truncated)",
    "proper ord": "Proper Order", "proper ord.": "Proper Order", "proper orde": "Proper Order",
    "referred to lokadalat": "Referred to Lok Adalat", "statement of accused": "Statement of Accused",
    "summons": "Summons", "transfer petition": "Transfer Petition", "unknown": "Not specified",
}

def humanize_case_type(code):
    state = st.session_state.get("state", "delhi")
    if state == "delhi":
        return CASE_TYPE_LABELS.get(code) or labels.case_type_label("delhi", code)
    return labels.case_type_label(state, code)

def humanize_purpose(code):
    return PURPOSE_LABELS.get(code, code.title())

def humanize_gender_field(code):
    return labels.gender_label(code)

def humanize_feature_name(raw_name):
    prefixes = [
        ("type_name_normalized_", "Case Type", humanize_case_type),
        ("purpose_name_s_", "Purpose", humanize_purpose),
        ("court_tier_", "Court Tier", lambda x: x),
        ("district_name_", "District", lambda x: x),
        ("female_defendant_", "Defendant", humanize_gender_field),
        ("female_petitioner_", "Petitioner", humanize_gender_field),
        ("female_adv_def_", "Defendant's advocate", humanize_gender_field),
        ("female_adv_pet_", "Petitioner's advocate", humanize_gender_field),
        ("primary_act_", "Act", lambda x: x),
        ("primary_section_", "Section", lambda x: x),
    ]
    for prefix, label, fn in prefixes:
        if raw_name.startswith(prefix):
            return label, fn(raw_name[len(prefix):])
    return raw_name, ""

# --- Design system ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #EFEDE4; }
    h1, h2, h3 { font-family: 'Lora', serif !important; color: #1B2A38 !important; }

    .header-band {
        background-color: #1B2A38;
        padding: 28px 36px;
        margin: -1rem -1rem 24px -1rem;
        border-bottom: 3px solid #B08D57;
    }
    .header-band h1 {
        color: #EFEDE4 !important;
        font-size: 28px;
        margin: 0;
    }
    .header-band p {
        color: #B8BFC7;
        font-family: 'IBM Plex Sans', sans-serif;
        margin: 4px 0 0 0;
        font-size: 14px;
    }

    .panel {
        background-color: #FAF8F2;
        border: 1px solid #D8D3C4;
        padding: 24px;
        border-radius: 2px;
    }
    .panel-title {
        font-family: 'Lora', serif;
        font-size: 17px;
        color: #1B2A38;
        border-bottom: 1px solid #D8D3C4;
        padding-bottom: 10px;
        margin-bottom: 16px;
    }

    div[data-testid="stButton"] button {
        background-color: #1B2A38;
        color: #EFEDE4;
        border: none;
        border-radius: 2px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        padding: 10px 0;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #B08D57;
        color: #1B2A38;
    }

    .stamp {
        display: inline-block;
        transform: rotate(-4deg);
        border: 3px solid;
        border-radius: 6px;
        padding: 10px 28px;
        font-family: 'Lora', serif;
        font-weight: 600;
        font-size: 26px;
        margin: 8px 0 20px 0;
    }
        .factor-row, .factor-row span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        padding: 4px 0;
        border-bottom: 1px solid #E8E4D8;
        color: #1B2A38 !important;
    }
        [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] small {
        color: #6B6558 !important;
    }
    .disclaimer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #6B6558;
        margin-top: 28px;
    }
           [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] small,
    [data-testid="stMarkdownContainer"] span {
        color: #1B2A38 !important;
    }

    div[data-testid="stButton"] button p {
        color: #EFEDE4 !important;
    }
    div[data-testid="stButton"] button:hover p {
        color: #1B2A38 !important;
    }
    }
        [data-testid="stMarkdownContainer"] .header-band h1,
    [data-testid="stMarkdownContainer"] .header-band h1 * {
        color: #EFEDE4 !important;
    }
    [data-testid="stMarkdownContainer"] .header-band p {
        color: #B8BFC7 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-band">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-family:'Lora',serif; font-weight:600; color:#EFEDE4; font-size:30px; line-height:1.2;">Nyaya Lens</div>
            <p style="color:#B8BFC7 !important; margin:4px 0 0 0; font-size:14px;">Case delay risk assessment for Delhi, Odisha and Bihar district courts</p>
        </div>
        <div style="color:#8FBF9F; font-family:'IBM Plex Mono',monospace; font-size:13px;">
            &#9679; Model Ready
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    states_info = api_get("/states")
except Exception:
    st.error(f"Could not reach the API at {API_URL}. Start it locally with: uvicorn api.main:app --port 8000 "
             "(or set NYAYA_API_URL to the hosted API; the free hosted API can take about a minute to wake up).")
    st.stop()

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown("""
    <h2 style="font-size:26px; line-height:1.3;">Know delay risk<br><em>before it happens.</em></h2>
    <p style="color:#4A4536; font-size:15px; line-height:1.6; max-width:34ch;">
    Trained on public court records from Delhi, Odisha and Bihar, this model reads only what is known
    the moment a case is filed, with no case-progress data used, so it is an early signal and not hindsight.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px;">
        <div style="padding:10px 0; border-top:1px solid #D8D3C4;">
            <strong>Filing-time signal only</strong><br>
            <span style="color:#6B6558; font-size:14px;">No leakage from hearings or outcomes already known</span>
        </div>
        <div style="padding:10px 0; border-top:1px solid #D8D3C4;">
            <strong>Every prediction explained</strong><br>
            <span style="color:#6B6558; font-size:14px;">SHAP factors show what pushed the result</span>
        </div>
        <div style="padding:10px 0; border-top:1px solid #D8D3C4; border-bottom:1px solid #D8D3C4;">
            <strong>Fairness-audited</strong><br>
            <span style="color:#6B6558; font-size:14px;">Accuracy gaps between groups are measured and published, including where they are large</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel"><div class="panel-title">Case Particulars</div>', unsafe_allow_html=True)
    state = st.selectbox("State", [k for k in STATE_LABELS if k in states_info],
                         format_func=lambda k: STATE_LABELS[k], key="state")
    opts = api_get(f"/options/{state}")

    def case_label(code):
        return humanize_case_type(code)

    formats = {"type_name_normalized": case_label}
    for g in GENDER_FIELDS:
        formats[g] = humanize_gender_field
    values = {}
    for field, choices in opts["fields"].items():
        values[field] = st.selectbox(FIELD_LABELS.get(field, field), choices,
                                     format_func=formats.get(field, str), key=f"{state}_{field}")

    chosen = {}
    optional = opts.get("optional_fields") or {}
    if optional:
        with st.expander("Optional: act and section (more accurate when both are known)"):
            for f, label in [("primary_act", "Primary act"), ("primary_section", "Primary section")]:
                v = st.selectbox(label, ["(not specified)"] + optional[f], key=f"{state}_{f}")
                if v != "(not specified)":
                    chosen[f] = v
        if len(chosen) == 1:
            st.warning("Choose both the act and the section, or neither. With only one, the standard model is used.")
    predict_clicked = st.button("Assess Delay Risk", use_container_width=True)

    if predict_clicked:
        payload = {"state": state, **values}
        if len(chosen) == 2:
            payload.update(chosen)
        try:
            with st.spinner("Asking the model (the hosted API can take about a minute to wake up)..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            tier = result["risk_tier"]

            stamp_colors = {"Low": "#4A6B4E", "Medium": "#B08D57", "High": "#A44A3F"}
            color = stamp_colors[tier]
            st.markdown(
                f'<div class="stamp" style="color:{color}; border-color:{color};">{tier.upper()} RISK</div>',
                unsafe_allow_html=True,
            )
            st.write("The tier ranks this case among the state's cases: Low is the third the model expects to "
                     "finish fastest, High the third it expects to take longest.")
            st.caption(f"Model: {result['model_version']} ({result['model_variant']}). {result['tier_basis']}.")
            for w in result["warnings"]:
                st.warning(w)
            with st.expander("Predicted days (not calibrated, use the tier)"):
                st.write(f"{result['expected_days']:.0f} days")
                st.caption(result["note"])

            st.markdown("**Contributing factors**")
            st.caption("Positive makes the expected duration longer, negative makes it shorter (log-days scale)")
            for factor in result["top_contributing_factors"]:
                label, value = humanize_feature_name(factor["feature"])
                sign = "+" if factor["shap_value"] > 0 else ""
                st.markdown(
                    f'<div class="factor-row"><strong>{label}:</strong> {value} '
                    f'<span style="float:right;">{sign}{factor["shap_value"]:.3f}</span></div>',
                    unsafe_allow_html=True,
                )
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the API at {API_URL}.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="disclaimer"><strong>Not a verdict:</strong> this tool estimates how long a case is likely to take, '
    'for resource-planning purposes only. It does not predict how any case will be decided. '
    'Trained on public e-Courts (Development Data Lab) records for cases filed 2010-2013 in Delhi, Odisha and Bihar. '
    'Case type labels are matched to legal terms for Delhi only; codes marked "unconfirmed" are court-clerk '
    'shorthand without an official public source. Gender fields are inferred from names in the court records and can be wrong.</p>',
    unsafe_allow_html=True,
)
