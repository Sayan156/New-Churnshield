"""
ChurnShield — Bank Customer Attrition Predictor (Enhanced + SHAP Edition)
Run:  streamlit run app.py
Deps: pip install streamlit scikit-learn xgboost pandas numpy plotly shap joblib
Place stacking_model.pkl and the .streamlit/ folder alongside app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnShield · Bank Attrition Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — decorative only; all widget colours come from .streamlit/config.toml
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
:root {
    --bg:       #060810;
    --surface:  #0c0f1a;
    --surface2: #111425;
    --border:   #1a2235;
    --border2:  #243050;
    --gold:     #e2b96a;
    --gold-dim: #a07c3a;
    --cyan:     #4fc8e8;
    --danger:   #ff4d6d;
    --safe:     #2ddbb0;
    --text:     #dde6f0;
    --muted:    #4a5a72;
    --muted2:   #6b7f96;
    --r:        14px;
    --rs:       8px;
}

/* ── Nuke every possible Streamlit container background ── */
html, body,
.stApp,
.stApp > *,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > *,
[data-testid="stHeader"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stMarkdown"],
[data-testid="stMarkdownContainer"],
[data-testid="stElementContainer"],
[data-testid="element-container"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
div.element-container,
div.stPlotlyChart,
section.main,
div.main {
    background-color: var(--bg) !important;
}

/* Extra catch for any stray white divs injected by React */
.stApp div:not([class*="result-"]):not([class*="hero"]):not([class*="card"]):not([class*="shap"]):not([class*="driver"]):not([class*="stat"]):not([class*="summary"]):not([class*="eng"]):not([class*="sec"]):not([class*="footer"]) {
    background-color: transparent !important;
}

#MainMenu, footer, header { visibility: hidden; }
.main .block-container {
    padding: 2rem 3.5rem 5rem !important;
    max-width: 1380px !important;
}

/* fonts globally */
html, body, p, span, div, label { font-family: 'Syne', sans-serif !important; }

/* ── HERO ── */
.hero-wrap {
    position: relative; padding: 2.8rem 3rem;
    background: linear-gradient(135deg, #0c0f1a, #0f1526, #0a1020) !important;
    border: 1px solid var(--border2); border-radius: var(--r);
    margin-bottom: 2.5rem; overflow: hidden;
}
.hero-wrap::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);
}
.hero-glow {
    position:absolute; top:-60px; right:-60px; width:320px; height:320px;
    background:radial-gradient(circle,rgba(226,185,106,0.1),transparent 65%);
    pointer-events:none;
}
.hero-eyebrow { font-family:'JetBrains Mono',monospace!important; font-size:.6rem; letter-spacing:.25em; text-transform:uppercase; color:var(--gold-dim)!important; margin-bottom:.6rem; }
.hero-title   { font-family:'Cormorant Garamond',serif!important; font-size:3.4rem; font-weight:700; color:var(--gold)!important; line-height:1; margin:0 0 .4rem; text-shadow:0 0 40px rgba(226,185,106,.3); }
.hero-subtitle{ font-family:'JetBrains Mono',monospace!important; font-size:.7rem; color:var(--muted2)!important; letter-spacing:.06em; }
.hero-stats   { display:flex; gap:1rem; margin-top:1.8rem; flex-wrap:wrap; }
.hero-stat    { background:rgba(226,185,106,.06); border:1px solid rgba(226,185,106,.15); border-radius:var(--rs); padding:.55rem 1.1rem; font-family:'JetBrains Mono',monospace!important; font-size:.65rem; color:var(--gold-dim)!important; }
.hero-stat strong { color:var(--gold)!important; display:block; font-size:.85rem; margin-top:.1rem; }

/* ── SECTION LABEL ── */
.sec-label {
    display:flex; align-items:center; gap:.6rem;
    font-family:'JetBrains Mono',monospace!important; font-size:.58rem;
    letter-spacing:.22em; text-transform:uppercase; color:var(--gold)!important;
    margin-bottom:.9rem; margin-top:.4rem;
}
.sec-label::before { content:''; width:18px; height:1px; background:var(--gold); flex-shrink:0; }
.sec-label::after  { content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border2),transparent); }

/* ── CARD ── */
.card {
    background:var(--surface)!important; border:1px solid var(--border);
    border-radius:var(--r); padding:1.6rem 1.8rem; margin-bottom:1.2rem; position:relative;
}
.card-accent {
    position:absolute; top:0; left:0; width:3px; height:100%;
    background:linear-gradient(to bottom,var(--gold),transparent);
    border-radius:var(--r) 0 0 var(--r);
}

/* ── INPUT LABELS ── */
div[data-testid="stNumberInput"] > label,
div[data-testid="stSelectbox"]   > label,
div[data-testid="stSlider"]      > label {
    font-family:'JetBrains Mono',monospace!important;
    font-size:.65rem!important; color:var(--muted2)!important;
    text-transform:uppercase!important; letter-spacing:.09em!important;
}

/* ── BUTTON ── */
div.stButton > button {
    width:100%!important; padding:1.05rem 0!important;
    background:linear-gradient(135deg,#a07c3a,#e2b96a,#c49845)!important;
    color:#060810!important; font-family:'Syne',sans-serif!important;
    font-size:.82rem!important; font-weight:800!important;
    letter-spacing:.18em!important; text-transform:uppercase!important;
    border:none!important; border-radius:var(--rs)!important;
    box-shadow:0 4px 28px rgba(226,185,106,.22)!important;
    transition:transform .15s ease!important;
}
div.stButton > button:hover { transform:translateY(-2px)!important; }

/* ── ENGAGEMENT CHIP ── */
.eng-chip {
    display:inline-flex; align-items:center; gap:.6rem;
    background:rgba(79,200,232,.07); border:1px solid rgba(79,200,232,.2);
    border-radius:var(--rs); padding:.55rem 1.1rem;
    font-family:'JetBrains Mono',monospace!important; font-size:.72rem;
    color:var(--cyan)!important; margin-top:.7rem; width:100%; box-sizing:border-box;
}
.info-note { font-family:'JetBrains Mono',monospace!important; font-size:.6rem; color:var(--muted)!important; margin-top:.25rem; padding-left:.3rem; }

/* ── RESULT BOX ── */
.result-outer { border-radius:var(--r); padding:2rem 1.8rem 1.8rem; text-align:center; position:relative; overflow:hidden; }
.result-outer.danger { background:linear-gradient(160deg,#120a0e,#0d0814)!important; border:1px solid rgba(255,77,109,.35); box-shadow:0 0 60px rgba(255,77,109,.12); }
.result-outer.safe   { background:linear-gradient(160deg,#08120f,#071210)!important; border:1px solid rgba(45,219,176,.35);  box-shadow:0 0 60px rgba(45,219,176,.1); }
.result-eyebrow { font-family:'JetBrains Mono',monospace!important; font-size:.58rem; letter-spacing:.22em; text-transform:uppercase; color:var(--muted2)!important; margin-bottom:.5rem; }
.result-verdict { font-family:'Cormorant Garamond',serif!important; font-size:2.8rem; font-weight:700; line-height:1; margin:.3rem 0 .2rem; }
.result-verdict.danger { color:var(--danger)!important; text-shadow:0 0 30px rgba(255,77,109,.4); }
.result-verdict.safe   { color:var(--safe)!important;   text-shadow:0 0 30px rgba(45,219,176,.3); }
.result-prob { font-family:'JetBrains Mono',monospace!important; font-size:.72rem; color:var(--muted2)!important; margin-top:.4rem; }
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:.6rem; margin-top:1.2rem; }
.stat-tile { background:rgba(255,255,255,.03)!important; border:1px solid var(--border); border-radius:var(--rs); padding:.65rem .8rem; font-family:'JetBrains Mono',monospace!important; font-size:.58rem; color:var(--muted2)!important; text-transform:uppercase; letter-spacing:.08em; text-align:left; }
.stat-tile strong { display:block; font-size:1.05rem; color:var(--text)!important; margin-top:.2rem; font-family:'Syne',sans-serif!important; }

/* ── SHAP SECTION ── */
.shap-wrap      { border-radius:var(--r); overflow:hidden; margin-top:2rem; border:1px solid rgba(255,77,109,.2); background:var(--bg)!important; }
.shap-head      { display:flex; align-items:center; gap:1rem; padding:1.4rem 1.8rem; background:linear-gradient(135deg,#100814,#0c0d1a)!important; border-bottom:1px solid var(--border); }
.shap-head-title{ font-family:'Cormorant Garamond',serif!important; font-size:1.5rem; font-weight:700; color:var(--danger)!important; margin:0; text-shadow:0 0 20px rgba(255,77,109,.3); }
.shap-head-sub  { font-family:'JetBrains Mono',monospace!important; font-size:.6rem; color:var(--muted)!important; text-transform:uppercase; letter-spacing:.1em; margin-top:.2rem; }
.shap-body      { background:var(--surface)!important; padding:1.8rem; }
.shap-chart-wrap{ background:var(--surface)!important; border:1px solid rgba(255,77,109,.2); border-top:none; border-radius:0 0 var(--r) var(--r); padding:1.2rem 1.8rem 1.8rem; }

.driver-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin-bottom:0; }
.driver-tile { background:var(--surface2)!important; border:1px solid var(--border2); border-radius:var(--r); padding:1.2rem 1.3rem; text-align:center; }
.driver-rank { font-family:'JetBrains Mono',monospace!important; font-size:.55rem; color:var(--muted)!important; text-transform:uppercase; letter-spacing:.15em; }
.driver-name { font-family:'Syne',sans-serif!important; font-size:.92rem; font-weight:700; color:var(--text)!important; margin:.5rem 0 .3rem; }
.driver-val  { font-family:'JetBrains Mono',monospace!important; font-size:1.1rem; font-weight:500; }
.driver-dir  { font-family:'JetBrains Mono',monospace!important; font-size:.62rem; color:var(--muted2)!important; margin-top:.3rem; }

.summary-box   { background:rgba(255,77,109,.05)!important; border:1px solid rgba(255,77,109,.15); border-radius:var(--r); padding:1.3rem 1.6rem; margin-top:1.2rem; }
.summary-label { font-family:'JetBrains Mono',monospace!important; font-size:.6rem; color:var(--danger)!important; text-transform:uppercase; letter-spacing:.15em; margin-bottom:.6rem; }
.summary-text  { font-family:'Syne',sans-serif!important; font-size:.82rem; color:var(--muted2)!important; line-height:1.7; }

hr { border-color:var(--border)!important; margin:2rem 0!important; }
.footer { text-align:center; padding:2.5rem 0 1rem; font-family:'JetBrains Mono',monospace!important; font-size:.58rem; color:#1e2d4a!important; letter-spacing:.1em; text-transform:uppercase; }


/* ── Fix broken expander arrow icon ── */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] {
    display: none !important;
}

[data-testid="stExpander"] summary::before {
    content: '▶';
    font-family: 'Syne', sans-serif !important;
    font-size: 0.6rem;
    color: var(--gold-dim);
    margin-right: 0.5rem;
    transition: transform 0.2s ease;
}

[data-testid="stExpander"][open] summary::before,
[data-testid="stExpander"] summary[aria-expanded="true"]::before {
    content: '▼';
}
/* ── Nuclear fix for broken expander icon ── */
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary > div > span:first-child,
[data-testid="stExpander"] summary span:has(+ div),
[data-testid="stExpander"] summary svg {
    font-size: 0 !important;
    color: transparent !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    display: none !important;
}

[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.6rem !important;
}

[data-testid="stExpander"] summary::before {
    content: '▶';
    font-family: 'Syne', sans-serif !important;
    font-size: 0.55rem !important;
    color: var(--gold-dim) !important;
    flex-shrink: 0 !important;
    transition: transform 0.2s ease !important;
}

[data-testid="stExpander"] details[open] summary::before {
    transform: rotate(90deg) !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_ORDER = [
    "Customer_Age", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "engagement_score",
    "Education_Level_Doctorate", "Education_Level_Graduate",
    "Education_Level_High School", "Education_Level_Post-Graduate",
    "Education_Level_Uneducated", "Education_Level_Unknown",
    "Income_Category_$40K - $60K", "Income_Category_$60K - $80K",
    "Income_Category_$80K - $120K", "Income_Category_Less than $40K",
    "Income_Category_Unknown", "Gender_M",
    "Marital_Status_Married", "Marital_Status_Single", "Marital_Status_Unknown",
    "Card_Category_Gold", "Card_Category_Platinum", "Card_Category_Silver",
]

FEATURE_LABELS = {
    "Customer_Age": "Customer Age", "Months_on_book": "Months on Book",
    "Total_Relationship_Count": "Products Held", "Months_Inactive_12_mon": "Months Inactive",
    "Contacts_Count_12_mon": "Contact Count", "Total_Revolving_Bal": "Revolving Balance",
    "Total_Amt_Chng_Q4_Q1": "Amt Change Q4-Q1", "Total_Trans_Amt": "Transaction Amount",
    "Total_Trans_Ct": "Transaction Count", "Total_Ct_Chng_Q4_Q1": "Count Change Q4-Q1",
    "Avg_Utilization_Ratio": "Utilization Ratio", "engagement_score": "Engagement Score",
    "Education_Level_Doctorate": "Edu: Doctorate", "Education_Level_Graduate": "Edu: Graduate",
    "Education_Level_High School": "Edu: High School", "Education_Level_Post-Graduate": "Edu: Post-Grad",
    "Education_Level_Uneducated": "Edu: Uneducated", "Education_Level_Unknown": "Edu: Unknown",
    "Income_Category_$40K - $60K": "Income $40K-$60K", "Income_Category_$60K - $80K": "Income $60K-$80K",
    "Income_Category_$80K - $120K": "Income $80K-$120K", "Income_Category_Less than $40K": "Income <$40K",
    "Income_Category_Unknown": "Income: Unknown", "Gender_M": "Gender: Male",
    "Marital_Status_Married": "Marital: Married", "Marital_Status_Single": "Marital: Single",
    "Marital_Status_Unknown": "Marital: Unknown", "Card_Category_Gold": "Card: Gold",
    "Card_Category_Platinum": "Card: Platinum", "Card_Category_Silver": "Card: Silver",
}

_BG = {
    "Customer_Age": 46.0, "Months_on_book": 36.0, "Total_Relationship_Count": 4.0,
    "Months_Inactive_12_mon": 2.0, "Contacts_Count_12_mon": 3.0,
    "Total_Revolving_Bal": 1200.0, "Total_Amt_Chng_Q4_Q1": 0.760,
    "Total_Trans_Amt": 4404.0, "Total_Trans_Ct": 68.0,
    "Total_Ct_Chng_Q4_Q1": 0.710, "Avg_Utilization_Ratio": 0.274,
    "engagement_score": 18.65,
}
for _f in FEATURES_ORDER:
    if _f not in _BG:
        _BG[_f] = 0.0
BACKGROUND_DF = pd.DataFrame([_BG])[FEATURES_ORDER]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("models/stacking_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("stacking_model.pkl not found. Place it in the same folder as app.py.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: KernelExplainer is intentionally NOT cached with @st.cache_resource.
# It is a stateful object (nsamplesRun / nsamplesAdded etc.).  Caching it
# causes a ValueError when Streamlit reruns the script mid-computation and
# the shared object's internal counters are left in an inconsistent state.
# Creating a fresh instance each time costs only one model forward-pass on
# the single-row background DataFrame, which is negligible.
def make_explainer(_model):
    def predict_fn(x):
        df = pd.DataFrame(x, columns=FEATURES_ORDER)
        return _model.predict_proba(df)[:, 1]
    return shap.KernelExplainer(predict_fn, BACKGROUND_DF.values)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(inp):
    edu_cats  = ["Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated", "Unknown"]
    inc_cats  = ["$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K", "Unknown"]
    mar_cats  = ["Married", "Single", "Unknown"]
    card_cats = ["Gold", "Platinum", "Silver"]
    row = {
        "Customer_Age":             inp["Customer_Age"],
        "Months_on_book":           inp["Months_on_book"],
        "Total_Relationship_Count": inp["Total_Relationship_Count"],
        "Months_Inactive_12_mon":   inp["Months_Inactive_12_mon"],
        "Contacts_Count_12_mon":    inp["Contacts_Count_12_mon"],
        "Total_Revolving_Bal":      inp["Total_Revolving_Bal"],
        "Total_Amt_Chng_Q4_Q1":     inp["Total_Amt_Chng_Q4_Q1"],
        "Total_Trans_Amt":          inp["Total_Trans_Amt"],
        "Total_Trans_Ct":           inp["Total_Trans_Ct"],
        "Total_Ct_Chng_Q4_Q1":      inp["Total_Ct_Chng_Q4_Q1"],
        "Avg_Utilization_Ratio":    inp["Avg_Utilization_Ratio"],
        "engagement_score":         inp["Total_Trans_Ct"] * inp["Avg_Utilization_Ratio"],
        **{f"Education_Level_{v}":  int(inp["Education_Level"] == v)  for v in edu_cats},
        **{f"Income_Category_{v}":  int(inp["Income_Category"] == v)  for v in inc_cats},
        "Gender_M":                 int(inp["Gender"] == "M"),
        **{f"Marital_Status_{v}":   int(inp["Marital_Status"] == v)   for v in mar_cats},
        **{f"Card_Category_{v}":    int(inp["Card_Category"] == v)    for v in card_cats},
    }
    return pd.DataFrame([row])[FEATURES_ORDER]


# ─────────────────────────────────────────────────────────────────────────────
# GAUGE
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(prob, verdict):
    color = "#ff4d6d" if verdict == "ATTRITED" else "#2ddbb0"
    pct   = round(prob * 100, 1)
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0,100], "tickwidth":1, "tickcolor":"#1a2235",
                     "tickfont": {"color":"#4a5a72","size":9,"family":"JetBrains Mono"}},
            "bar": {"color": color, "thickness": 0.24},
            "bgcolor": "#0c0f1a", "borderwidth": 0,
            "steps": [
                {"range":[0,40],  "color":"rgba(45,219,176,0.05)"},
                {"range":[40,70], "color":"rgba(226,185,106,0.05)"},
                {"range":[70,100],"color":"rgba(255,77,109,0.05)"},
            ],
            "threshold": {"line":{"color":color,"width":3},"thickness":0.85,"value":pct},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(t=20,b=10,l=20,r=20), font_family="JetBrains Mono",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SHAP CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_shap_chart(sv_row, feature_values, base_value, top_n=12):
    abs_sv  = np.abs(sv_row)
    top_idx = np.argsort(abs_sv)[::-1][:top_n]
    top_idx = top_idx[np.argsort(sv_row[top_idx])]
    labels  = [FEATURE_LABELS.get(FEATURES_ORDER[i], FEATURES_ORDER[i]) for i in top_idx]
    vals    = sv_row[top_idx]
    fvals   = feature_values[top_idx]
    colors  = ["#ff4d6d" if v > 0 else "#2ddbb0" for v in vals]

    def fmt(v):
        if abs(v) >= 1000: return f"{v:,.0f}"
        if abs(v) >= 1:    return f"{v:.2f}"
        return f"{v:.3f}"

    hover = ["<b>"+labels[i]+"</b><br>Value: "+fmt(fvals[i])+"<br>SHAP: "+f"{vals[i]:+.4f}" for i in range(len(labels))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertext=hover, hoverinfo="text",
        hoverlabel=dict(bgcolor="#111425",bordercolor="#243050",font=dict(family="JetBrains Mono",size=11,color="#dde6f0")),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color="#243050", width=1.5, dash="dot"))
    for i, v in enumerate(vals):
        clr = "#ff4d6d" if v > 0 else "#2ddbb0"
        fig.add_annotation(
            x=v+(0.003 if v>0 else -0.003), y=i,
            xanchor="left" if v>0 else "right", yanchor="middle", showarrow=False,
            text="<span style='font-family:JetBrains Mono;font-size:9px;color:"+clr+"'>"+f"{v:+.3f}"+"</span>",
        )
    fig.add_annotation(
        x=0, y=-0.9, xref="x", yref="paper", xanchor="center", yanchor="top", showarrow=False,
        text="<span style='font-family:JetBrains Mono;font-size:10px;color:#4a5a72'>Base churn probability: "+f"{base_value:.3f}"+"</span>",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,14,26,0.8)",
        height=max(360, top_n*34), margin=dict(t=30,b=55,l=20,r=80),
        xaxis=dict(
            title=dict(text="SHAP value  (impact on churn probability)",font=dict(family="JetBrains Mono",size=10,color="#4a5a72")),
            gridcolor="#1a2235",gridwidth=1,zeroline=False,
            tickfont=dict(family="JetBrains Mono",size=9,color="#4a5a72"),tickformat="+.3f",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(family="Syne",size=11,color="#9ab0cc"),ticklen=0),
        bargap=0.35,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-eyebrow">Financial Intelligence Platform</div>
  <div class="hero-title">ChurnShield</div>
  <div class="hero-subtitle">Bank Customer Attrition Predictor &nbsp;·&nbsp; Stacking Ensemble &nbsp;·&nbsp; SHAP Explainability</div>
  <div class="hero-stats">
    <div class="hero-stat">Model<strong>stacking_model.pkl</strong></div>
    <div class="hero-stat">Features<strong>30 inputs</strong></div>
    <div class="hero-stat">Explainability<strong>SHAP Analysis</strong></div>
    <div class="hero-stat">Derived<strong>Trans_Ct x Util_Ratio</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="sec-label">Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-accent"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    customer_age   = c1.number_input("Customer Age",   min_value=18, max_value=100, value=45)
    months_on_book = c2.number_input("Months on Book", min_value=1,  max_value=60,  value=36)
    c3, c4 = st.columns(2)
    total_relationship_count = c3.slider("Total Products Held",       1, 6, 3)
    months_inactive          = c4.slider("Months Inactive (last 12)", 0, 6, 2)
    contacts_count = st.slider("Contacts Count (last 12 months)", 0, 6, 2)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Transaction and Balance</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-accent"></div>', unsafe_allow_html=True)
    total_revolving_bal = st.number_input("Total Revolving Balance ($)", min_value=0, max_value=3000, value=800, step=50)
    c5, c6 = st.columns(2)
    total_trans_amt = c5.number_input("Total Transaction Amount ($)", min_value=0, max_value=20000, value=4500, step=100)
    total_trans_ct  = c6.number_input("Total Transaction Count",      min_value=0, max_value=150,   value=60)
    c7, c8 = st.columns(2)
    total_amt_chng = c7.number_input("Amt Change Q4 to Q1 Ratio",   min_value=0.0, max_value=3.5, value=0.75, step=0.01, format="%.3f")
    total_ct_chng  = c8.number_input("Count Change Q4 to Q1 Ratio", min_value=0.0, max_value=3.5, value=0.70, step=0.01, format="%.3f")
    avg_utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.25, step=0.01)
    eng_score = round(total_trans_ct * avg_utilization, 4)
    st.markdown(
        '<div class="eng-chip">'
        '<span style="color:#4fc8e8">engagement_score</span>'
        '&nbsp;=&nbsp;Trans_Ct x Util_Ratio&nbsp;=&nbsp;'
        '<strong style="color:#dde6f0">' + str(eng_score) + '</strong>'
        '<span style="color:#4a5a72;font-size:0.65rem;margin-left:auto">auto-computed</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="sec-label">Demographic Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-accent"></div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["F", "M"], format_func=lambda x: "Female" if x == "F" else "Male")
    education_level = st.selectbox(
        "Education Level",
        ["Doctorate","Graduate","College","High School","Post-Graduate","Uneducated","Unknown"],
        index=2,
    )
    st.markdown('<div class="info-note">Baseline: College (all dummies = 0)</div>', unsafe_allow_html=True)
    marital_status = st.selectbox("Marital Status", ["Married","Divorced","Single","Unknown"], index=1)
    st.markdown('<div class="info-note">Baseline: Divorced (all dummies = 0)</div>', unsafe_allow_html=True)
    income_category = st.selectbox(
        "Income Category",
        ["$120K +","$40K - $60K","$60K - $80K","$80K - $120K","Less than $40K","Unknown"],
        index=0,
    )
    st.markdown('<div class="info-note">Baseline: $120K+ (all dummies = 0)</div>', unsafe_allow_html=True)
    card_category = st.selectbox("Card Category", ["Blue","Gold","Platinum","Silver"], index=0)
    st.markdown('<div class="info-note">Baseline: Blue (all dummies = 0)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Prediction</div>', unsafe_allow_html=True)
    predict_clicked = st.button("Predict Churn Risk")

    if predict_clicked:
        inp = {
            "Customer_Age": customer_age, "Months_on_book": months_on_book,
            "Total_Relationship_Count": total_relationship_count,
            "Months_Inactive_12_mon": months_inactive, "Contacts_Count_12_mon": contacts_count,
            "Total_Revolving_Bal": total_revolving_bal, "Total_Amt_Chng_Q4_Q1": total_amt_chng,
            "Total_Trans_Amt": total_trans_amt, "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chng, "Avg_Utilization_Ratio": avg_utilization,
            "Education_Level": education_level, "Income_Category": income_category,
            "Gender": gender, "Marital_Status": marital_status, "Card_Category": card_category,
        }
        X_input    = build_feature_vector(inp)
        prob_churn = float(model.predict_proba(X_input)[0, 1])
        prediction = int(prob_churn >= 0.27)
        st.session_state["result"] = {
            "X_input": X_input, "prediction": prediction,
            "prob_churn": prob_churn, "eng_score": eng_score,
        }

    if "result" in st.session_state:
        r          = st.session_state["result"]
        prob_churn = r["prob_churn"]
        prediction = r["prediction"]
        verdict    = "ATTRITED" if prediction == 1 else "RETAINED"
        vc         = "danger"   if prediction == 1 else "safe"
        emoji      = "🔴"       if prediction == 1 else "🟢"
        risk_label = "High Risk"  if prob_churn >= 0.70 else ("Medium Risk" if prob_churn >= 0.27 else "Low Risk")
        risk_color = "#ff4d6d"    if prob_churn >= 0.70 else ("#e2b96a"     if prob_churn >= 0.27 else "#2ddbb0")
        eng_s      = r["eng_score"]
        churn_pct  = prob_churn * 100
        retain_pct = (1 - prob_churn) * 100

        st.plotly_chart(make_gauge(prob_churn, verdict), use_container_width=True)
        st.markdown(
            '<div class="result-outer ' + vc + '">'
            '<div class="result-eyebrow">Model Verdict</div>'
            '<div class="result-verdict ' + vc + '">' + emoji + ' ' + verdict + '</div>'
            '<div class="result-prob">Churn probability:&nbsp;<strong style="color:' + risk_color + '">' + f"{churn_pct:.1f}%" + '</strong>&nbsp;·&nbsp;' + risk_label + '</div>'
            '<div class="stat-grid">'
            '<div class="stat-tile">Churn Prob<strong style="color:' + risk_color + '">' + f"{churn_pct:.2f}%" + '</strong></div>'
            '<div class="stat-tile">Retain Prob<strong style="color:#2ddbb0">' + f"{retain_pct:.2f}%" + '</strong></div>'
            '<div class="stat-tile">Risk Tier<strong style="color:' + risk_color + '">' + risk_label + '</strong></div>'
            '<div class="stat-tile">Eng Score<strong>' + f"{eng_s:.3f}" + '</strong></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
<div class="card" style="text-align:center;padding:3rem 1.5rem;border-style:dashed;border-color:#243050">
  <div style="font-size:3rem;margin-bottom:1rem;opacity:0.5">&#127919;</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4a5a72;text-transform:uppercase;letter-spacing:0.15em">
    Fill in the details and click Predict
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP SECTION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("result", {}).get("prediction") == 1:
    r       = st.session_state["result"]
    X_input = r["X_input"]

    with st.spinner("Computing SHAP values — may take 15–30 seconds…"):
        # Always create a fresh explainer here to avoid stale internal state
        # from a previous interrupted Streamlit rerun (root cause of the
        # "could not broadcast (200,1) into (0,1)" ValueError).
        try:
            _exp     = make_explainer(model)
            raw_shap = _exp.shap_values(X_input.values, nsamples=200, silent=True)
            base_val = float(_exp.expected_value)
        except (ValueError, RuntimeError) as _shap_err:
            st.warning(f"SHAP retry after error: {_shap_err}")
            _exp     = make_explainer(model)
            raw_shap = _exp.shap_values(X_input.values, nsamples=200, silent=True)
            base_val = float(_exp.expected_value)
        sv_row = np.array(raw_shap).flatten()

    abs_sv   = np.abs(sv_row)
    top3_idx = np.argsort(abs_sv)[::-1][:3]

    driver_tiles_html = '<div class="driver-grid">'
    for rank, idx in enumerate(top3_idx):
        feat_lbl  = FEATURE_LABELS.get(FEATURES_ORDER[idx], FEATURES_ORDER[idx])
        sv        = float(sv_row[idx])
        fval      = float(X_input.iloc[0, idx])
        dir_color = "#ff4d6d" if sv > 0 else "#2ddbb0"
        arrow     = "&#8593;" if sv > 0 else "&#8595;"
        direction = "Raises churn risk" if sv > 0 else "Lowers churn risk"
        driver_tiles_html += (
            '<div class="driver-tile">'
            '<div class="driver-rank">#' + str(rank+1) + ' Driver</div>'
            '<div class="driver-name">' + feat_lbl + '</div>'
            '<div class="driver-val" style="color:' + dir_color + '">' + f"{sv:+.4f}" + '</div>'
            '<div class="driver-dir">' + arrow + ' ' + direction + ' | value: ' + f"{fval:.3g}" + '</div>'
            '</div>'
        )
    driver_tiles_html += '</div>'

    pos_pairs = [(FEATURE_LABELS.get(FEATURES_ORDER[i], FEATURES_ORDER[i]), float(sv_row[i]))
                 for i in np.argsort(sv_row)[::-1] if sv_row[i] > 0.001][:3]
    neg_pairs = [(FEATURE_LABELS.get(FEATURES_ORDER[i], FEATURES_ORDER[i]), float(sv_row[i]))
                 for i in np.argsort(sv_row) if sv_row[i] < -0.001][:2]
    pos_html = ", ".join('<span style="color:#ff4d6d;font-weight:600">'+n+'</span> (+'+f"{v:.3f}"+')'for n,v in pos_pairs) or "none identified"
    neg_html = ", ".join('<span style="color:#2ddbb0;font-weight:600">'+n+'</span> ('+f"{v:.3f}"+')'for n,v in neg_pairs)
    offset   = (" Partially offset by: " + neg_html + ".") if neg_html else ""
    prob_churn = r["prob_churn"]

    summary_html = (
        '<div class="summary-box">'
        '<div class="summary-label">&#9888; Attrition Risk Summary</div>'
        '<div class="summary-text">'
        "The model's average churn probability (base value) is "
        '<strong style="color:#e2b96a">' + f"{base_val:.1%}" + '</strong>. '
        "This customer's score was pushed upward to "
        '<strong style="color:#ff4d6d">' + f"{prob_churn:.1%}" + '</strong>, '
        'primarily driven by: ' + pos_html + '.' + offset +
        '</div></div>'
    )

    # Header + driver tiles — one single markdown call, no Streamlit widgets inside
    st.markdown(
        '<div class="shap-wrap">'
        '<div class="shap-head">'
        '<div style="font-size:1.8rem">&#128300;</div>'
        '<div>'
        '<div class="shap-head-title">Why Is This Customer At Risk?</div>'
        '<div class="shap-head-sub">SHAP Feature Attribution &nbsp;·&nbsp; Top contributing factors to churn</div>'
        '</div></div>'
        '<div class="shap-body">' + driver_tiles_html + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

  
    st.markdown('<div class="shap-chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(make_shap_chart(sv_row, X_input.values[0], base_val, top_n=12), use_container_width=True)
    st.markdown(summary_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("Inspect Feature Vector"):
    st.caption('All 30 features in order')
    debug_inp = {
        "Customer_Age": customer_age, "Months_on_book": months_on_book,
        "Total_Relationship_Count": total_relationship_count,
        "Months_Inactive_12_mon": months_inactive, "Contacts_Count_12_mon": contacts_count,
        "Total_Revolving_Bal": total_revolving_bal, "Total_Amt_Chng_Q4_Q1": total_amt_chng,
        "Total_Trans_Amt": total_trans_amt, "Total_Trans_Ct": total_trans_ct,
        "Total_Ct_Chng_Q4_Q1": total_ct_chng, "Avg_Utilization_Ratio": avg_utilization,
        "Education_Level": education_level, "Income_Category": income_category,
        "Gender": gender, "Marital_Status": marital_status, "Card_Category": card_category,
    }
    X_debug = build_feature_vector(debug_inp)
    st.dataframe(X_debug.T.rename(columns={0: "value"}), use_container_width=True, height=700)

st.markdown("""
<div class="footer">
  ChurnShield &nbsp;|&nbsp; Stacking Ensemble &nbsp;|&nbsp; 30 Features &nbsp;|&nbsp; SHAP KernelExplainer &nbsp;|&nbsp; engagement_score = Trans_Ct x Util_Ratio
</div>
""", unsafe_allow_html=True)