import io, json, os, warnings
warnings.filterwarnings("ignore")
import cloudpickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
DATA_PATH = "BankChurners.csv"
BEST_KEY = "stacking_lr_meta"
BEST_LABEL = "Stacking (LR Meta)"
THRESHOLD = 0.50
PALETTE = ["#7C3AED", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899"]
PLOTLY_DARK = dict(
    paper_bgcolor="#0D0F1E", plot_bgcolor="#12152C",
    font_color="#C8D0E7", font_family="Inter",
    xaxis=dict(gridcolor="#1E2440", zerolinecolor="#1E2440"),
    yaxis=dict(gridcolor="#1E2440", zerolinecolor="#1E2440"),
)

# ───────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [
        "CLIENTNUM", "Credit_Limit", "Dependent_count", "Avg_Open_To_Buy",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
    ]
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ `{DATA_PATH}` not found. Place it in the same directory.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df["Attrition_Flag"] = df["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1})
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = df.drop("Attrition_Flag", axis=1)
    y = df["Attrition_Flag"]
    return X, y

@st.cache_resource(show_spinner=False)
def load_model(key: str):
    path = os.path.join(MODEL_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return cloudpickle.load(f)

@st.cache_data(show_spinner=False)
def get_comparison_results() -> pd.DataFrame:
    res_path = os.path.join(MODEL_DIR, "comparison_results.json")
    if os.path.exists(res_path):
        return pd.read_json(res_path)
    # Fixed trailing spaces in keys & aligned with notebook output
    return pd.DataFrame([
        {"Model": "Random Forest", "Accuracy": 0.9457, "Precision": 0.7942, "Recall": 0.8934, "F1-Score": 0.8409, "ROC-AUC": 0.9817},
        {"Model": "Gradient Boosting", "Accuracy": 0.9500, "Precision": 0.7887, "Recall": 0.9406, "F1-Score": 0.8579, "ROC-AUC": 0.9910},
        {"Model": "XGBoost", "Accuracy": 0.9645, "Precision": 0.8571, "Recall": 0.9344, "F1-Score": 0.8941, "ROC-AUC": 0.9934},
        {"Model": "CatBoost", "Accuracy": 0.9658, "Precision": 0.9324, "Recall": 0.8484 , "F1-Score": 0.8884 , "ROC-AUC": 0.9915},
        {"Model": "Stacking (LR Meta)", "Accuracy": 0.9589, "Precision": 0.8247 , "Recall": 0.9447, "F1-Score": 0.8806, "ROC-AUC": 0.9930},
        {"Model": "Stacking (XGB Meta)", "Accuracy": 0.9585 , "Precision":  0.8255, "Recall": 0.9406 , "F1-Score": 0.8793 , "ROC-AUC": 0.9926},
    ])

# ───────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER BUILDER
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_shap_explainer(_model, _X_bg: pd.DataFrame):
    bg = _X_bg.sample(n=100, random_state=42)
    def _predict(X_in):
        if not isinstance(X_in, pd.DataFrame):
            X_in = pd.DataFrame(X_in, columns=_X_bg.columns)
        return _model.predict_proba(X_in)[:, 1]
    return shap.KernelExplainer(_predict, bg, link="logit"), bg

# ───────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────
def fmt_pct(val): return f"{val:.2%}"
def risk_label(prob):
    if prob >= 0.75: return "Very High Risk"
    if prob >= 0.50: return "High Risk"
    if prob >= 0.25: return "Medium Risk"
    return "Low Risk"
def risk_color(prob):
    if prob >= 0.75: return "#EF4444"
    if prob >= 0.50: return "#F59E0B"
    if prob >= 0.25: return "#3B82F6"
    return "#10B981"

def gauge_chart(prob):
    color = risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),  # Fixed: proper percentage value
        delta={'reference': 50, 'valueformat': '.1f', 'suffix': '%'},
        number={'suffix': '%', 'font': {'size': 44, 'color': color, 'family': 'Inter'}},  # Fixed: added % suffix
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#3D4270',
                    'tickfont': {'color': '#7B82A8', 'size': 11}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#12152C',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#0B2E1B'},
                {'range': [40, 70], 'color': '#2E2010'},
                {'range': [70, 100], 'color': '#2E0B0B'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': prob * 100,
            },
        },
        title={'text': 'Churn Probability', 'font': {'size': 14, 'color': '#7B82A8'}},
    ))
    fig.update_layout(
        height=280, margin=dict(t=50, b=0, l=30, r=30),
        paper_bgcolor='#0D0F1E', font_family='Inter',
    )
    return fig

def shap_waterfall_fig(explainer, shap_values, x_row):
    fig = plt.figure(figsize=(8, 5))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0D0F1E")
    shap.waterfall_plot(shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=x_row.values, feature_names=x_row.index.values), max_display=10, show=False)
    plt.tight_layout()
    return fig

def shap_summary_plot(shap_values, X_sample):
    fig = plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0D0F1E")
    shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=12, show=False)
    plt.title("Feature Impact on Churn Probability", fontsize=14, color="#C8D0E7", pad=10)
    plt.tight_layout()
    return fig

def shap_bar_fig(shap_values, X_sample):
    fig = plt.figure(figsize=(9, 6))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0D0F1E")
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=12, show=False)
    plt.title("Mean |SHAP| — Feature Importance Ranking", fontsize=14, color="#C8D0E7", pad=10)
    plt.tight_layout()
    return fig

# ───────────────────────────────────────────────────────────────────────────────
# PAGES
# ───────────────────────────────────────────────────────────────────────────────
def page_overview(model_loaded, data_ok, X, y, df_results):
    st.markdown("## 🛡️ ChurnShield AI Dashboard")
    st.markdown("Real-time churn intelligence powered by ensemble stacking")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        status_html = f"""
        <div style="font-size:0.75rem; color:#7B82A8;">
        <div style="margin-bottom:6px;">
        {'<span style="color:#10B981">●</span>' if model_loaded else '<span style="color:#EF4444">●</span>'}
        &nbsp; Primary model { "loaded" if model_loaded else "not found" }
        </div>
        <div>
        {'<span style="color:#10B981">●</span>' if data_ok else '<span style="color:#EF4444">●</span>'}
        &nbsp; Dataset { "ready" if data_ok else "not found" }
        </div>
        </div>"""
        st.markdown(status_html, unsafe_allow_html=True)
    with c2:
        if not model_loaded:
            st.error("Run the notebook to generate `models/stacking_lr_meta.pkl`")

    if not (model_loaded and data_ok): return

    best_row = df_results[df_results["Model"] == BEST_LABEL].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_data = [
        (c1, "ROC-AUC", fmt_pct(best_row["ROC-AUC"]), "Best Model", "#A78BFA"),
        (c2, "F1-Score", fmt_pct(best_row["F1-Score"]), "Churn Class", "#22D3EE"),
        (c3, "Recall", fmt_pct(best_row["Recall"]), "Churn Recall", "#10B981"),
        (c4, "Precision", fmt_pct(best_row["Precision"]), "Churn Precision", "#F59E0B"),
        (c5, "Accuracy", fmt_pct(best_row["Accuracy"]), "Overall", "#EC4899"),
    ]
    for col, label, val, sub, color in metrics_data:
        with col:
            st.markdown(f'<div style="text-align:center; padding:10px; background:#12152C; border-radius:8px;"><div style="color:#7B82A8; font-size:0.8rem;">{label}</div><div style="font-size:1.5rem; font-weight:700; color:{color};">{val}</div><div style="color:#6B7280; font-size:0.7rem;">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Leaderboard")
    styled = df_results.copy()
    styled.insert(0, "Rank", range(1, len(styled) + 1))
    styled = styled.sort_values("Recall", ascending=False).reset_index(drop=True)
    styled["Rank"] = range(1, len(styled) + 1)
    styled.insert(1, "⭐", styled["Model"].apply(lambda m: "🏆 Best" if m == BEST_LABEL else ""))
    def highlight_best(row):
        if row["Model"] == BEST_LABEL: return ["background-color: rgba(124,58,237,0.18); font-weight:600;"] * len(row)
        return [""] * len(row)
    st.dataframe(styled.style.apply(highlight_best, axis=1).format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]}), use_container_width=True, hide_index=True, height=240)

def page_predict(model, X_all, y_all):
    st.markdown("## 🔮 Customer Risk Predictor")
    st.markdown("Enter customer profile to get an instant churn probability and SHAP explanation")

    if model is None:
        st.error("❌ Primary model not found.")
        return

    # Load all models for comparison
    models_dict = {
        'Stacking (LR Meta)': model
    }
    
    # Try to load other models if available
    try:
        from pathlib import Path
        model_dir = Path("models")
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                if model_file.stem != "stacking_lr_meta":
                    try:
                        model_name = model_file.stem.replace("_", " ").title()
                        models_dict[model_name] = load_model(model_file.stem)
                    except:
                        pass
    except:
        pass

    with st.form("predict_form"):
        st.markdown("### Customer Profile")
        c1, c2, c3 = st.columns(3)
        inputs = {}
        with c1:
            st.caption("**Demographics**")
            inputs["Customer_Age"] = st.number_input("Customer Age", 20, 80, 45, 1)
            inputs["Gender"] = st.selectbox("Gender", ["F", "M"])
            inputs["Education_Level"] = st.selectbox("Education", ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
            inputs["Marital_Status"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            inputs["Income_Category"] = st.selectbox("Income", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
        with c2:
            st.caption("**Account Details**")
            inputs["Card_Category"] = st.selectbox("Card Type", ["Blue", "Silver", "Gold", "Platinum"])
            inputs["Months_on_book"] = st.number_input("Months on Book", 12, 60, 36, 1)
            inputs["Total_Relationship_Count"] = st.number_input("Products Held", 1, 8, 4, 1)
            inputs["Months_Inactive_12_mon"] = st.number_input("Inactive Months (12mo)", 0, 6, 2, 1)
            inputs["Contacts_Count_12_mon"] = st.number_input("Contacts Count (12mo)", 0, 6, 3, 1)
        with c3:
            st.caption("**Transaction Behaviour**")
            inputs["Total_Revolving_Bal"] = st.number_input("Revolving Balance", 0, 3000, 1000, 10)
            inputs["Avg_Utilization_Ratio"] = st.slider("Avg Utilization", 0.0, 1.0, 0.3, 0.01)
            inputs["Total_Trans_Amt"] = st.number_input("Total Trans Amount", 0, 20000, 4000, 100)
            inputs["Total_Trans_Ct"] = st.number_input("Total Trans Count", 0, 150, 60, 1)
            inputs["Total_Amt_Chng_Q4_Q1"] = st.slider("Amt Change Q4→Q1", 0.0, 3.0, 0.8, 0.01)
            inputs["Total_Ct_Chng_Q4_Q1"] = st.slider("Ct Change Q4→Q1", 0.0, 2.0, 0.7, 0.01)

        submitted = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    if submitted:
        X_input = pd.DataFrame([inputs])[X_all.columns]
        
        # Get prediction from primary model
        prob = float(model.predict_proba(X_input)[0, 1])
        pred = int(prob >= THRESHOLD)
        
        # Get predictions from all available models
        model_predictions = {}
        for model_name, model_instance in models_dict.items():
            if model_instance is not None:
                try:
                    model_prob = float(model_instance.predict_proba(X_input)[0, 1])
                    model_pred = int(model_prob >= THRESHOLD)
                    model_predictions[model_name] = {
                        'probability': model_prob,
                        'prediction': model_pred,
                        'risk': risk_label(model_prob)
                    }
                except:
                    pass
        
        # Display primary prediction
        st.markdown("### 🎯 Primary Prediction (Stacking LR Meta)")
        r1, r2, r3 = st.columns([1, 1, 1])
        with r1: 
            st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})
        with r2:
            st.markdown(f"""
            <div style="background:#12152C; padding:15px; border-radius:8px; text-align:center;">
            <div style="font-size:1.4rem; font-weight:800; color:{risk_color(prob)};">{'Churned' if pred else 'Retained'}</div>
            <div style="color:#9CA3AF; margin:5px 0;">Probability: <b style="color:{risk_color(prob)}">{prob*100:.1f}%</b></div>
            <div style="color:#6B7280; font-size:0.8rem;">Risk: <b style="color:{risk_color(prob)}">{risk_label(prob)}</b></div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown("### 📝 Input Summary")
            for k, v in inputs.items():
                st.markdown(f"`{k}`: `{v}`")

        # Display multi-model comparison
        if len(model_predictions) > 1:
            st.markdown("---")
            st.markdown("### 📊 Multi-Model Prediction Comparison")
            
            # ✅ FIX: Store probability as float, let Pandas format it
            comparison_data = []
            for model_name, pred_data in model_predictions.items():
                comparison_data.append({
                    'Model': model_name,
                    'Probability': pred_data['probability'], # <-- Keep as float (0.0 to 1.0)
                    'Prediction': 'Churned' if pred_data['prediction'] == 1 else 'Retained',
                    'Risk Level': pred_data['risk']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            def color_predictions(val):
                if 'Churned' in val:
                    return 'background-color: rgba(239, 68, 68, 0.2); color: #EF4444; font-weight: bold'
                return 'background-color: rgba(16, 185, 129, 0.2); color: #10B981; font-weight: bold'
            
            st.dataframe(
                comparison_df.style.applymap(color_predictions, subset=['Prediction'])
                .format({'Probability': '{:.2%}'}), # ✅ Pandas handles float -> percentage conversion safely
                use_container_width=True,
                hide_index=True,
                height=200
            )
            
            # Visualization: Bar chart comparison
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = go.Figure()
                for model_name, pred_data in model_predictions.items():
                    fig_bar.add_trace(go.Bar(
                        name=model_name,
                        x=[model_name],
                        y=[pred_data['probability'] * 100],
                        marker_color=risk_color(pred_data['probability']),
                        text=[f"{pred_data['probability']*100:.1f}%"],
                        textposition='outside'
                    ))
                
                fig_bar.update_layout(
                    title="Churn Probability by Model",
                    yaxis_title="Probability (%)",
                    xaxis_title="Model",
                    height=300,
                    showlegend=False,
                    paper_bgcolor='#0D0F1E',
                    plot_bgcolor='#12152C',
                    font=dict(color='#C8D0E7'),
                    yaxis=dict(gridcolor='#1E2440'),
                    xaxis=dict(gridcolor='#1E2440')
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            
            with col2:
                avg_prob = np.mean([p['probability'] for p in model_predictions.values()])
                consensus = 'Churned' if avg_prob >= THRESHOLD else 'Retained'
                
                st.markdown(f"""
                <div style="background:#12152C; padding:20px; border-radius:8px; text-align:center;">
                <div style="font-size:1.1rem; color:#7B82A8; margin-bottom:15px;">📈 Model Consensus</div>
                <div style="font-size:2rem; font-weight:800; color:{risk_color(avg_prob)};">{consensus}</div>
                <div style="color:#9CA3AF; margin:10px 0;">Average Probability: <b style="color:{risk_color(avg_prob)}">{avg_prob*100:.1f}%</b></div>
                <div style="font-size:0.85rem; color:#6B7280; margin-top:10px;">
                Models agreeing: {sum(1 for p in model_predictions.values() if p['prediction'] == (1 if consensus == 'Churned' else 0))}/{len(model_predictions)}
                </div>
                </div>
                """, unsafe_allow_html=True)

        # SHAP explanation section
        st.markdown("---")
        st.markdown("### 🔍 SHAP Explanation")
        with st.spinner("Computing individual SHAP values…"):
            explainer_i, _ = build_shap_explainer(model, X_all)
            sv_i = explainer_i.shap_values(X_input, nsamples=100, l1_reg="auto")
            sv_arr = sv_i[0] if isinstance(sv_i, list) else sv_i
            if sv_arr.ndim > 1: sv_arr = sv_arr[0]
            fig_wf = shap_waterfall_fig(explainer_i, sv_arr, X_input.iloc[0])
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()

def page_arena(df_results):
    st.markdown("## 📊 Model Arena")
    st.markdown("Head-to-head comparison of all models")
    
    st.markdown("### 📈 Per-Metric Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics, shared_yaxes=False)
    for i, met in enumerate(metrics, 1):
        vals = df_results[met].tolist()
        names = df_results["Model"].tolist()
        colors = ["#7C3AED" if n == BEST_LABEL else "#1E2440" for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, marker=dict(color=colors, line=dict(color="#2A2F52", width=1)), text=[f"{v:.3f}" for v in vals], textposition="outside"), row=1, col=i)
        fig.update_yaxes(range=[min(vals)*0.99, 1.0], row=1, col=i)
    fig.update_layout(height=300, **PLOTLY_DARK)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def page_shap_explorer(model, X_all, y_all):
    st.markdown("## 🔍 SHAP Explorer")
    st.markdown("Understand why the model predicts churn — globally and individually")

    if model is None:
        st.error("❌ Model not loaded.")
        return

    shap_tab, indiv_tab = st.tabs(["🌐 Global Analysis", "👤 Individual Explanation"])
    
    with shap_tab:
        st.info("💡 Computes SHAP on a 100-row sample. May take ~15-30s on first run.")
        if st.button("🚀 Compute Global SHAP Values", use_container_width=True, type="primary"):
            with st.spinner("Building KernelExplainer and computing SHAP values…"):
                explainer, _ = build_shap_explainer(model, X_all)
                X_sample = X_all.sample(n=100, random_state=42).reset_index(drop=True)
                sv = explainer.shap_values(X_sample, nsamples=100, l1_reg="auto")
                st.session_state["global_shap_values"] = sv
                st.session_state["global_shap_X"] = X_sample
                st.session_state["global_explainer"] = explainer
                st.success("✅ SHAP values computed!")

        if "global_shap_values" in st.session_state:
            sv = st.session_state["global_shap_values"]
            X_sample = st.session_state["global_shap_X"]
            c1, c2 = st.columns(2)
            with c1:
                fig_bar = shap_bar_fig(sv, X_sample)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close()
            with c2:
                fig_summary = shap_summary_plot(sv, X_sample)
                st.pyplot(fig_summary, use_container_width=True)
                plt.close()

    with indiv_tab:
        X_sample = X_all.sample(n=200, random_state=42)
        idx = st.selectbox("Select Customer Index", range(len(X_sample)), format_func=lambda x: f"Index {x}")
        chosen = X_sample.iloc[idx]
        X_row = chosen.to_frame().T.reset_index(drop=True)
        actual = y_all.iloc[idx]
        prob = float(model.predict_proba(X_row)[:, 1][0])
        
        r1, r2 = st.columns(2)
        with r1: st.plotly_chart(gauge_chart(prob), use_container_width=True)
        with r2:
            st.markdown(f"""
            <div style="background:#12152C; padding:15px; border-radius:8px;">
            <div style="font-size:1.2rem; font-weight:700;">Actual: {'Churned' if actual==1 else 'Retained'}</div>
            <div style="color:#9CA3AF;">Predicted Prob: <b>{prob*100:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)

        with st.spinner("Computing SHAP…"):
            explainer, _ = build_shap_explainer(model, X_all)
            sv = explainer.shap_values(X_row, nsamples=100, l1_reg="auto")
            sv_arr = sv[0] if isinstance(sv, list) else sv
            if sv_arr.ndim > 1: sv_arr = sv_arr[0]
            fig_wf = shap_waterfall_fig(explainer, sv_arr, X_row.iloc[0])
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()

    
    

# ───────────────────────────────────────────────────────────────────────────────
# MAIN APP EXECUTION
# ───────────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
# MAIN APP EXECUTION
# ───────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="ChurnShield AI", page_icon="🛡️", layout="wide")
    
    # Custom CSS for tab styling
    st.markdown("""
    <style>
        /* Hide default tab padding */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding-bottom: 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            background-color: #12152C;
            border: 1px solid #1E2440;
            border-radius: 8px 8px 0 0;
            padding: 0 20px;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #7C3AED !important;
            border-color: #7C3AED !important;
            color: white !important;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 0.95rem;
            color: #C8D0E7;
        }
        .stTabs [aria-selected="true"] p {
            color: white !important;
        }
        /* Hide the default Streamlit header/footer */
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)
    
    # Load model & data
    model = load_model(BEST_KEY)
    model_loaded = model is not None
    data_ok = os.path.exists(DATA_PATH)
    df_results = get_comparison_results()
    df_results.columns = [c.strip() for c in df_results.columns]
    
    X_all, y_all = load_data() if data_ok else (pd.DataFrame(), pd.Series())
    if not X_all.empty:
        X_all = X_all.reindex(columns=sorted(X_all.columns))
    
    # ── TABBED NAVIGATION (Nabber Style) ─────────────────────────────────────
    tab_overview, tab_predict, tab_arena, tab_shap = st.tabs([
        "🏠 Overview",
        "🔮 Predict", 
        "📊 Model Arena",
        "🔍 SHAP Explorer"
    ])
    
    # Render each page inside its tab
    with tab_overview:
        page_overview(model_loaded, data_ok, X_all, y_all, df_results)
    
    with tab_predict:
        page_predict(model, X_all, y_all)
    
    with tab_arena:
        page_arena(df_results)
    
    with tab_shap:
        page_shap_explorer(model, X_all, y_all)
    
    # Footer
    st.markdown("<div style='text-align:center; color:#3D4270; font-size:0.65rem; padding:20px; margin-top:2rem;'>ChurnShield v2.0 · Stacking (LR Meta)</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()