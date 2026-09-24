import os
import json
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Soil Quality Classification ML",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Styling & Glassmorphism
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Title Banner */
    .title-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #065f46 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
        text-align: center;
    }
    
    .title-banner h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .title-banner p {
        color: #cbd5e1;
        margin-top: 8px;
        font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        color: #94a3b8;
        font-size: 0.95rem;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .val-high { color: #34d399; }
    .val-medium { color: #fbbf24; }
    .val-low { color: #f87171; }
    
    /* Section Containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        background-color: rgba(30, 41, 59, 0.6);
        color: #cbd5e1;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #059669 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load artifacts
@st.cache_resource
def load_ml_artifacts():
    dt_model = joblib.load(os.path.join("models", "decision_tree.pkl"))
    rf_model = joblib.load(os.path.join("models", "random_forest.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))
    
    with open(os.path.join("models", "metrics.json"), "r") as f:
        metrics = json.load(f)
        
    return dt_model, rf_model, scaler, le, metrics

try:
    dt_model, rf_model, scaler, le, metrics = load_ml_artifacts()
except Exception as e:
    st.error("Model artifacts not found! Please run `train_models.py` first.")
    st.stop()

# Header Banner
st.markdown("""
    <div class="title-banner">
        <h1>🌱 Soil Quality Classification Engine</h1>
        <p>Comparative Machine Learning Evaluation: <b>Decision Tree</b> vs. <b>Random Forest Ensemble</b></p>
        <p style="font-size: 0.85rem; color: #94a3b8;">Authors: P. Naveen | J. Advaith | P. Shashe Preetham &nbsp;•&nbsp; Guided by: Prof. G. Ravindra Babu</p>
    </div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Classification", 
    "📊 Performance Analytics", 
    "🌲 Feature & Tree Insights", 
    "📁 Dataset Explorer"
])

# ==============================================================================
# TAB 1: LIVE CLASSIFICATION
# ==============================================================================
with tab1:
    st.markdown("### 🧪 Input Soil Parameters")
    
    col_input, col_pred = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("Adjust the soil chemical and physical test parameters below:")
        
        pH = st.slider("Soil pH Level", 3.5, 9.0, 6.8, 0.1, help="Optimal range: 6.0 - 7.5")
        nitrogen = st.slider("Nitrogen (N) [kg/ha]", 50.0, 480.0, 290.0, 5.0, help="Optimal range: > 280 kg/ha")
        phosphorus = st.slider("Phosphorus (P) [kg/ha]", 5.0, 55.0, 28.0, 1.0, help="Optimal range: > 25 kg/ha")
        potassium = st.slider("Potassium (K) [kg/ha]", 60.0, 320.0, 180.0, 5.0, help="Optimal range: > 150 kg/ha")
        organic_matter = st.slider("Organic Matter (%)", 0.5, 6.0, 3.5, 0.1, help="Optimal range: > 3.0%")
        moisture = st.slider("Soil Moisture (%)", 5.0, 55.0, 25.0, 0.5, help="Optimal range: 20% - 40%")
        
        sample_input = pd.DataFrame([{
            'pH': pH,
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Organic_Matter': organic_matter,
            'Moisture': moisture
        }])
        
    with col_pred:
        st.markdown("### 🎯 Model Predictions")
        
        # Scaling input
        sample_scaled = scaler.transform(sample_input)
        
        # Predictions
        dt_pred_idx = dt_model.predict(sample_scaled)[0]
        dt_pred_label = le.inverse_transform([dt_pred_idx])[0]
        dt_probs = dt_model.predict_proba(sample_scaled)[0]
        
        rf_pred_idx = rf_model.predict(sample_scaled)[0]
        rf_pred_label = le.inverse_transform([rf_pred_idx])[0]
        rf_probs = rf_model.predict_proba(sample_scaled)[0]
        
        # Display side by side predictions
        p1, p2 = st.columns(2)
        
        with p1:
            cls_dt = f"val-{dt_pred_label.lower()}"
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Decision Tree</h3>
                    <div class="value {cls_dt}">{dt_pred_label} Quality</div>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Single Decision Tree Split</p>
                </div>
            """, unsafe_allow_html=True)
            
        with p2:
            cls_rf = f"val-{rf_pred_label.lower()}"
            st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #059669;">
                    <h3>Random Forest</h3>
                    <div class="value {cls_rf}">{rf_pred_label} Quality</div>
                    <p style="color: #34d399; font-size: 0.85rem; margin-top: 4px;">Recommended Ensemble Model</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("#### Class Probability Distributions")
        
        prob_df = pd.DataFrame({
            'Class': le.classes_,
            'Decision Tree (%)': np.round(dt_probs * 100, 1),
            'Random Forest (%)': np.round(rf_probs * 100, 1)
        })
        
        st.dataframe(prob_df, width="stretch", hide_index=True)
        
        # Agronomic Guidance Box
        st.markdown("#### 💡 Agronomic Recommendations")
        rec_rf = rf_pred_label
        if rec_rf == "High":
            st.success("🟢 **Excellent Soil Fertility!** Optimal for major crop cultivation. Maintain organic matter levels with rotation and sustainable irrigation.")
        elif rec_rf == "Medium":
            st.warning("🟡 **Moderate Soil Quality.** Fertilizer augmentation recommended: check N-P-K balance and maintain 20–30% moisture.")
        else:
            st.error("🔴 **Low Soil Quality.** High soil stress detected. Lime treatment (if acidic) or organic composting needed to boost organic matter and nutrient availability.")

# ==============================================================================
# TAB 2: PERFORMANCE ANALYTICS
# ==============================================================================
with tab2:
    st.markdown("### 📈 Comprehensive Model Comparison")
    
    dt_m = metrics['decision_tree']
    rf_m = metrics['random_forest']
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random Forest Accuracy", f"{rf_m['accuracy']*100:.2f}%", f"+{(rf_m['accuracy']-dt_m['accuracy'])*100:.2f}% vs Decision Tree")
    m2.metric("Random Forest F1-Score", f"{rf_m['f1_macro']*100:.2f}%", f"+{(rf_m['f1_macro']-dt_m['f1_macro'])*100:.2f}% vs Decision Tree")
    m3.metric("5-Fold Cross Validation Score", f"{rf_m['cv_mean']*100:.2f}%", f"+{(rf_m['cv_mean']-dt_m['cv_mean'])*100:.2f}% vs Decision Tree")
    m4.metric("Variance Reduction (CV Std)", f"±{rf_m['cv_std']*100:.2f}%", f"Lower variance")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join("static", "plots", "model_comparison.png"), caption="Performance Metrics Comparison Bar Chart", width="stretch")
    with c2:
        st.image(os.path.join("static", "plots", "confusion_matrices.png"), caption="Confusion Matrices (Decision Tree vs. Random Forest)", width="stretch")

# ==============================================================================
# TAB 3: FEATURE & TREE INSIGHTS
# ==============================================================================
with tab3:
    st.markdown("### 🔍 Model Interpretability & Tree Structure")
    
    col_feat, col_tree = st.columns(2)
    with col_feat:
        st.image(os.path.join("static", "plots", "feature_importance.png"), caption="Soil Feature Importance Weights", width="stretch")
    with col_tree:
        st.image(os.path.join("static", "plots", "decision_tree_structure.png"), caption="Visualization of Decision Tree Splits (Depth=3)", width="stretch")
        
    st.markdown("""
        #### 📌 Key Takeaways:
        - **Random Forest Ensemble Stability**: By combining 100 decision trees via bagging and random feature sub-selection, Random Forest overcomes single-tree variance and overfitting.
        - **Most Influential Features**: Soil Nitrogen, pH, and Organic Matter contribute highest to soil quality classification.
    """)

# ==============================================================================
# TAB 4: DATASET EXPLORER
# ==============================================================================
with tab4:
    st.markdown("### 📁 Dataset Overview (`soil_data.csv`)")
    
    data_path = os.path.join("data", "soil_data.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        
        st.write(f"Total Record Count: **{len(df_raw)} samples**")
        st.dataframe(df_raw.head(100), width="stretch")
        
        st.markdown("#### Summary Statistics")
        st.dataframe(df_raw.describe().T, width="stretch")
    else:
        st.warning("Dataset not found on disk.")
