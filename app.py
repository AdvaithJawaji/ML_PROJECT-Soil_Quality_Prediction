import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import plot_tree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "soil_data.csv")

st.set_page_config(
    page_title="Terra | Soil Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --cream: #f7f3eb;
        --paper: #fffdf8;
        --ink: #31261d;
        --muted: #786b5d;
        --brown: #76543b;
        --brown-dark: #4e3828;
        --sage: #667b55;
        --line: #e9dfd1;
    }
    .stApp {
        background: radial-gradient(ellipse at 10% 0%, #f2e9d9 0%, transparent 42%),
                    linear-gradient(150deg, #f8f5ee 0%, #f1ebdf 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); }
    h1, h2 { font-family: 'Playfair Display', Georgia, serif; }
    .hero {
        background: linear-gradient(120deg, #4e3828 0%, #76543b 57%, #8b704f 100%);
        border-radius: 24px;
        padding: 2.3rem 2.6rem;
        color: #fffaf0;
        box-shadow: 0 18px 42px rgba(78, 56, 40, .15);
        position: relative;
        overflow: hidden;
        animation: rise-in .65s ease-out both;
    }
    .hero:after {
        content: "✳";
        position: absolute;
        right: 4%;
        top: -45%;
        color: rgba(255,255,255,.08);
        font-size: 18rem;
        line-height: 1;
    }
    .hero h1 { color: #fffaf0; font-size: clamp(2rem, 4vw, 3.1rem); margin: .35rem 0; }
    .hero p { color: #eadfce; margin: .3rem 0 0; font-size: 1rem; }
    .eyebrow { color: #d6c3a7; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
    .section-kicker { color: var(--sage); text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
    [data-testid="stTabs"] { margin-top: 1.3rem; }
    [data-baseweb="tab-list"] { gap: .55rem; border-bottom: 1px solid var(--line); }
    [data-baseweb="tab"] {
        height: 3.1rem;
        color: var(--muted);
        background: rgba(255,253,248,.65);
        border-radius: 12px 12px 0 0;
        padding: 0 1.2rem;
        transition: background .2s ease, color .2s ease;
    }
    [data-baseweb="tab"]:hover { color: var(--brown-dark); background: #fffdf8; }
    [aria-selected="true"][data-baseweb="tab"] { color: var(--brown-dark); border-bottom-color: var(--brown); }
    [data-testid="stMetric"] {
        background: rgba(255,253,248,.88);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 6px 22px rgba(78,56,40,.045);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(78,56,40,.09); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--brown-dark); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,253,248,.63);
        border-color: rgba(255,255,255,.78) !important;
        border-radius: 18px;
        box-shadow: 0 12px 34px rgba(78,56,40,.055), inset 0 1px rgba(255,255,255,.72);
        backdrop-filter: blur(16px);
        transition: transform .22s ease, box-shadow .22s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(78,56,40,.085), inset 0 1px rgba(255,255,255,.8);
    }
    .prediction-card {
        background: rgba(255,253,248,.65);
        border: 1px solid rgba(255,255,255,.85);
        border-radius: 18px;
        padding: 1.35rem;
        min-height: 142px;
        box-shadow: 0 12px 30px rgba(78,56,40,.075), inset 0 1px rgba(255,255,255,.8);
        backdrop-filter: blur(16px);
        animation: rise-in .4s ease-out both;
        transition: transform .22s ease, box-shadow .22s ease;
    }
    .prediction-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 38px rgba(78,56,40,.12), inset 0 1px rgba(255,255,255,.9);
    }
    .prediction-card.rf { border-top: 4px solid var(--sage); }
    .prediction-card.dt { border-top: 4px solid var(--brown); }
    .prediction-title { color: var(--muted); font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; }
    .prediction-value { color: var(--brown-dark); font-family: 'Playfair Display', Georgia, serif; font-size: 2rem; font-weight: 700; margin: .4rem 0; }
    .prediction-note { color: var(--muted); font-size: .85rem; }
    div.stButton > button {
        border-radius: 11px;
        border-color: #d5c4ae;
        color: var(--brown-dark);
        background: #fffdf8;
        transition: all .18s ease;
    }
    div.stButton > button:hover { border-color: var(--brown); color: #fffaf0; background: var(--brown); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .footnote { color: var(--muted); font-size: .8rem; }
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        color: var(--brown-dark);
        background: rgba(102,123,85,.12);
        border: 1px solid rgba(102,123,85,.22);
        border-radius: 99px;
        padding: .45rem .75rem;
        font-size: .82rem;
        font-weight: 600;
    }
    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        min-height: 2.8rem;
        border-radius: 11px;
        border: 1px solid #d5c4ae;
        color: var(--brown-dark);
        background: rgba(255,253,248,.72);
        transition: all .2s ease;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        border-color: var(--brown);
        color: #fffaf0;
        background: var(--brown);
    }
    @keyframes rise-in { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SOIL_PROFILES = {
    "Balanced loam": {
        "pH": 6.8, "Nitrogen": 290.0, "Phosphorus": 28.0,
        "Potassium": 180.0, "Organic_Matter": 3.5, "Moisture": 25.0,
    },
    "Acidic sample": {
        "pH": 4.8, "Nitrogen": 120.0, "Phosphorus": 12.0,
        "Potassium": 85.0, "Organic_Matter": 1.1, "Moisture": 14.0,
    },
    "Nutrient-rich sample": {
        "pH": 7.1, "Nitrogen": 410.0, "Phosphorus": 45.0,
        "Potassium": 270.0, "Organic_Matter": 5.2, "Moisture": 30.0,
    },
    "Dry sample": {
        "pH": 7.8, "Nitrogen": 180.0, "Phosphorus": 16.0,
        "Potassium": 120.0, "Organic_Matter": 1.6, "Moisture": 8.0,
    },
}


@st.cache_resource
def load_ml_artifacts():
    decision_tree = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
    random_forest = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as metrics_file:
        model_metrics = json.load(metrics_file)
    return decision_tree, random_forest, scaler, label_encoder, model_metrics


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    dt_model, rf_model, scaler, label_encoder, metrics = load_ml_artifacts()
except (FileNotFoundError, OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the trained model artifacts from `{MODEL_DIR}`: {error}")
    st.info("Run `train_models.py` from the ML project folder to create or refresh them.")
    st.stop()

feature_names = metrics.get("feature_names", list(scaler.feature_names_in_))
class_names = list(label_encoder.classes_)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">TERRA · SOIL INTELLIGENCE</div>
      <h1>Read the land.<br>Grow with confidence.</h1>
      <p>Explore soil quality predictions and the trained models behind them.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_live, tab_performance, tab_insights, tab_data = st.tabs(
    ["🌱  Live classification", "📈  Model performance", "🌿  Feature & tree insights", "🗂️  Dataset explorer"]
)

with tab_live:
    st.markdown('<div class="section-kicker">Soil assessment</div>', unsafe_allow_html=True)
    st.subheader("Enter your soil test values")
    input_col, result_col = st.columns([1, 1.08], gap="large")
    default_inputs = {
        "pH": 6.8,
        "Nitrogen": 290.0,
        "Phosphorus": 28.0,
        "Potassium": 180.0,
        "Organic_Matter": 3.5,
        "Moisture": 25.0,
    }

    with input_col:
        with st.container(border=True):
            st.caption("Start with a profile or tune each measurement to your soil test.")
            selected_profile = st.selectbox(
                "Quick-start profile",
                ["Custom values", *SOIL_PROFILES],
                key="soil_profile",
            )
            profile_defaults = SOIL_PROFILES.get(selected_profile, default_inputs)
            profile_key = selected_profile.lower().replace(" ", "_")
            pH = st.slider(
                "Soil pH", 3.5, 9.0, value=profile_defaults["pH"], step=0.1,
                key=f"{profile_key}_pH", help="Scale from acidic to alkaline.",
            )
            nitrogen = st.slider(
                "Nitrogen · kg/ha", 50.0, 480.0, value=profile_defaults["Nitrogen"],
                step=5.0, key=f"{profile_key}_nitrogen",
            )
            phosphorus = st.slider(
                "Phosphorus · kg/ha", 5.0, 55.0, value=profile_defaults["Phosphorus"],
                step=1.0, key=f"{profile_key}_phosphorus",
            )
            potassium = st.slider(
                "Potassium · kg/ha", 60.0, 320.0, value=profile_defaults["Potassium"],
                step=5.0, key=f"{profile_key}_potassium",
            )
            organic_matter = st.slider(
                "Organic matter · %", 0.5, 6.0, value=profile_defaults["Organic_Matter"],
                step=0.1, key=f"{profile_key}_organic_matter",
            )
            moisture = st.slider(
                "Soil moisture · %", 5.0, 55.0, value=profile_defaults["Moisture"],
                step=0.5, key=f"{profile_key}_moisture",
            )

    sample = pd.DataFrame(
        [[pH, nitrogen, phosphorus, potassium, organic_matter, moisture]],
        columns=feature_names,
    )
    sample_scaled = scaler.transform(sample)
    dt_index = int(dt_model.predict(sample_scaled)[0])
    rf_index = int(rf_model.predict(sample_scaled)[0])
    dt_label = str(label_encoder.inverse_transform([dt_index])[0])
    rf_label = str(label_encoder.inverse_transform([rf_index])[0])
    dt_probabilities = dt_model.predict_proba(sample_scaled)[0]
    rf_probabilities = rf_model.predict_proba(sample_scaled)[0]

    with result_col:
        st.markdown('<div class="section-kicker">Model results</div>', unsafe_allow_html=True)
        st.subheader("Classification")
        dt_col, rf_col = st.columns(2)
        dt_confidence = float(np.max(dt_probabilities)) * 100
        rf_confidence = float(np.max(rf_probabilities)) * 100
        with dt_col:
            st.markdown(
                f'<div class="prediction-card dt"><div class="prediction-title">Decision tree</div>'
                f'<div class="prediction-value">{dt_label}</div>'
                f'<div class="prediction-note">Highest class probability · {dt_confidence:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with rf_col:
            st.markdown(
                f'<div class="prediction-card rf"><div class="prediction-title">Random forest · 100 trees</div>'
                f'<div class="prediction-value">{rf_label}</div>'
                f'<div class="prediction-note">Highest class probability · {rf_confidence:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        agreement_col, download_col = st.columns([1, 1])
        with agreement_col:
            if dt_label == rf_label:
                st.markdown(
                    '<div class="status-chip">✦ Models agree on this classification</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-chip">↗ Models differ · review probabilities</div>',
                    unsafe_allow_html=True,
                )
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: float(value) for name, value in sample.iloc[0].items()},
            "decision_tree": {
                "prediction": dt_label,
                "class_probabilities": {
                    str(name): float(probability)
                    for name, probability in zip(class_names, dt_probabilities)
                },
            },
            "random_forest": {
                "prediction": rf_label,
                "class_probabilities": {
                    str(name): float(probability)
                    for name, probability in zip(class_names, rf_probabilities)
                },
            },
            "models_agree": dt_label == rf_label,
        }
        with download_col:
            st.download_button(
                "↓  Download assessment · JSON",
                data=json.dumps(report, indent=2),
                file_name="terra-soil-assessment.json",
                mime="application/json",
            )

        st.write("")
        probability_frame = pd.DataFrame(
            {
                "Class": class_names,
                "Decision tree": dt_probabilities,
                "Random forest": rf_probabilities,
            }
        ).set_index("Class")
        st.caption("Predicted class probabilities")
        st.bar_chart(probability_frame, color=["#9b7653", "#667b55"], x_label="Quality class", y_label="Probability")

        recommendations = {
            "High": ("Strong result", "The model classifies this sample as high quality. Maintain soil structure and organic matter; confirm management decisions with local soil guidance."),
            "Medium": ("Room to improve", "The model classifies this sample as medium quality. Review the nutrient balance and moisture readings, and use a soil test to guide any amendments."),
            "Low": ("Needs attention", "The model classifies this sample as low quality. Check pH and nutrient availability with a local soil test before choosing corrective amendments."),
        }
        title, recommendation = recommendations.get(
            rf_label, ("Assessment", "Review the input measurements and consult a local soil specialist.")
        )
        st.markdown(f"**{title}**  \n{recommendation}")
        st.caption("Model output is an estimate, not a substitute for laboratory testing or local agronomic advice.")

        if os.path.isfile(DATA_PATH):
            reference_data = load_dataset()
            if all(name in reference_data.columns for name in feature_names):
                with st.expander("How does this compare with training samples?"):
                    range_rows = []
                    sample_values = sample.iloc[0]
                    for name in feature_names:
                        observed = reference_data[name].dropna()
                        low, high = observed.quantile([0.05, 0.95])
                        value = float(sample_values[name])
                        range_rows.append(
                            {
                                "Measurement": name.replace("_", " "),
                                "Your value": value,
                                "Training data · 5th–95th percentile": f"{low:.2f} – {high:.2f}",
                                "Range check": "Within observed range" if low <= value <= high else "Outside observed range",
                            }
                        )
                    st.dataframe(pd.DataFrame(range_rows), hide_index=True, width="stretch")
                    st.caption(
                        "This is a descriptive comparison with this dataset, not an agronomic optimum or model guarantee."
                    )
            else:
                st.info("Training data does not contain all model features, so the range comparison is unavailable.")

    st.write("")
    with st.expander("Current sample values"):
        st.dataframe(sample, hide_index=True, width="stretch")

with tab_performance:
    st.markdown('<div class="section-kicker">Evaluation results</div>', unsafe_allow_html=True)
    st.subheader("How the models performed")
    dt_metrics = metrics["decision_tree"]
    rf_metrics = metrics["random_forest"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Random forest accuracy", f"{rf_metrics['accuracy']:.1%}")
    metric_cols[1].metric("Macro F1 score", f"{rf_metrics['f1_macro']:.1%}")
    metric_cols[2].metric("5-fold CV mean", f"{rf_metrics['cv_mean']:.1%}")
    metric_cols[3].metric("CV standard deviation", f"±{rf_metrics['cv_std']:.1%}")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Held-out test metrics**")
        score_frame = pd.DataFrame(
            {
                "Decision tree": [
                    dt_metrics["accuracy"],
                    dt_metrics["precision_macro"],
                    dt_metrics["recall_macro"],
                    dt_metrics["f1_macro"],
                ],
                "Random forest": [
                    rf_metrics["accuracy"],
                    rf_metrics["precision_macro"],
                    rf_metrics["recall_macro"],
                    rf_metrics["f1_macro"],
                ],
            },
            index=["Accuracy", "Macro precision", "Macro recall", "Macro F1"],
        )
        st.bar_chart(score_frame, color=["#9b7653", "#667b55"], y_label="Score")
    with right:
        selected_model = st.selectbox("Confusion matrix", ["Random forest", "Decision tree"])
        selected_metrics = rf_metrics if selected_model == "Random forest" else dt_metrics
        matrix = np.asarray(selected_metrics["confusion_matrix"])
        matrix_frame = pd.DataFrame(
            matrix,
            index=[f"Actual {name}" for name in class_names],
            columns=[f"Predicted {name}" for name in class_names],
        )
        st.dataframe(matrix_frame, width="stretch")
        st.caption("Rows are actual labels; columns are predicted labels. Values are test-set sample counts.")

    st.markdown(
        '<p class="footnote">Metrics are computed on the stratified held-out test split. '
        'Cross-validation values summarize five stratified folds.</p>',
        unsafe_allow_html=True,
    )

with tab_insights:
    st.markdown('<div class="section-kicker">Model interpretability</div>', unsafe_allow_html=True)
    st.subheader("What drives the predictions")
    importance_col, tree_col = st.columns([.9, 1.1], gap="large")
    with importance_col:
        importance_model = st.selectbox("Feature importance model", ["Random forest", "Decision tree"])
        importance_metrics = rf_metrics if importance_model == "Random forest" else dt_metrics
        importance_key = "feature_importances"
        if importance_key not in importance_metrics:
            st.error("Feature importance values are missing from the metrics artifact. Retrain the models to refresh it.")
        else:
            importance = pd.Series(importance_metrics[importance_key], dtype=float).reindex(feature_names)
            if importance.isna().any():
                st.error("Feature importance values do not match the model feature names.")
            else:
                importance = importance.sort_values()
                st.markdown(f"**{importance_model} · impurity-based importance**")
                st.bar_chart(
                    importance.rename("Importance"),
                    horizontal=True,
                    color="#8b704f",
                    x_label="Relative importance",
                )
                top_feature = str(importance.index[-1])
                st.caption(
                    f"{top_feature} has the highest relative importance in this model. "
                    "Importance describes contribution to model splits, not causality."
                )

    with tree_col:
        st.markdown("**Trained decision tree · first three levels**")
        figure, axis = plt.subplots(figsize=(13, 6.5))
        figure.patch.set_facecolor("#fffdf8")
        axis.set_facecolor("#fffdf8")
        plot_tree(
            dt_model,
            max_depth=3,
            feature_names=feature_names,
            class_names=[str(name) for name in class_names],
            filled=True,
            rounded=True,
            proportion=True,
            impurity=False,
            fontsize=7,
            ax=axis,
        )
        axis.set_title("Actual trained decision tree", color="#4e3828", fontsize=12, pad=14)
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption("Nodes show learned feature thresholds and class proportions. The displayed tree is truncated for readability.")

    st.markdown("**Interpretation notes**")
    st.write(
        "Feature importance is read directly from the selected trained estimator. "
        "It is relative within that model and does not establish that a feature causes a soil-quality outcome."
    )

with tab_data:
    st.markdown('<div class="section-kicker">Training data</div>', unsafe_allow_html=True)
    st.subheader("Explore the soil samples")
    if os.path.isfile(DATA_PATH):
        dataset = load_dataset()
        label_column = "Soil_Quality" if "Soil_Quality" in dataset.columns else (
            "Soil_Type" if "Soil_Type" in dataset.columns else None
        )
        count_col, label_col = st.columns(2)
        count_col.metric("Samples", f"{len(dataset):,}")
        count_col.metric("Measurements", f"{len(dataset.columns) - (1 if label_column else 0)}")
        if label_column:
            distribution = dataset[label_column].value_counts().reindex(class_names, fill_value=0)
            with label_col:
                st.markdown("**Class distribution**")
                st.bar_chart(distribution.rename("Samples"), color="#667b55")
        search = st.text_input("Filter rows", placeholder="Search values or class labels")
        filtered_dataset = dataset
        if search:
            row_matches = dataset.astype(str).apply(
                lambda column: column.str.contains(search, case=False, regex=False)
            ).any(axis=1)
            filtered_dataset = dataset[row_matches]
        st.caption(f"Showing {len(filtered_dataset):,} of {len(dataset):,} rows.")
        st.dataframe(filtered_dataset, width="stretch", hide_index=True)
        with st.expander("Summary statistics"):
            st.dataframe(dataset.describe().T, width="stretch")
    else:
        st.error(f"Dataset not found at `{DATA_PATH}`.")

st.divider()
st.markdown(
    '<p class="footnote">TERRA · Soil quality classification · Decision tree and random forest</p>',
    unsafe_allow_html=True,
)
