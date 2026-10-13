# Terra — Soil Quality Prediction

Terra is a soil-quality classification project with two interactive interfaces: a Streamlit dashboard and a standalone browser dashboard. It compares a Decision Tree with a 100-tree Random Forest and presents model predictions, probability estimates, evaluation metrics, and model insights.

> **Important:** This is an educational decision-support project. The bundled data is synthetic and the labels are generated/curated for experimentation. Model scores are not evidence of field accuracy. Do not use predictions as a substitute for laboratory soil testing or advice from a qualified agronomist.

## At a glance

- **Purpose:** Explore how basic soil measurements can be used to classify samples as `Low`, `Medium`, or `High` quality.
- **Inputs:** pH, nitrogen, phosphorus, potassium, organic matter, and moisture.
- **Models:** Standard-scaled Decision Tree and Random Forest classifiers.
- **Interfaces:** Streamlit app (`app.py`) and browser dashboard (`index.html`).
- **Data:** 1,500-row soil dataset in `data/soil_data.csv`.
- **Reproducibility:** Fixed random seeds in the data generation and model training scripts.

## Features

### Live classification

- Adjust six soil measurements with interactive controls.
- Start from quick soil profiles and customize their values.
- View Decision Tree and Random Forest class predictions and per-class probabilities.
- See whether the two classifiers agree.
- Compare measurements with the training dataset's 5th–95th percentile ranges.
- Download the current inputs, predictions, and probabilities as a JSON assessment.
- Review cautious, general-purpose guidance based on the Random Forest class. Recommendations are not prescriptions.

### Model performance

- Compare accuracy, macro precision, macro recall, and macro F1.
- View the held-out test confusion matrix for either model.
- Inspect five-fold stratified cross-validation mean and standard deviation.

The checked-in metrics currently report:

| Metric | Decision Tree | Random Forest |
| --- | ---: | ---: |
| Hold-out accuracy | 63.67% | 74.00% |
| Macro F1 | 64.56% | 73.96% |
| 5-fold CV mean accuracy | 67.27% | 75.73% |
| 5-fold CV standard deviation | 3.59% | 4.14% |

These figures describe performance on the project dataset only. Because the labels are synthetic, they should not be interpreted as real-world agricultural validation.

### Interpretability and dataset exploration

- Plot feature importances directly from the stored model metrics.
- Render the trained Decision Tree's first levels.
- Browse, search, and summarize the dataset and class distribution.
- The browser dashboard also includes presets, feature-importance views, confusion matrices, a tree inspector, and a printable report.

## Advantages

- Makes model outputs and class probabilities easy to explore without writing code.
- Provides a compact comparison between a simple single tree and a tree ensemble.
- Keeps preprocessing consistent by applying the fitted scaler before prediction.
- Includes the dataset, trained artifacts, scripts, and visualizations needed to inspect or retrain the example.
- Offers both a Python dashboard and a standalone web experience for demonstrations.

## Project structure

```text
ML/
├── app.py                              # Streamlit dashboard
├── index.html                          # Standalone browser dashboard
├── styles.css                          # Browser dashboard styling
├── app.js                              # Browser dashboard interactions/inference
├── server.py                           # Local static server and prediction API
├── requirements.txt                    # Python dependencies
├── generate_data.py                    # Generate reproducible synthetic soil data
├── train_models.py                     # Train/evaluate and save model artifacts
├── generate_plots.py                   # Generate model evaluation plots
├── data/
│   └── soil_data.csv                   # Soil measurements and quality labels
├── models/
│   ├── decision_tree.pkl               # Trained Decision Tree
│   ├── random_forest.pkl               # Trained Random Forest
│   ├── scaler.pkl                      # Fitted StandardScaler
│   ├── label_encoder.pkl               # Fitted class-label encoder
│   ├── metrics.json                    # Evaluation metrics and importances
│   └── model_export.json               # Browser inference model export
├── static/
│   └── plots/                          # Generated evaluation visualizations
├── ML ABSTRACT.pdf                     # Project abstract
└── Soil Quality Classification.pptx    # Project presentation
```

Python cache folders, virtual environments, and local editor files are excluded from version control.

## Getting started

### Requirements

- Python 3.10 or newer
- pip
- A modern web browser

### Install dependencies

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

Streamlit prints the local URL (usually `http://localhost:8501`) when the app is ready.

### Run the standalone browser dashboard

The static site and model artifacts need to be served over HTTP so the browser can load them:

```bash
python server.py
```

Open `http://localhost:8080`. The server serves the dashboard and exposes `POST /api/predict` for predictions using the Python model artifacts.

## Generate data, retrain, and regenerate plots

The training script supports either `Soil_Quality` or `Soil_Type` as the target-label column. The supplied CSV uses `Soil_Type`; `generate_data.py` creates a reproducible synthetic dataset using `Soil_Quality`.

To replace the current CSV with newly generated synthetic data and retrain:

```bash
python generate_data.py
python train_models.py
python generate_plots.py
```

> Running `generate_data.py` replaces `data/soil_data.csv`. Keep a copy first if you need to preserve the supplied dataset.

The training pipeline uses an 80/20 stratified train/test split (random seed 42), fits a `StandardScaler` on the training features, trains a Decision Tree and a 100-estimator Random Forest, and reports five-fold stratified cross-validation metrics. Model files are serialized with joblib. Only load model files from trusted sources.

## Data dictionary

| Feature | Meaning | Example unit |
| --- | --- | --- |
| `pH` | Soil acidity/alkalinity measurement | pH scale |
| `Nitrogen` | Available nitrogen | kg/ha |
| `Phosphorus` | Available phosphorus | kg/ha |
| `Potassium` | Available potassium | kg/ha |
| `Organic_Matter` | Organic matter content | percent |
| `Moisture` | Soil moisture content | percent |
| `Soil_Type` / `Soil_Quality` | Classification target in the CSV/generator | Low, Medium, High |

The model expects the six numeric measurements in the order listed above. The training script treats either supported label-column name as the target and uses all other columns as model features; the supplied dataset should therefore contain only the six model features and one target column.

## Model interpretation

- **Decision Tree:** A compact set of learned decision rules that is straightforward to inspect.
- **Random Forest:** An ensemble of 100 trees; averaging across trees generally provides a more stable prediction than an individual tree.
- **Feature importance:** Split-based relative importance helps describe which features the fitted model used most. It is not a causal explanation.
- **Probability:** `predict_proba` values summarize model estimates over the available classes. They are not necessarily calibrated probabilities of real-world soil conditions.

## Limitations and responsible use

- The dataset and class labels are synthetic/educational, so high test scores can reflect the synthetic label-generation process.
- No independent field dataset or laboratory validation is included.
- Soil fertility depends on crop, region, season, sampling method, and many factors that are not represented by these six inputs.
- Training-data percentiles are descriptive dataset ranges, not recommended agronomic thresholds.
- The app's suggestions are general prompts to investigate, not fertilizer or lime application instructions.
- Model artifacts are version-sensitive. If loading warnings or errors occur, install compatible dependencies and retrain the models from the same environment.

## Additional project material

The PDF abstract and PowerPoint presentation are included for project review and demonstrations. The local `chronic_kidney_disease_analysis.ipynb` file is unrelated to soil quality and is intentionally not part of this repository.

## License and attribution

No license is currently specified. Ask the project owner before reusing, redistributing, or adapting the dataset, model artifacts, presentation, or other project material.
