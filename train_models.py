import os
import json
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

def train_and_evaluate():
    data_path = os.path.join("data", "soil_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    
    target_column = next(
        (column for column in ('Soil_Quality', 'Soil_Type') if column in df.columns),
        None,
    )
    if target_column is None:
        raise ValueError(
            "Dataset must contain a target column named 'Soil_Quality' or 'Soil_Type'."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    feature_names = list(X.columns)
    
    # 1. Encode Target Labels
    le = LabelEncoder()
    # Ensure consistent order: ['Low', 'Medium', 'High']
    le.fit(['Low', 'Medium', 'High'])
    y_encoded = le.transform(y)
    
    # 2. Train-Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Decision Tree Model
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    dt_model.fit(X_train_scaled, y_train)
    dt_pred = dt_model.predict(X_test_scaled)
    
    # 5. Random Forest Model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=8,
        max_features='sqrt',
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # 6. Cross-Validation (5-Fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt_cv_scores = cross_val_score(dt_model, scaler.transform(X), y_encoded, cv=cv)
    rf_cv_scores = cross_val_score(rf_model, scaler.transform(X), y_encoded, cv=cv)
    
    # 7. Metrics Computation Helper
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            'accuracy': float(acc),
            'precision_macro': float(p_macro),
            'recall_macro': float(r_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(p_weighted),
            'recall_weighted': float(r_weighted),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm
        }
        
    dt_metrics = get_metrics(y_test, dt_pred)
    dt_metrics['cv_mean'] = float(dt_cv_scores.mean())
    dt_metrics['cv_std'] = float(dt_cv_scores.std())
    dt_metrics['feature_importances'] = {
        feat: float(imp) for feat, imp in zip(feature_names, dt_model.feature_importances_)
    }
    
    rf_metrics = get_metrics(y_test, rf_pred)
    rf_metrics['cv_mean'] = float(rf_cv_scores.mean())
    rf_metrics['cv_std'] = float(rf_cv_scores.std())
    rf_metrics['feature_importances'] = {
        feat: float(imp) for feat, imp in zip(feature_names, rf_model.feature_importances_)
    }
    
    # Save Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(dt_model, os.path.join("models", "decision_tree.pkl"))
    joblib.dump(rf_model, os.path.join("models", "random_forest.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(le, os.path.join("models", "label_encoder.pkl"))
    
    metrics_summary = {
        'class_names': list(le.classes_),
        'feature_names': feature_names,
        'decision_tree': dt_metrics,
        'random_forest': rf_metrics
    }
    
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("=" * 60)
    print("MODEL TRAINING & EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Decision Tree Accuracy:  {dt_metrics['accuracy']*100:.2f}% (5-Fold CV: {dt_metrics['cv_mean']*100:.2f}%)")
    print(f"Random Forest Accuracy:  {rf_metrics['accuracy']*100:.2f}% (5-Fold CV: {rf_metrics['cv_mean']*100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()
