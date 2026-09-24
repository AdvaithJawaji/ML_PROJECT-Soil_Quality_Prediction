import os
import json
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def generate_visualizations():
    plots_dir = os.path.join("static", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    metrics_path = os.path.join("models", "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("metrics.json not found. Run train_models.py first.")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    class_names = metrics['class_names']
    feature_names = metrics['feature_names']
    dt_metrics = metrics['decision_tree']
    rf_metrics = metrics['random_forest']
    
    # Custom Styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    palette = ['#e74c3c', '#f39c12', '#2ecc71']
    
    # -------------------------------------------------------------
    # PLOT 1: Confusion Matrices Side-by-Side
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    dt_cm = np.array(dt_metrics['confusion_matrix'])
    rf_cm = np.array(rf_metrics['confusion_matrix'])
    
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[0].set_title('Decision Tree Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[1].set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # PLOT 2: Model Performance Comparison Bar Chart
    # -------------------------------------------------------------
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
    dt_values = [
        dt_metrics['accuracy'] * 100,
        dt_metrics['precision_macro'] * 100,
        dt_metrics['recall_macro'] * 100,
        dt_metrics['f1_macro'] * 100,
        dt_metrics['cv_mean'] * 100
    ]
    rf_values = [
        rf_metrics['accuracy'] * 100,
        rf_metrics['precision_macro'] * 100,
        rf_metrics['recall_macro'] * 100,
        rf_metrics['f1_macro'] * 100,
        rf_metrics['cv_mean'] * 100
    ]
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, dt_values, width, label='Decision Tree', color='#3498db')
    rects2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='#2ecc71')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Decision Tree vs Random Forest Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='lower right')
    
    # Add labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
                    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # PLOT 3: Feature Importance Comparison
    # -------------------------------------------------------------
    dt_importances = dt_metrics['feature_importances']
    rf_importances = rf_metrics['feature_importances']
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Decision Tree': [dt_importances[f] for f in feature_names],
        'Random Forest': [rf_importances[f] for f in feature_names]
    }).sort_values(by='Random Forest', ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(df_imp))
    
    ax.barh(y_pos - 0.2, df_imp['Decision Tree'], height=0.4, label='Decision Tree', color='#9b59b6')
    ax.barh(y_pos + 0.2, df_imp['Random Forest'], height=0.4, label='Random Forest', color='#1abc9c')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_imp['Feature'], fontsize=11)
    ax.set_xlabel('Relative Importance Weight', fontsize=12)
    ax.set_title('Feature Importance Comparison Across Soil Attributes', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # PLOT 4: Decision Tree Structure Visualization
    # -------------------------------------------------------------
    dt_model = joblib.load(os.path.join("models", "decision_tree.pkl"))
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(dt_model, max_depth=3, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, fontsize=10)
    ax.set_title('Trained Decision Tree Structure (First 3 Levels)', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "decision_tree_structure.png"), dpi=300)
    plt.close()
    
    print(f"Visualization plots successfully generated in '{plots_dir}'.")

if __name__ == "__main__":
    generate_visualizations()
