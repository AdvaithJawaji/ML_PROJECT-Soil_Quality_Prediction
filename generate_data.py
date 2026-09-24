import os
import numpy as np
import pandas as pd

def generate_soil_dataset(num_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Feature distributions based on typical agricultural soil tests
    pH = np.random.uniform(4.5, 8.8, num_samples)
    N = np.random.uniform(50.0, 480.0, num_samples)       # Nitrogen (kg/ha)
    P = np.random.uniform(5.0, 55.0, num_samples)         # Phosphorus (kg/ha)
    K = np.random.uniform(60.0, 320.0, num_samples)       # Potassium (kg/ha)
    organic_matter = np.random.uniform(0.5, 5.5, num_samples)  # Organic Matter (%)
    moisture = np.random.uniform(5.0, 55.0, num_samples)       # Soil Moisture (%)
    
    # 2. Agronomic quality scoring function
    scores = np.zeros(num_samples)
    
    for i in range(num_samples):
        score = 0.0
        
        # pH score (optimal: 6.0 - 7.5)
        if 6.0 <= pH[i] <= 7.5:
            score += 25.0
        elif 5.5 <= pH[i] < 6.0 or 7.5 < pH[i] <= 8.2:
            score += 15.0
        else:
            score += 5.0
            
        # Nitrogen score (optimal: > 280)
        if N[i] >= 280:
            score += 20.0
        elif N[i] >= 160:
            score += 12.0
        else:
            score += 4.0
            
        # Phosphorus score (optimal: > 25)
        if P[i] >= 25:
            score += 15.0
        elif P[i] >= 14:
            score += 9.0
        else:
            score += 3.0
            
        # Potassium score (optimal: > 150)
        if K[i] >= 150:
            score += 15.0
        elif K[i] >= 100:
            score += 9.0
        else:
            score += 3.0
            
        # Organic Matter score (optimal: > 3.0%)
        if organic_matter[i] >= 3.0:
            score += 15.0
        elif organic_matter[i] >= 1.5:
            score += 9.0
        else:
            score += 3.0
            
        # Moisture score (optimal: 20% - 40%)
        if 20.0 <= moisture[i] <= 40.0:
            score += 10.0
        elif 12.0 <= moisture[i] < 20.0 or 40.0 < moisture[i] <= 48.0:
            score += 6.0
        else:
            score += 2.0
            
        # Add slight natural noise / micro-variability (+/- 4.0 points)
        score += np.random.normal(0, 2.5)
        scores[i] = score

    # 3. Categorize into Low, Medium, High quality
    # Score range ~ 20 to 100
    q33, q66 = np.percentile(scores, [33.3, 66.6])
    
    quality = []
    for s in scores:
        if s < q33:
            quality.append("Low")
        elif s < q66:
            quality.append("Medium")
        else:
            quality.append("High")
            
    df = pd.DataFrame({
        'pH': np.round(pH, 2),
        'Nitrogen': np.round(N, 1),
        'Phosphorus': np.round(P, 1),
        'Potassium': np.round(K, 1),
        'Organic_Matter': np.round(organic_matter, 2),
        'Moisture': np.round(moisture, 2),
        'Soil_Quality': quality
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_soil_dataset(num_samples=1500)
    output_path = os.path.join("data", "soil_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} rows.")
    print("Class distribution:")
    print(df['Soil_Quality'].value_counts())
