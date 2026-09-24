import os
import json
import http.server
import socketserver
# pyrefly: ignore [missing-import]
import joblib
import numpy as np

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Load ML artifacts if available
try:
    dt_model = joblib.load(os.path.join(DIRECTORY, "models", "decision_tree.pkl"))
    rf_model = joblib.load(os.path.join(DIRECTORY, "models", "random_forest.pkl"))
    scaler = joblib.load(os.path.join(DIRECTORY, "models", "scaler.pkl"))
    le = joblib.load(os.path.join(DIRECTORY, "models", "label_encoder.pkl"))
    with open(os.path.join(DIRECTORY, "models", "metrics.json"), "r") as f:
        metrics_data = json.load(f)
    print("[TERRA.AI] ML models & metrics successfully loaded into memory.")
except Exception as e:
    print(f"[TERRA.AI] Note: Models will run on client inference ({e})")

class TerraHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                # Expecting: [pH, Nitrogen, Phosphorus, Potassium, Organic_Matter, Moisture]
                raw_features = np.array([
                    float(data.get("pH", 6.8)),
                    float(data.get("Nitrogen", 280.0)),
                    float(data.get("Phosphorus", 25.0)),
                    float(data.get("Potassium", 150.0)),
                    float(data.get("Organic_Matter", 3.0)),
                    float(data.get("Moisture", 25.0))
                ]).reshape(1, -1)

                scaled = scaler.transform(raw_features)

                dt_pred = dt_model.predict(scaled)[0]
                dt_probs = dt_model.predict_proba(scaled)[0].tolist()

                rf_pred = rf_model.predict(scaled)[0]
                rf_probs = rf_model.predict_proba(scaled)[0].tolist()

                response = {
                    "classes": list(le.classes_),
                    "decision_tree": {
                        "prediction": le.inverse_transform([dt_pred])[0],
                        "probabilities": dt_probs
                    },
                    "random_forest": {
                        "prediction": le.inverse_transform([rf_pred])[0],
                        "probabilities": rf_probs
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return
            except Exception as ex:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(ex)}).encode("utf-8"))
                return
                
        self.send_error(404, "Endpoint not found")

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), TerraHandler) as httpd:
        print("=" * 65)
        print(f"  TERRA.AI // Minimalist Web Dashboard Server Running")
        print(f"  Local URL: http://localhost:{PORT}")
        print("=" * 65)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer shutting down.")
