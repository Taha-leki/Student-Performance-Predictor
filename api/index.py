
import json
from pathlib import Path

import joblib
import pandas as pd
from http.server import BaseHTTPRequestHandler

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

FEATURES = [
    "age",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "paid",
    "activities",
    "higher",
    "traveltime",
    "goout",
]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_json(200, {
            "message": "Student Performance Predictor API is running",
            "features": FEATURES,
        })

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))

            if not isinstance(payload, dict):
                raise ValueError("Send a JSON object containing the student features.")

            missing = [name for name in FEATURES if name not in payload]
            if missing:
                self.send_json(400, {
                    "error": "Missing required features",
                    "missing": missing,
                })
                return

            row = {name: payload[name] for name in FEATURES}
            student = pd.DataFrame([row])

            model = joblib.load(MODEL_PATH)
            prediction = float(model.predict(student)[0])

            self.send_json(200, {
                "predicted_grade": round(max(0, min(20, prediction)), 2),
                "scale": "0-20",
            })

        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception:
            self.send_json(500, {
                "error": "Prediction failed. Check the deployment logs."
            })

    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
