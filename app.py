from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import io

app = Flask(__name__)
app.secret_key = "cloudpulse-local-research-prototype"
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
UPLOAD_DIR = BASE / "uploads"
MODEL_DIR = BASE / "models"
for folder in (DATA_DIR, UPLOAD_DIR, MODEL_DIR):
    folder.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "resource_usage.csv"
MODEL_FILE = MODEL_DIR / "cloudpulse_model.joblib"

REQUIRED = ["timestamp", "cpu", "memory", "network", "disk", "workloads"]
FEATURES = ["cpu", "memory", "network", "disk", "workloads", "hour_sin", "hour_cos"]

def make_demo_data(n=240):
    rng = np.random.default_rng(42)
    end = datetime.now().replace(second=0, microsecond=0)
    records = []
    for i in range(n):
        ts = end - timedelta(minutes=5*(n-1-i))
        hour = ts.hour + ts.minute/60
        wave = np.sin(i*0.115) + 0.45*np.sin(i*0.031)
        cpu = np.clip(48 + wave*17 + rng.normal(0, 5), 8, 96)
        memory = np.clip(55 + wave*10 + rng.normal(0, 4), 15, 94)
        network = np.clip(260 + wave*100 + rng.normal(0, 38), 10, 900)
        disk = np.clip(35 + wave*16 + rng.normal(0, 7), 1, 150)
        workloads = int(np.clip(110 + wave*42 + rng.normal(0, 12), 5, 500))
        records.append([ts.isoformat(timespec="minutes"), cpu, memory, network, disk, workloads])
    return pd.DataFrame(records, columns=REQUIRED)

def load_data():
    if not DATA_FILE.exists():
        make_demo_data().to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED).sort_values("timestamp").reset_index(drop=True)
    for col in ["cpu", "memory"]:
        df[col] = df[col].clip(0, 100)
    return df

def feature_frame(df):
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60
    x = df[["cpu", "memory", "network", "disk", "workloads"]].copy()
    x["hour_sin"] = np.sin(hour * np.pi/12)
    x["hour_cos"] = np.cos(hour * np.pi/12)
    return x[FEATURES]

def prepare_supervised(df):
    x = feature_frame(df).iloc[:-1].copy()
    y = df[["cpu", "memory"]].shift(-1).iloc[:-1].copy()
    return x, y

def train_model():
    df = load_data()
    if len(df) < 20:
        raise ValueError("Please provide at least 20 valid timestamped readings.")
    X, y = prepare_supervised(df)
    split = max(10, int(len(X)*0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=140, max_depth=9, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred, multioutput="uniform_average")
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred, multioutput="uniform_average")
    # Keep the hold-out metrics honest, then refit the deployed model on all available rows.
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4), "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test))}

def get_forecast():
    if not MODEL_FILE.exists():
        train_model()
    model = joblib.load(MODEL_FILE)
    df = load_data()
    X = feature_frame(df).tail(1)
    cpu, memory = model.predict(X)[0]
    cpu, memory = float(np.clip(cpu, 0, 100)), float(np.clip(memory, 0, 100))
    risk = "HIGH" if cpu >= 80 or memory >= 85 else ("MEDIUM" if cpu >= 65 or memory >= 75 else "LOW")
    advice = "Scale out / add capacity" if risk == "HIGH" else ("Monitor workload closely" if risk == "MEDIUM" else "No scaling action needed")
    return {"cpu": round(cpu, 1), "memory": round(memory, 1), "risk": risk, "advice": advice,
            "forecast_time": (df["timestamp"].iloc[-1] + timedelta(minutes=5)).isoformat(timespec="minutes")}

def dashboard_payload():
    df = load_data()
    model_metrics = None
    try:
        model_metrics = train_model()
    except Exception:
        model_metrics = None
    try:
        forecast = get_forecast()
    except Exception:
        forecast = None
    latest = df.tail(24)
    recent = df.tail(12).iloc[::-1].copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%d %b %H:%M")
    return {
        "metrics": {
            "cpu": round(float(df.cpu.mean()), 1), "memory": round(float(df.memory.mean()), 1),
            "network": round(float(df.network.mean()), 1), "disk": round(float(df.disk.mean()), 1),
            "records": int(len(df))
        },
        "chart": {"labels": latest.timestamp.dt.strftime("%H:%M").tolist(),
                  "cpu": latest.cpu.round(1).tolist(), "memory": latest.memory.round(1).tolist()},
        "recent": recent[REQUIRED].to_dict(orient="records"),
        "forecast": forecast, "model": model_metrics
    }

@app.route("/")
def index():
    return render_template("index.html", data=dashboard_payload())

@app.post("/api/train")
def api_train():
    try:
        result = train_model()
        return jsonify({"ok": True, "metrics": result, "forecast": get_forecast()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.post("/api/readings")
def add_reading():
    try:
        payload = request.get_json(force=True)
        row = {k: payload[k] for k in REQUIRED}
        row["timestamp"] = pd.to_datetime(row["timestamp"], errors="raise").isoformat(timespec="minutes")
        for k in REQUIRED[1:]:
            row[k] = float(row[k])
        if not (0 <= row["cpu"] <= 100 and 0 <= row["memory"] <= 100):
            raise ValueError("CPU and memory must be between 0 and 100.")
        df = load_data()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        if MODEL_FILE.exists():
            MODEL_FILE.unlink()
        return jsonify({"ok": True, "message": "Reading saved. Retrain the model to refresh predictions."})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.post("/api/upload")
def upload_csv():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Choose a CSV file.")
        return redirect(url_for("index"))
    try:
        df = pd.read_csv(file)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        df = df[REQUIRED].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in REQUIRED[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=REQUIRED).sort_values("timestamp")
        if len(df) < 20:
            raise ValueError("CSV must contain at least 20 valid rows.")
        if ((df.cpu < 0) | (df.cpu > 100) | (df.memory < 0) | (df.memory > 100)).any():
            raise ValueError("CPU and memory values must be between 0 and 100.")
        df.to_csv(DATA_FILE, index=False)
        if MODEL_FILE.exists():
            MODEL_FILE.unlink()
        flash(f"Imported {len(df)} valid readings. Train the model to generate new results.")
    except Exception as exc:
        flash("CSV upload failed: " + str(exc))
    return redirect(url_for("index"))

@app.get("/download")
def download_csv():
    return send_file(DATA_FILE, as_attachment=True, download_name="cloud_resource_usage.csv", mimetype="text/csv")

@app.get("/sample")
def sample_csv():
    sample = make_demo_data(80)
    buf = io.BytesIO(sample.to_csv(index=False).encode("utf-8"))
    return send_file(buf, as_attachment=True, download_name="cloudpulse_sample_dataset.csv", mimetype="text/csv")

@app.post("/api/reset")
def reset_data():
    make_demo_data().to_csv(DATA_FILE, index=False)
    if MODEL_FILE.exists():
        MODEL_FILE.unlink()
    return jsonify({"ok": True})

if __name__ == "__main__":
    load_data()
    app.run(host="127.0.0.1", port=5000, debug=True)
