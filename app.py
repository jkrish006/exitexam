import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def train_and_save_model(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.RandomState(42)
    n = 2000
    age = rng.randint(18, 75, size=n)
    income = rng.normal(50000, 20000, size=n).clip(5000, 200000)
    visit_freq = rng.poisson(4, size=n)
    avg_spend = rng.normal(60, 30, size=n).clip(5, 200)
    prefers_online = rng.binomial(1, 0.4, size=n)
    uses_mobile = rng.binomial(1, 0.5, size=n)
    time_spent = rng.normal(25, 10, size=n).clip(1, 180)
    categories = rng.randint(1, 10, size=n)

    X = np.vstack([age, income, visit_freq, avg_spend, prefers_online, uses_mobile, time_spent, categories]).T

    # Create a simple rule-based label plus some noise for synthetic training
    score_online = 0.03 * (age < 40) + 0.00001 * income + 0.2 * prefers_online + 0.15 * uses_mobile - 0.01 * visit_freq
    score_instore = 0.02 * (age >= 40) + 0.000005 * income + 0.25 * visit_freq + 0.1 * (avg_spend > 80)
    score_hybrid = 0.15 * (avg_spend <= 80) + 0.1 * (categories >= 5)
    scores = np.vstack([score_online, score_instore, score_hybrid]).T
    y = scores.argmax(axis=1)

    # Preprocessing: scale numeric features, one-hot encode small categorical/binary features
    # Feature order: [age, income, visit_freq, avg_spend, prefers_online, uses_mobile, time_spent, categories]
    numeric_idxs = [0, 1, 2, 3, 6, 7]
    categorical_idxs = [4, 5]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_idxs),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_idxs),
    ])

    pipe = make_pipeline(preprocessor, LogisticRegression(multi_class="multinomial", max_iter=400))
    pipe.fit(X, y)
    joblib.dump(pipe, path)
    return pipe


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return train_and_save_model(path)


app = Flask(__name__)
model = load_model()


MODE_MAP = {0: "Online", 1: "In-Store", 2: "Hybrid"}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form.get("age", 30))
        income = float(request.form.get("income", 50000))
        visit_freq = float(request.form.get("visit_freq", 4))
        avg_spend = float(request.form.get("avg_spend", 50))
        prefers_online = 1 if request.form.get("prefers_online") == "yes" else 0
        uses_mobile = 1 if request.form.get("uses_mobile") == "yes" else 0
        time_spent = float(request.form.get("time_spent", 20))
        categories = float(request.form.get("categories", 3))

        X = np.array([[age, income, visit_freq, avg_spend, prefers_online, uses_mobile, time_spent, categories]])
        pred_idx = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        proba_map = {MODE_MAP[i]: float(probs[i]) for i in range(len(probs))}

        return render_template("result.html", prediction=MODE_MAP[pred_idx], probabilities=proba_map, inputs=request.form)
    except Exception as e:
        return f"Error: {e}", 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
