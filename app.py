from flask import Flask, render_template, request, jsonify
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from datetime import datetime

app = Flask(__name__, static_folder="static")

# ✅ 1️⃣ 데이터 생성 및 모델 학습
X, y = make_classification(n_samples=500, n_features=8, n_informative=6, n_redundant=2, random_state=42)
X[:, 4] = np.clip(X[:, 4], np.percentile(X[:, 4], 5), np.percentile(X[:, 4], 95))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(C=0.05, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ 모델 및 스케일러가 새롭게 학습됨!")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["gender"]
    location = request.form["location"]
    age = int(request.form["age"])
    cause = request.form["cause"]
    fee_charged = float(request.form["fee_charged"])
    membership_period = int(request.form["membership_period"])
    number_of_claims = int(request.form["number_of_claims"])
    number_of_dependants = int(request.form["number_of_dependants"])

    gender_map = {"Male": 1, "Female": 2}
    cause_map = {"Road Traffic Accident": 3, "Accident At Work": 2, "Accident At Home": 1, "Other": 4}
    location_map = {
        "Kwekwe": 1, "Harare": 2, "Bulawayo": 3, "Masvingo": 4, "Marondera": 5,
        "Gweru": 6, "Rusape": 7, "Nyanga": 8, "Kadoma": 9, "Gwanda": 10
    }

    gender = gender_map.get(gender, 2)
    cause = cause_map.get(cause, 4)
    location = location_map.get(location, 5)  

    input_data = np.array([[gender, location, age, cause, fee_charged, membership_period, number_of_claims, number_of_dependants]])
    input_data_clipped = np.clip(input_data, X_train.min(axis=0), X_train.max(axis=0))
    input_data_scaled = scaler.transform(input_data_clipped)

    raw_probs = model.predict_proba(input_data_scaled)
    fraud_probability = round(raw_probs[0][1] * 100, 2)

    return render_template("result.html", fraud_probability=fraud_probability)

@app.route("/lime_result")
def lime_result():
    return render_template("lime_result.html")

@app.route("/lime_analysis", methods=["GET"])
def lime_analysis():
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=["Gender", "Location", "Age", "Cause", "Fee Charged", "Membership Period", "Number of Claims", "Dependants"],
        class_names=["Not Fraud", "Fraud"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=scaler.transform(np.array([[1, 5, 35, 3, 5000, 5, 2, 1]]))[0],
        predict_fn=model.predict_proba
    )

    lime_results = exp.as_list()
    importance_data = [{"feature": feature, "importance": importance} for feature, importance in lime_results]

    return jsonify(importance_data)


if __name__ == "__main__":
    app.run(debug=True)
