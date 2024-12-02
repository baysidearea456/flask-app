from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("random_forest_model_v2.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [data["feature1"], data["feature2"], data["feature3"]]  # 必要な特徴量に合わせて修正
    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
