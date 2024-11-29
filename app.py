from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# モデルのロード
model = joblib.load("random_forest_model_v2.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # 入力データの形を整形
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
