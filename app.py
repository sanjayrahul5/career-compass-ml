from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        payload = request.get_json()
        print(payload)

        # Extract values from JSON data
        arr = [value for value in payload.values()]
        data = np.array(arr).astype(float)
        data = data.reshape(1, -1)
        print(data)

        # Load the trained model
        loaded_model = pickle.load(open("careerlast.pkl", "rb"))

        # Make predictions
        predictions = loaded_model.predict(data)
        print("pred --> ", predictions)

        pred = loaded_model.predict_proba(data)
        print(pred)

        # Apply threshold to predictions
        pred = pred > 0.05

        # Process predictions
        res = {}
        final_res = {}
        for j in range(16):
            if pred[0, j]:
                res[j] = j

        for key, value in res.items():
            if value != predictions[0]:
                final_res[key] = value
                print('final_res[index]:', final_res[key])

        return jsonify({'prediction': predictions[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
