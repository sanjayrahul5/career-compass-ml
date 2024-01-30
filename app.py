from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        i = 0
        print(result)
        res = result.to_dict(flat=True)
        print("res:", res)
        arr1 = res.values()
        arr = ([value for value in arr1])

        data = np.array(arr).astype(float)

        data = data.reshape(1, -1)
        print(data)
        loaded_model = pickle.load(open("careerlast.pkl", "rb"))
        predictions = loaded_model.predict(data)

        print("pred --> ", predictions)
        pred = loaded_model.predict_proba(data)
        print(pred)

        pred = pred > 0.05

        i = 0
        j = 0
        index = 0
        res = {}
        final_res = {}
        while j < 16:
            if pred[i, j]:
                res[index] = j
                index += 1
            j += 1

        index = 0
        for key, values in res.items():
            if values != predictions[0]:
                final_res[index] = values
                print('final_res[index]:', final_res[index])
                index += 1

        job = {}
        index = 1

        data1 = predictions[0]
        print(data1)
        return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)