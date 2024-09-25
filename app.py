from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model/spam_detector.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)  # Debugging line to check data from front-end

    prediction = model.predict([data['text']])
    return jsonify({'prediction': int(prediction[0])})  # Ensure the response is properly formatted

if __name__ == '__main__':
    app.run(debug=True)
