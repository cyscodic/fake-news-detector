from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = os.path.join('model', 'fake_news_model.pkl')
tfidf_path = os.path.join('model', 'tfidf_vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from request
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'})

    # Transform and predict
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0]

    label = 'REAL' if prediction == 1 else 'FAKE'
    conf_score = round(confidence[prediction] * 100, 1)

    return jsonify({
        'prediction': label,
        'confidence': conf_score
    })

if __name__ == '__main__':
    app.run(debug=True)