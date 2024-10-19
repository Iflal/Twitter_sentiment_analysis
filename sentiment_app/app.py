from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer from .sav files
with open('sentiment_analysis_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.sav', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the tweet text from the form submission
        tweet = request.form['tweet']

        # Vectorize the tweet
        tweet_vector = vectorizer.transform([tweet])

        # Make prediction using the loaded model
        prediction = model.predict(tweet_vector)

        # Return the result in JSON format
        return jsonify({'prediction': str(prediction[0])})

    except ValueError as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
