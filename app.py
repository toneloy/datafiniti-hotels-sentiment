from flask import Flask, jsonify, request, render_template
from keras.models import model_from_json
import pickle

app = Flask(__name__)


def predict_rating(text, model_, tokenizer_):
    text_vector = tokenizer_.texts_to_matrix([text])
    prediction = model_.predict(text_vector)
    
    return float(prediction[0, 0])


def load_model():
    with open('datafiniti_hotel_reviews_sentiment.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("datafiniti_hotel_reviews_sentiment_weights.h5")
    
    with open('datafiniti_hotel_reviews_sentiment_tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        
    return loaded_model, loaded_tokenizer
        

model, tokenizer = load_model()


@app.route('/evaluate', methods=["GET"])
def evaluate():
    text = request.args.get('t')
    if text:

        rating = predict_rating(text, model, tokenizer)

        return jsonify({'text': text, 'rating': rating})
    
    return jsonify({'text': '', 'rating': 0})


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)
