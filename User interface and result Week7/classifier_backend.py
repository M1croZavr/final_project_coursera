from sentiment_classifier import SentimentClassifier
from codecs import open
import time
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)

print('Preparing classifier...')
start_time = time.time()
classifier = SentimentClassifier()
print(f'Classifier is ready, time required {time.time() - start_time} seconds.')

# Temporary redirect main page to predictor
@app.route('/')
def temp():
    return redirect(url_for('predictor'))

@app.route('/predictor', methods=['POST', 'GET'])
def predictor(prediction_message='', probability=''):
    if request.method == 'POST':
        text = request.form['text']
        prediction_message = classifier.get_prediction_message(text)
        class_label, probability = classifier.predict_text(text)
        with open('classifier_logs.txt', 'a', 'utf-8') as logfile:
            logfile.write('<response>\n')
            logfile.write(text + '\n')
            logfile.write(prediction_message + '\n')
            logfile.write('</response>' + '\n')
    return render_template('predict_page.html',
                            prediction_message=prediction_message,
                            probability=str(probability * 100))

@app.route('/<name>')
def rick_roll(name):
    return render_template('rick_roll.html',
                           name=name)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
