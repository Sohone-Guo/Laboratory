import os
from flask import Flask, render_template
import requests
from flask import Flask, render_template, request
import time

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the user has entered
        try:
            input_language = request.form['from_language']
            output_language = request.form['to_language']
        except:errors.append("The output language is error")

        try:
            num_sentences = request.form['sentences']
            num_sentences = int(num_sentences)
        except:errors.append("The number of sentences is error")

        try:
            source_text = request.form['source_text']
        except:errors.append("The source text is error")
        
        if len(errors) == 0:
            print("Result:",input_language,output_language,num_sentences, source_text)
            results["text"] = source_text
            print(results)
            time.sleep(5)
    return render_template('index.html', errors=errors, results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    
from flask import Flask
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Perform these actions when the first
# request is made
@app.before_first_request
def load_model_to_app():
    # Load the model
    app.model = load_model('models/model.h5')
    
    # Save the graph to the app framework.
    app.graph = tf.get_default_graph()

app.route(‘/<image_path>’)
def classify(image_path):
    model = app.model
    graph = app.graph
    with graph.as_default():
        return(our_function(image_path, model))
