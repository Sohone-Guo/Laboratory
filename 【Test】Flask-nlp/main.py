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