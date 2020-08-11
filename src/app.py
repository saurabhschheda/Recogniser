import Preprocessor
import Trainer
import Predictor

import os
from flask import Flask, render_template, request, json, send_file
from werkzeug import secure_filename
import base64

app = Flask(__name__)

predictions = {"count": 0}
new_faces = 0

@app.route("/")
def home():
    return "Server working fine"

@app.route('/<string:page_name>/')
def static_page(page_name):
    return render_template('%s.html' % page_name)

def get_prediction(filename):
    preprocessor = Preprocessor.Preprocessor()
    # trainer = Trainer.Trainer(preprocessor)
    predictor = Predictor.Predictor(preprocessor)
    # image_name_list = os.listdir(os.path.join(os.getcwd(), "img"))
    # for image_name in image_name_list:
    #     image_path = os.path.join(os.path.join(os.getcwd(), "img"), image_name) 
    return predictor.predict([filename])

def save_file(file_stream, file_path):
    file_stream.replace(' ', '+')
    file_stream = file_stream.encode()
    with open(file_path, 'wb') as img:
        img.write(base64.decodebytes(file_stream))

@app.route('/predict', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = str(request.form['encoded_string'])
        file_path = "img/test_" + str(predictions["count"]) + ".png"
        save_file(f, file_path)
        prediction = get_prediction(file_path)
        predictions[file_path] = prediction
        predictions["count"] += 1
        print(predictions)
        response = json.dumps({"server_response":[{"header":"success"}]})
        return response

@app.route('/get_predictions', methods = ['GET', 'POST'])
def get_all_predictions():
    obj = json.dumps(predictions)
    return obj

@app.route('/add_face', methods = ['GET', 'POST'])
def add_new_face():
    if request.method == 'POST':
        global new_faces
        f = str(request.form['encoded_string'])
        file_path = "img/add_" + str(new_faces) + ".png"
        save_file(f, file_path)
        # f = request.files['file']
        # file_path = "img/add_" + str(new_faces) + ".png"
        new_faces += 1
        # f.save(file_path)
        name = request.form['name']
        trainer = Trainer.Trainer(Preprocessor.Preprocessor())
        trainer.update_dataset([file_path], name)
        print(name)
        response = json.dumps({"server_response":[{"header":"success"}]})
        return response
        # return "Done!"

@app.route('/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.root_path)
    return send_file(filename, mimetype='image/png') 

if __name__ == '__main__':
    image_name_list = os.listdir(os.path.join(os.getcwd(), "img"))
    for image_name in image_name_list:
        image_path = os.path.join(os.path.join(os.getcwd(), "img"), image_name)
        os.remove(image_path)
    app.run(host='0.0.0.0')
