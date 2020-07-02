import os
from flask import Flask, render_template, request, redirect
from flask_dropzone import Dropzone
import cv2
import glob
import timeit
from features import testImage, nameFlower, choose_similar_files, input_CNN
import joblib
import shutil
import uuid
import numpy as np


svm_linear = joblib.load('svm_linear.pkl')
fine_tuning = joblib.load('fine_tuning.pkl')

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    STATIC_PATH=os.path.join(basedir, 'static/upload'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=10,
    DROPZONE_MAX_FILES=100,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',
    DROPZONE_UPLOAD_BTN_ID='submit',
)

dropzone = Dropzone(app)


class Result:
    def __init__(self, predict, time, similar):
        self.predict = predict
        self.time = time
        self.similar = similar
        self.id = str(uuid.uuid4())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def handle_upload():
    shutil.rmtree(os.path.join(app.config['STATIC_PATH']))
    os.makedirs(os.path.join(app.config['STATIC_PATH']))

    for key, f in request.files.items():
        if key.startswith('file'):
            f.save(os.path.join(app.config['STATIC_PATH'], f.filename))

    return '', 204


@app.route('/form', methods=['POST'])
def handle_form():

    opt = request.form.get('opt')
    predict_dict = {}

    if opt == '0':
        for path, subdirs, files in os.walk(app.config['STATIC_PATH']):
            for filename in files:
                start = timeit.default_timer()

                imgfeat = testImage(os.path.join(
                    app.config['STATIC_PATH'], filename))
                preds2 = svm_linear.predict(imgfeat.reshape(1, -1))
                # print('Giá trị dự đoán linearSVM: ',nameFlower(preds2))
                stop = timeit.default_timer()
                predict_dict[filename] = Result(
                    nameFlower(preds2), round((stop-start), 2), similar=choose_similar_files(preds2))
    # print(predict_dict['OIP.jpg'].similar)
    if opt == '1':

        for path, subdirs, files in os.walk(app.config['STATIC_PATH']):
            for filename in files:
                start = timeit.default_timer()

                img = input_CNN(os.path.join(
                    app.config['STATIC_PATH'], filename))

                pred = fine_tuning.predict(img)
                rounded_pred = np.argmax(pred, axis=1)
                stop = timeit.default_timer()
                print('Giá trị dự đoán model CNN: ', nameFlower(rounded_pred))

                predict_dict[filename] = Result(
                    nameFlower(rounded_pred), round((stop-start), 2), similar=choose_similar_files(rounded_pred))

    return render_template('result.html', predict_list=predict_dict)


@app.route('/result')
def viewResult():
    return render_template('result.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(port=9212, debug=False, threaded=False)
    TEMPLATES_AUTO_RELOAD = True
