from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from lib.knn import predictClass

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__, template_folder='template', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["DEBUG"] = True


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    fileName = []
    images = np.array([])
    if 'file' not in request.files:
        return "File tidak ada"
    if 'file' not in request.files:
        return "File tidak memenuhi standard"
    file = request.files['file']
    if file.filename == '':
        filename = 'No selected file'
        return "SADKEK"
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        histogram, bin_edges = np.histogram(gray, bins=10, range=(0, 256))
        if histogram is not None:
            images = np.append([[images]], histogram)
        images = images.reshape(int(len(images)/10), 10)

    return render_template('process.html', histogram=images)


if __name__ == "__main__":
    app.run(debug=True)
