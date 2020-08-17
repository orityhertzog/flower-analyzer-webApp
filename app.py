import glob
import os
import os.path
from flask import Flask, render_template, request

from analyzing import analyzing, detecting

app = Flask(__name__)
UPLOAD_FOLDER = '/static/uploads/'


@app.route('/', methods=['POST', 'GET'])
def home():
    # cleaning the upload directory from past analyzing requests
    files = glob.glob('static/uploads/*.jpg')
    for f in files:
        os.remove(f)
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', msg="no file has found")
        file = request.files['file']
        if file == '':
            return render_template('index.html', msg="file is empty")
        if file:
            path = os.path.abspath(os.path.dirname(__file__))
            img_path = UPLOAD_FOLDER + file.filename
            file.save(path + '/' + img_path)
            return render_template('index.html', img_src=img_path, file_name=file.filename)
        else:
            return render_template('index.html', msg="no file has found")
    else:
        return render_template('index.html')


@app.route('/analyze_img/<string:img_path>')
def analyze_img(img_path):
    full_path = 'static/uploads/' + img_path
    rect_img, rect_img_path, flower_obj, leaf_obj = detecting(full_path)
    if len(flower_obj) == 0 and len(leaf_obj) == 0:
        return render_template("analyze_img.html", msg="sorry, no objects were found")
    else:
        flower_msg, leaf_msg = analyzing(flower_obj, leaf_obj)
        return render_template("analyze_img.html", flower_msg=flower_msg, leaf_msg=leaf_msg)


if __name__ == "__main__":
    app.run(debug=True)
