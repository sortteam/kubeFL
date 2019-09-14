# -*- coding: utf-8 -*-
import os
import numpy as np
import PIL.Image as img
from flask import Flask, render_template, request
from werkzeug import secure_filename
import argparse

app = Flask(__name__)

@app.route('/')
def index():
   return 'test'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        fname = secure_filename(f.filename)
        f.save(os.path.join('/tmp', fname))
        return 'success'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
