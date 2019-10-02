# -*- coding: utf-8 -*-
import os
from flask import Flask, request
from werkzeug import secure_filename
import argparse
import requests

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--master', help='address of master aggregator',
                    default='http://15.164.78.19:5000/upload')
parser.add_argument("--threshold", type=int, default=2)
args = parser.parse_args()
print(args)

threshold = args.threshold

@app.route('/')
def index():
   return 'health check'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        fname = secure_filename(f.filename)
        print(fname)
        f.save(os.path.join('/tmp/models', fname))

        global updates
        updates.append(fname)

        if len(updates) >= threshold:
            try:
                file_list = [os.path.join('/tmp/models', _name) for _name in updates]
                files = [eval(f'("inline", open("{file}", "rb"))') for file in file_list]

                r = requests.post(args.master,  files=files,
                                  data={'count': len(files)})
                print(r.text)

            except:
                print('FL Server is not connected!!')

            updates = []

        return 'success'
