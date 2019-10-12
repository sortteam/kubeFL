# -*- coding: utf-8 -*-
import os
from flask import Flask, request
from werkzeug import secure_filename
import argparse
import requests

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--master', help='address of master aggregator',
                    required=True, type=str)
parser.add_argument("--threshold", type=int, default=1)
args = parser.parse_args()
print(args)

threshold = args.threshold
updates = []

@app.route('/')
def index():
   return 'health check'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        n_round = request.form['round']
        fname = secure_filename(f.filename)
        print(n_round, fname)
        f.save(os.path.join('/tmp/models', fname))

        global updates
        updates.append(fname)

        if len(updates) >= threshold:
            try:
                file_list = [os.path.join('/tmp/models', _name) for _name in updates]
                files = [('file', open(file, 'rb')) for file in file_list]

                r = requests.post(args.master,  files=files,
                                  data={'count': len(files), 'round' : n_round})

            except:
                print('FL Server is not connected!!')

            updates = []

        return 'success'

if __name__ == '__main__':
    if not os.path.isdir('/tmp/models'):
        os.mkdir('/tmp/models')

    app.run(host='0.0.0.0', debug=False)
