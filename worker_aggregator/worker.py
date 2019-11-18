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
args = parser.parse_args()
print(args)

@app.route('/')
def index():
   return 'health check'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        n_round = request.form['round']
        loss = request.form['loss']

        fname = secure_filename(f.filename)
        print(n_round, fname)
        f.save(os.path.join('/tmp/models', fname))

        filename = os.path.join('/tmp/models', fname)

        try:
            with open(filename, 'rb') as f:
                r = requests.post(args.master, files={'file': f},
                                  data={'round' : n_round, 'loss' : loss})
                print(r.text)
        except:
            print('FL Server is not connected!!')

        return 'success'

if __name__ == '__main__':
    # temporary model which will be passed worker aggregator
    if not os.path.isdir('/tmp/models'):
        os.mkdir('/tmp/models')

    app.run(host='0.0.0.0', debug=False)
