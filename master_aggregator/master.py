# -*- coding: utf-8 -*-
import os, re
import torch
import copy
from flask import Flask, request
from werkzeug import secure_filename
import argparse
import boto3
from tensorboardX import SummaryWriter

from requests import get  # to make GET request
s3 = boto3.resource('s3')

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--web_model', help='init_model',
                default='https://ywj-horovod.s3.ap-northeast-2.amazonaws.com/torchmodels/model.pt')
parser.add_argument('--model', help='path which will be downloaded', default='/tmp/init_model.pt')
parser.add_argument("--threshold", type=int, default=10)
parser.add_argument("--lr", type=float, default=1.0)
args = parser.parse_args()
print(args)

threshold = args.threshold
updates = []
total_loss = []

def download(url, file_name):
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)

@app.route('/')
def index():
   return 'test'

@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        f = request.files.get('file')
        fname = secure_filename(f.filename)
        f.save(os.path.join('/tmp', fname))
        return 'upload success'

def cal_mean_weight(weight_paths):
    """
    :param weight_paths: list[str,]
    :return:
    """
    weights = [torch.load(os.path.join('/tmp/models', filename))
               for filename in weight_paths]
    pre_model = dict(torch.load(args.model))

    model = copy.deepcopy(weights[0])
    for layer_k, layer_v in model.items():
        mean = torch.mean(torch.stack([weight[layer_k] for weight in weights]), dim=0)
        model[layer_k] = mean * args.lr

        # W_t = W_(t-1) + n * H
        model[layer_k] += pre_model[layer_k]

    # Remove before weight
    if os.path.isfile(args.model):
        os.remove(args.model)

    torch.save(model, args.model) # overwrite
    data = open(args.model, 'rb')
    s3.Bucket('ywj-horovod').put_object(Key='torchmodels/model.pt',
                                    Body=data, ACL='public-read')

    return model

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':

        f = request.files.get('file')
        n_round = request.form['round']
        loss = request.form['loss']
        fname = secure_filename(f.filename)
        f.save(os.path.join('/tmp/models', fname))

        global updates
        updates.append(fname)

        global total_loss
        total_loss.append(loss)

        if len(updates) >= threshold:
            new_weight = cal_mean_weight(updates)

            # This line for save temporary
            torch.save(new_weight, os.path.join('/tmp', str(n_round) + '.pt'))
            print(total_loss)

            with open("/tmp/loss.txt", "a") as f:
                f.write(str(n_round) + ''.join('\t'+l for l in total_loss) + '\n')

            graph = dict()
            for i, _ in enumerate(range(len(updates))):
                key = str(updates[i]).rsplit('-', 1)[0]
                graph[key] = torch.tensor(float(total_loss[i]), dtype=torch.float32).item()
            print(graph)

            writer = SummaryWriter('/tmp/logger')
            writer.add_scalars('devices', graph, int(n_round))
            writer.close()

            updates = []
            total_loss = []
            # Test for show loss, acc with number of Round
            print('Round', n_round, 'Updated!!')

        return 'success'

def purge(dir="/tmp"):
    for f in os.listdir(dir):
        if '.pt' in f:
            os.remove(os.path.join(dir, f))

if __name__ == '__main__':
    purge()
    # Load init Model
    download(url=args.web_model, file_name=args.model)

    if not os.path.isdir('/tmp/models'):
        os.mkdir('/tmp/models')

    if os.path.exists('/tmp/loss.txt'):
        os.remove('/tmp/loss.txt')

    app.run(host='0.0.0.0', debug=False)
