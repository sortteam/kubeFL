"""
    init model unit test
"""
import os
import copy
import torch
import subprocess
import requests
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import socket
from datetime import datetime
from requests import get  # to make GET request

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def download(url, file_name):
    # Idempotency
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)

def cal_delta_weight(prev_model, new_model):
    weights = [dict(prev_model.state_dict()),
               dict(new_model.state_dict())]
    model = dict(prev_model.named_parameters())

    for layer_k, layer_v in model.items():
        # if use mean calculate
        # mean = torch.mean(torch.stack([weight[layer_k] for weight in weights]), dim=0)
        delta = weights[1][layer_k] - weights[0][layer_k]
        model[layer_k].data.copy_(delta)
    return model

def train(model, train_loader, optimizer, epochs):
    prev_model = copy.deepcopy(model)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                        loss.item(), correct/len(data) * 100))


    model_tag = str(socket.gethostname()) + '-' + str(datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3])
    model = cal_delta_weight(prev_model, model)
    filename = os.path.join('./models', model_tag) + '.pt'
    torch.save(model, filename)
    print(filename, 'saved!')

    return filename

def main(args):
    # Load init Model
    download(url=args.web_model, file_name=args.model)

    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum)
    model.load_state_dict(torch.load(args.model))
    print(model)

    # Load Data
    flatten = lambda l: [item for sublist in l for item in sublist]
    if os.path.exists(args.data_path):
        load_data = torch.load(args.data_path)
        load_data = flatten(load_data)
        train_loader = torch.utils.data.DataLoader(load_data,
                                                   batch_size=64,
                                                   shuffle=True,)

        filename = train(model, train_loader, optimizer, args.epoch)

        try:
            with open(filename, 'rb') as f:
                r = requests.post(args.FL_server, files={'file': f},
                                  data={'round' : args.round})
                print(r.text)
        except:
            print('FL Server is not connected!!')


if __name__ == '__main__':
    # Idempotency > remove all local models when start
    if os.path.isdir('./models'):
        subprocess.run(['rm', '-rf', './models'])
        os.mkdir('./models')

    parser = argparse.ArgumentParser()
    parser.add_argument('--web_model', help='init_model',
                default='https://ywj-horovod.s3.ap-northeast-2.amazonaws.com/torchmodels/model.pt')
    parser.add_argument('--model', help='path which will be downloaded', default='/tmp/init_model.pt')
    parser.add_argument('--data_path', help='train data', default='/tmp/data.pt')
    parser.add_argument('--lr', help='learning rate', default=0.01, type=float)
    parser.add_argument('--momentum', help='momentum', default=0.5, type=float)
    parser.add_argument('--epoch', help='number of epoch', default=50, type=int)
    parser.add_argument('--round', help='number of round', type=int)
    parser.add_argument('--FL_server', help='FL_server ip address', default='http://15.164.78.19:5000/upload', type=str)
    known_args, _ = parser.parse_known_args()
    print(known_args)
    main(args=known_args)
