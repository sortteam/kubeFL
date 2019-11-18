import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

######################################################
# Change Model which you want to run on Mobile Device
######################################################
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
######################################################

"""
  calculate accuracy and create `/tmp/loss.txt` file every each rounds.
"""

if __name__ == '__main__':
    if os.path.exists('/tmp/acc.txt'):
        os.remove('/tmp/acc.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', help='max number of round', type=int, default=10)
    args, _ = parser.parse_known_args()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    test_data = datasets.MNIST('/tmp/MNIST', train=False,
                             transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=len(test_data), shuffle=True)

    model = Net()
    maxlen = args.maxlen

    file_list = ['/tmp/init_model.pt']
    file_list += [str(i) + '.pt' for i in range(1, maxlen + 1)]
    print(file_list)

    for n_round, file in enumerate(file_list):
        file = os.path.join('/tmp', file)
        model.load_state_dict(torch.load(file))
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()

                print('Acc: {:.2f}'.format(correct / len(data) * 100))
                with open("/tmp/acc.txt", "a") as f:
                    f.write(str(n_round) + '\t' + str(correct / len(data) * 100) + '\n')
