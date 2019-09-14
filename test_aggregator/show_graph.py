import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

def show_loss():
    with open("/tmp/loss.txt", 'r') as f:
        lines = f.readlines()
        xs, y1, y2 = [], [], []
        for line in lines:
            arr = line.split('\t')
            xs.append(arr[0])
            y1.append(arr[1])
            y2.append(arr[2])
        plt.plot(xs, y1, label=r'device 1')
        plt.plot(xs, y2, label=r'device 2')
        plt.title('loss')

        plt.legend()
        plt.savefig('./loss.png')
        plt.show()

def show_acc():
    with open("/tmp/acc.txt", 'r') as f:
        lines = f.readlines()
        xs, ys, = [], []
        for line in lines:
            arr = line.split('\t')
            xs.append(arr[0])
            ys.append(arr[1])
        plt.plot(xs, ys, label=r'acc')
        plt.title('acc')

        plt.legend()
        plt.savefig('./acc.png')
        plt.show()

# show_loss()
show_acc()
