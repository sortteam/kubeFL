import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

def show_loss():
    n_device = 10
    with open("/tmp/loss.txt", 'r') as f:
        lines = f.readlines()
        for device in range(1, n_device + 1):
            xs, ys = [], []
            for line in lines:
                arr = line.split('\t')
                xs.append(arr[0])
                ys.append(arr[device])
            plt.plot(xs, ys, label=r'device ' + str(device))
            plt.title('loss')
            plt.xlabel('round')
            plt.ylabel('loss')

        plt.legend()
        plt.savefig('./loss.png')
        plt.show()

def show_acc():
    with open("/tmp/acc.txt", 'r') as f:
        lines = f.readlines()
        xs, ys, = [], []
        for line in lines:
            arr = line.split('\t')
            xs.append(arr[0]) # round
            ys.append(arr[1]) # acc
        plt.plot(xs, ys, label=r'acc')
        plt.title('acc')
        plt.xlabel('round')
        plt.ylabel('acc')

        plt.legend()
        plt.savefig('./acc.png')
        plt.show()

# show_loss()
# show_acc()
