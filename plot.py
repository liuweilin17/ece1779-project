import matplotlib
matplotlib.use("TkAgg")
from  matplotlib import pyplot as plt
import json

def transfer(filename):
    train_losses, valid_losses, test_losses = [], [], []
    train_acc, valid_acc, test_acc = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if 'epoch' not in line: continue
            line_arr = line.strip().split(',')
            for kv in line_arr:
                k, v = kv.strip().split(':')
                if k == 'trainloss':
                    train_losses.append(float(v))
                elif k == 'validloss':
                    valid_losses.append(float(v))
                elif k == 'testloss':
                    test_losses.append(float(v))
                elif k == 'trainacc':
                    train_acc.append(float(v))
                elif k == 'validacc':
                    valid_acc.append(float(v))
                elif k == 'testacc':
                    test_acc.append(float(v))
                else: pass
    return [train_losses, valid_losses, test_losses, train_acc, valid_acc, test_acc]

if __name__ == '__main__':

    log_name = "mlp.log"

    train_losses, valid_losses, test_losses, train_acc, valid_acc, test_acc = \
    transfer(log_name)
    x = list(range(len(train_losses)))

    plt.figure(1)
    plt.plot(x, train_losses)
    plt.plot(x, test_losses)
    plt.plot(x, valid_losses)
    plt.legend(['training loss', 'validation loss', 'testing loss'], loc='upper right')
    plt.show()

    plt.figure(2)
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.plot(x, valid_acc)
    plt.legend(['training accuracy', 'validation accuracy', 'testing accuray'], loc='lower right')
    plt.show()


