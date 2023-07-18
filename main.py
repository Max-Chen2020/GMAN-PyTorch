
import argparse
import time
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test import test

# add device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_nodes', default=200,
                    help='number of nodes')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='./data/pems.h5',
                    help='traffic file')
parser.add_argument('--SE_file', default='./data/SE(PeMS).txt',
                    help='spatial embedding file')
parser.add_argument('--model_file', default='./data/GMAN_pems.pkl',
                    help='save the model to disk')
parser.add_argument('--log_file', default='./data/log',
                    help='log file')
args = parser.parse_args()
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
T = 24 * 60 // args.time_slot  # Number of time steps in one day
# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'speed mean:   {mean[0]:.4f}\t\tstd:   {std[0]:.4f}')
log_string(log, f'flow mean:   {mean[1]:.4f}\t\tstd:   {std[1]:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE, mean, std
# build model
log_string(log, 'compiling model...')

model = GMAN(SE.to(device), args, bn_decay=0.1)
loss_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    start = time.time()

    # add device variable to train and test functions
    loss_train, loss_val = train(device, model, args, log, loss_criterion, optimizer, scheduler)
    plot_train_val_loss(loss_train, loss_val, 'figure/train_val_loss.png')
    trainPred, valPred, testPred = test(device, args, log)
  
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
    # trainPred, valPred, testPred = trainPred.cpu(), valPred.cpu(), testPred.cpu()
    # trainPred_ = trainPred.numpy().reshape(-1, trainY.shape[-1])
    # trainY_ = trainY.numpy().reshape(-1, trainY.shape[-1])
    # valPred_ = valPred.numpy().reshape(-1, valY.shape[-1])
    # valY_ = valY.numpy().reshape(-1, valY.shape[-1])
    # testPred_ = testPred.numpy().reshape(-1, testY.shape[-1])
    # testY_ = testY.numpy().reshape(-1, testY.shape[-1])

    # Save training, validation and testing datas to disk
    # l = [trainPred_, trainY_, valPred_, valY_, testPred_, testY_]
    # name = ['trainPred', 'trainY', 'valPred', 'valY', 'testPred', 'testY']
    # l = [testPred_, testY_]
    # name = ['testPred', 'testY']
    # for i, data in enumerate(l):
    #     np.savetxt('./figure/' + name[i] + '.txt', data, fmt='%s')
        
    # Plot the test prediction vs target（optional)
    num_nodes = args.num_nodes
    fig = plt.figure(figsize=(20, 280))
    for k in range(5):
        plt.subplot(num_nodes, 1, k + 1)
        for j in range(len(testPred)):
            c, d = [], []
            for i in range(12):
                c.append(testPred[j, i, k, 0])
                d.append(testY[j, i, k, 0])
            plt.plot(range(1 + j, 12 + 1 + j), c, c='b')
            plt.plot(range(1 + j, 12 + 1 + j), d, c='r')
    fig.suptitle('Test prediction vs Target', fontsize = 14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./figure/test_results.png')
