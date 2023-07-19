import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.nanmean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.nanmean(rmse))
    mape = mape * mask
    mape = torch.nanmean(mape)
    return mae, rmse, mape


def seq2instance(data, num_his, num_pred):
    # add traffic variable dimension 
    if len(data.shape) == 3:
      num_step, dims, var = data.shape
    else:
      num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    if len(data.shape) == 3:
      x = torch.zeros((num_sample, num_his, dims, var))
      y = torch.zeros((num_sample, num_pred, dims, var))
    else:
      x = torch.zeros((num_sample, num_his, dims))
      y = torch.zeros((num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def load_data(args):
    # Traffic
    speed = pd.read_hdf(args.traffic_file, key = 'speed', mode = 'r')
    flow = pd.read_hdf(args.traffic_file, key = 'flow', mode = 'r')
    traffic = np.stack((speed.values, flow.values), axis = -1)
    traffic = torch.from_numpy(traffic)
    # train/val/test
    num_step = speed.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    # normalization
    mean, std = torch.mean(trainX, dim = (0, 1, 2)), torch.std(trainX, dim = (0, 1, 2))
    trainX[:, :, :, 0] = (trainX[:, :, :, 0] - mean[0]) / std[0]
    trainX[:, :, :, 1] = (trainX[:, :, :, 1] - mean[1]) / std[1]
    valX[:, :, :, 0] = (valX[:, :, :, 0] - mean[0]) / std[0]
    valX[:, :, :, 1] = (valX[:, :, :, 1] - mean[1]) / std[1]
    testX[:, :, :, 0] = (testX[:, :, :, 0] - mean[0]) / std[0]
    testX[:, :, :, 1] = (testX[:, :, :, 1] - mean[1]) / std[1]

    # spatial embedding
    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

    # temporal embedding
    time = pd.DatetimeIndex(speed.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                // (args.time_slot * 60)
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    # shape = (num_sample, 2)
    
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


# dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Physics-informed loss
def physical_loss(pred, label):

    # filtering 0s and nans, recalculate weights 
    mask = torch.ne(label, 0).type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)

    # extract traffic varialbes: v for speed, q for flow and k for density 
    v = pred[:, :, :, 0]
    q = pred[:, :, :, 1]
    k = torch.div(q, v)

    # parameters: free flow speed, critical density and backward wave speed
    v_f = 60

    # merge and recover shape, apply mask
    loss = torch.stack((torch.square(q / v_f - k), torch.square(v_f * k - q)), axis = -1)
    loss = torch.multiply(loss, mask)
    loss = torch.nanmean(loss)
    return loss 

# Weighted total loss
def wt_loss(pred, label, alpha = 0.5):
    dl_loss = torch.nn.MSELoss()
    return (1 - alpha) * dl_loss(pred, label), alpha * physical_loss(pred, label)
    


# The following function can be replaced by 'loss = torch.nn.L1Loss()  loss_out = loss(pred, target)
def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss


# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))
