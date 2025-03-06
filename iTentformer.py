from sklearn.model_selection import KFold
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import time
from utils.EarlyStopping import EarlyStopping
from utils.Haversine_Loss import HaversineLoss

sys.path.append("../../")
from model import *
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from utils.bohai_diff import window_slice
import numpy as np
import matplotlib.pyplot as plt

"""
Limited by my code level and time factors, it is recommended to re-write the standardized training code according to 
your own needs, here can be for reference
"""
delta_cols = [8, 9, 10, 11]
intent_cols = [2]
src_cols = [2, 3, 4, 5]
tgt_cols = [2, 3, 4, 5] + [-10, -9]
in_cols = src_cols + delta_cols
local_intent_size = len(intent_cols)
intent_size = 8
input_size = 10
input_size_tcn = 8
output_size = 4
d_model = 128
num_channels = [32] * 2
concat_dim = input_size_tcn + num_channels[0]

kernel_size = 3
dropout = 0.2
clip = 0.1
batch_size = 16
input_length = 10
target_length = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion1 = nn.MSELoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)
HaversineLoss = HaversineLoss().cuda()


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    sample = 0
    ADE_list = []
    FDE_list = []
    rmse_cog_list = []
    rmse_sog_list = []
    with torch.no_grad():
        for idx in range(0, len(eval_idx_list), batch_size):
            batch_indices = eval_idx_list[idx:idx + batch_size]
            delta = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            src = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            tgt_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, src_cols] for i in batch_indices]).cuda()
            intent_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, intent_cols] for i in batch_indices]).cuda()

            intent, output = model(delta, src)
            value_output = output
            value_target = tgt_y

            mse_loss = criterion1(value_output, value_target)
            intent = intent.reshape(-1, intent.size(-1))
            intent_y = intent_y.reshape(-1, intent_y.size(-1))
            loss_int = criterion1(intent, intent_y)
            loss = awl(loss_int, mse_loss)

            value_output = value_output @ torch.from_numpy(transform_matrix).float().cuda() + torch.from_numpy(
                mean_values[:4]).float().cuda()
            value_target = value_target @ torch.from_numpy(transform_matrix).float().cuda() + torch.from_numpy(
                mean_values[:4]).float().cuda()
            dist = HaversineLoss(value_output[:, :, 1:3].float(), value_target[:, :, 1:3].float()).cuda()

            ADE_list.append(torch.mean(dist))
            dist_reshape = dist.reshape(-1, target_length)
            FDE_list.append(torch.mean(dist_reshape[:, -1]))

            rmse_sog_list.append(torch.sqrt(torch.mean((value_output[:, :, 3] - value_target[:, :, 3]) ** 2)))
            rmse_cog_list.append(torch.sqrt(torch.mean((value_output[:, :, 0] - value_target[:, :, 0]) ** 2)))

            pred_list.append(value_output.cpu())
            Y_list.append(value_target.cpu())

            total_loss += loss.item()
            sample += 1

        eval_loss = total_loss / sample
        print(name + " loss: {:.5f}".format(eval_loss))
        ADE = torch.sum(torch.tensor(ADE_list)) / sample
        print(" ADE: {:.5f}nmi-->{:.5f}m".format(ADE, ADE * 1852))
        FDE = torch.sum(torch.tensor(FDE_list)) / sample
        print(" FDE: {:.5f}nmi-->{:.5f}m".format(FDE, FDE * 1852))
        rmse_cog = torch.sum(torch.tensor(rmse_cog_list)) / sample
        print(" RMSE_COG: {:.5f}°".format(rmse_cog))
        rmse_sog = torch.sum(torch.tensor(rmse_sog_list)) / sample
        print(" RMSE_SOG: {:.5f}kn".format(rmse_sog))

        return eval_loss, ADE, FDE, rmse_cog, rmse_sog


def train(ep, parallel_train=False):
    model.train()
    total_loss = 0
    sample = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    ADE_list = []
    FDE_list = []
    rmse_cog_list = []
    rmse_sog_list = []
    for idx in range(0, len(train_idx_list), batch_size):
        batch_indices = train_idx_list[idx:idx + batch_size]
        delta = torch.stack([X_train[i][:input_length, in_cols] for i in batch_indices]).cuda()
        src = torch.stack([X_train[i][:input_length, in_cols] for i in batch_indices]).cuda()
        tgt_y = torch.stack(
            [X_train[i][input_length:input_length + target_length, src_cols] for i in batch_indices]).cuda()
        intent_y = torch.stack(
            [X_train[i][input_length:input_length + target_length, intent_cols] for i in batch_indices]).cuda()

        optimizer.zero_grad()

        intent, output = model(delta, src)

        value_output = output
        value_target = tgt_y

        mse_loss = criterion1(value_output, value_target)
        intent = intent.reshape(-1, intent.size(-1))
        intent_y = intent_y.reshape(-1, intent_y.size(-1))
        loss_int = criterion1(intent, intent_y)
        loss = awl(loss_int, mse_loss)

        value_output = value_output @ torch.from_numpy(transform_matrix).float().cuda() + torch.from_numpy(
            mean_values[:4]).float().cuda()
        value_target = value_target @ torch.from_numpy(transform_matrix).float().cuda() + torch.from_numpy(
            mean_values[:4]).float().cuda()
        dist = HaversineLoss(value_output[:, :, 1:3].float(), value_target[:, :, 1:3].float()).cuda()

        ADE_list.append(torch.mean(dist))
        dist_reshape = dist.reshape(-1, target_length)
        FDE_list.append(torch.mean(dist_reshape[:, -1]))

        rmse_sog_list.append(torch.sqrt(torch.mean((value_output[:, :, 3] - value_target[:, :, 3]) ** 2)))
        rmse_cog_list.append(torch.sqrt(torch.mean((value_output[:, :, 0] - value_target[:, :, 0]) ** 2)))

        total_loss += loss.item()
        sample += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()

        if idx > 0 and idx % (10 * batch_size) == 0:
            cur_loss = total_loss / sample
            print("Epoch {:4d} | lr {:.9f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_loss = 0.0
            sample = 0


def data_prepare(data, train_scale, valid_scale, lay_data=True):
    """
    先将时间窗拼接为3维->标准化->还原回原始形式->打乱每一个窗口->以窗口划分训练集和验证集->重新拼接
    """
    data_2lay = np.concatenate(data, axis=0)
    length = [len(l) for i, l in enumerate(data)]
    scaler_data = data_2lay[:, 2:-1]
    scaler = StandardScaler()
    scaler_data = scaler.fit_transform(scaler_data)
    mean_values = scaler.mean_
    std_values = scaler.scale_
    mean_values.astype(np.float32)
    std_values.astype(np.float32)

    data_2lay = np.concatenate((data_2lay[:, :2], scaler_data, data_2lay[:, -1:]), axis=-1)
    # 根据长度列表切割数组
    result_list = []
    start_idx = 0
    for leng in length:
        end_idx = start_idx + leng
        result_list.append(data_2lay[start_idx:end_idx])
        start_idx = end_idx
    return result_list, mean_values, std_values


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=5, sci_mode=False)

    sample = 5
    np.random.seed(42)
    torch.manual_seed(42)
    # 'MMSI','Length','Course','Lon_d','Lat_d','SOG','vx','vy', delta 'Course','Lon_d','Lat_d','SOG','vx','vy', 'UnixTime'
    data = pd.read_pickle(f"dataset/example_bohai.pkl")

    evaluation_scores = []
    pred_list = []
    Y_list = []
    start_time = time.time()

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for i, (train_indices, test_indices) in enumerate(k_fold.split(data)):

        train_data, mean_values, std_values = data_prepare([data[i] for i in train_indices], 0.6, 0.2)
        transform_matrix = np.diag(std_values[2:6])

        test_data = [data[i] for i in test_indices]
        test_2lay = np.concatenate(test_data, axis=0)
        length = [len(l) for i, l in enumerate(test_data)]
        scaler_data = test_2lay[:, 2:-1]
        scaler_data = (scaler_data - mean_values) / std_values

        test_2lay = np.concatenate((test_2lay[:, :2], scaler_data, test_2lay[:, -1:]), axis=-1)
        # 根据长度列表切割数组
        test_list = []
        start_idx = 0
        for leng in length:
            end_idx = start_idx + leng
            test_list.append(test_2lay[start_idx:end_idx])
            start_idx = end_idx

        valid_indices = np.random.choice([i for i in range(len(train_data))], size=5, replace=False)
        train_indices_set = set([i for i in range(len(train_data))])
        valid_indices_set = set(valid_indices)
        train_indices = np.array(list(train_indices_set - valid_indices_set))


        def create_window_slices(data):
            return [window_slice(trj, win_size=20, step=20) for trj in data]


        X_train_list = create_window_slices([train_data[i] for i in train_indices])
        X_valid_list = create_window_slices([train_data[i] for i in valid_indices])
        X_test_list = create_window_slices(test_list)

        X_train_list = np.concatenate(X_train_list, axis=0)
        X_valid_list = np.concatenate(X_valid_list, axis=0)
        X_test_list = np.concatenate(X_test_list, axis=0)

        X_train, X_valid, X_test = torch.tensor(X_train_list).float(), torch.tensor(
            X_valid_list).float(), torch.tensor(
            X_test_list).float()
        """---------------------------"""
        lr = 2e-4
        model = iTentformer(input_size_tcn, input_size, local_intent_size, output_size, concat_dim, input_length,
                              num_channels, kernel_size, d_model, dropout).to(device)
        awl = AutomaticWeightedLoss(2).cuda()
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters()}], lr=lr, weight_decay=0)

        best_vloss = 1e8
        lr_lower_bound = 1e-10
        vloss_list = []
        model_name = f"save_models/bohai_K{fold}.pt"
        early_stopping = EarlyStopping(patience=10, verbose=True)

        # trainning
        for ep in range(1, 301):
            train(ep, parallel_train=False)
            vloss, _, _, _, _ = evaluate(X_valid, name='Validation')
            tloss, _, _, _, _ = evaluate(X_test, name='Test')
            # 设置 EarlyStopping
            early_stopping(vloss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if vloss < best_vloss:
                with open(model_name, "wb") as f:
                    torch.save(model, f)
                    print("Saved model!\n")
                best_vloss = vloss

            if ep > 5 and vloss > max(vloss_list[-3:]) and lr > lr_lower_bound:
                lr /= 2.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            vloss_list.append(vloss)

        model = torch.load(open(model_name, "rb"))
        """
        You can test the model by applying the training process annotation to the following code
        """
        # tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(X_test, plot=False)
        # print('-' * 89)
        # print("K={}: ADE: {:.5f}nmi, FDE: {:.5f}nmi, COG_RMSE: {:.5f}°, SOG_RMSE: {:.5f}kn".format(fold, ADE, FDE,
        #                                                                                            rmse_cog, rmse_sog))
        # print('-' * 89)
        # evaluation_score = np.array([ADE, FDE, rmse_cog, rmse_sog])
        # evaluation_scores.append(evaluation_score)

        fold += 1

    end_time = time.time()
    print('-' * 89)
    print("Training time: {:.3f} s".format((end_time - start_time)))
    mean_evaluation_score = np.mean(evaluation_scores, axis=0)
    print("Mean evaluation score across all folds:", mean_evaluation_score)
