from sklearn.model_selection import KFold
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from torch.autograd import Variable
import torch.optim as optim
import geopandas as gpd
import torch.nn.functional as F
import sys
import time
from traj_prediction.EarlyStopping import EarlyStopping
from traj_prediction.Haversine_Loss import HaversineLoss, haversine_distance_2arrays
from matplotlib.lines import Line2D

sys.path.append("../../")
from traj_prediction.my_model.iTentformer.model import \
    IntentFlowNet  # from traj_prediction.my_model.IFNet_no_decoder.model import IntentFlowNet
from traj_prediction.my_model.AutomaticWeightedLoss import AutomaticWeightedLoss
from traj_prediction.c1_bohai_diff import window_slice
import numpy as np
import matplotlib.pyplot as plt

delta_cols = [6, 7, 8, 9]
intent_cols = [2]  # [-8, -7, -6, -5, -4, -3, -2, -1]
src_cols = [2, 3, 4, 5]
tgt_cols = [2, 3, 4, 5] + [-10, -9]
in_cols = src_cols + delta_cols
local_intent_size = len(intent_cols)
intent_size = 8  # 局部意图编码维度
input_size = 10  # 转置后时间输入
input_size_tcn = 8
output_size = 4  # 运动信息+全局意图
d_model = 128
num_channels = [32] * 2
concat_dim = input_size_tcn + num_channels[0]  # 40  # d_model + 32

kernel_size = 3
dropout = 0.2
clip = 0.1
batch_size = 16
input_length = 10
target_length = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion1 = nn.MSELoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数,包括了softmax和logloss,最后一层不需要激活
HaversineLoss = HaversineLoss().cuda()


def evaluate(X_data, plot=False, plot_distribution=False, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    sample = 0
    count = 0
    ADE_list = []
    FDE_list = []
    rmse_cog_list = []
    rmse_sog_list = []
    with torch.no_grad():
        for idx in range(0, len(eval_idx_list), batch_size):
            batch_indices = eval_idx_list[idx:idx + batch_size]
            delta = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            src = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            # tgt = torch.stack([X_data[i][input_length - 1, tgt_cols] for i in batch_indices]).unsqueeze(
            #     1).cuda()  # 测试时仅能用第一个
            tgt_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, src_cols] for i in batch_indices]).cuda()
            intent_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, intent_cols] for i in batch_indices]).cuda()

            intent, output, attn_weights = model(delta, src)

            # 序列one-hot不能直接用交叉熵，模型最后一层要激活
            value_output = output
            value_target = tgt_y

            # 计算数值部分的 MSE 损失
            mse_loss = criterion1(value_output, value_target)
            # 计算 one-hot 部分的交叉熵损失
            # ce_loss = criterion2(one_hot_output[:, -1, :], one_hot_target[:, -1, :])

            # intent_list = torch.concat(intent_list)
            intent = intent.reshape(-1, intent.size(-1))
            # intent = torch.sigmoid(intent)
            # intent_y_list = [intent_y] * 10
            # intent_y = torch.cat(intent_y_list, dim=0)
            # intent_y = intent_y.reshape(-1, intent_y.size(-1))
            # intent = torch.sigmoid(intent)
            intent_y = intent_y.reshape(-1, intent_y.size(-1))
            loss_int = criterion1(intent, intent_y)

            # loss = loss_int + mse_loss / (mse_loss / loss_int + 1e-8).detach() + ce_loss / (
            #     ce_loss / loss_int + 1e-8).detach()
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


def data_prepare_DMA(data, train_scale, valid_scale, lay_data=True):
    """
    先将时间窗拼接为3维->标准化->还原回原始形式->打乱每一个窗口->以窗口划分训练集和验证集->重新拼接
    """
    # 拼成2维并标准化
    data_2lay = np.concatenate(data, axis=0)
    length = [len(l) for i, l in enumerate(data)]
    scaler_data = data_2lay[:, 2:-11]
    scaler = StandardScaler()
    scaler_data = scaler.fit_transform(scaler_data)
    mean_values = scaler.mean_
    std_values = scaler.scale_
    mean_values.astype(np.float32)
    std_values.astype(np.float32)

    data_2lay = np.concatenate((data_2lay[:, :2], scaler_data, data_2lay[:, -11:]), axis=-1)
    # 根据长度列表切割数组
    result_list = []
    start_idx = 0
    for leng in length:
        end_idx = start_idx + leng
        result_list.append(data_2lay[start_idx:end_idx])
        start_idx = end_idx

    return result_list, mean_values, std_values


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.max_rows", 10000000)
    pd.set_option("display.width", 100000)
    pd.set_option("display.max_colwidth", 100000)
    plt.rcParams['savefig.dpi'] = 800  # 图片像素
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=5, sci_mode=False)

    # 'MMSI', 'Length', 'COG','Lon_d','Lat_d','SOG', delta 'COG','Lon_d','Lat_d','SOG',
    # 'UnixTime', 'Intent(2dim)', local intent(8dim)
    sample = 15
    np.random.seed(42)
    torch.manual_seed(42)
    data = pd.read_pickle(f"../../data/DMA/net_data_{sample}min_Ti.pkl")
    X_data, mean_values, std_values = data_prepare_DMA(data, 0.6, 0.2)
    transform_matrix = np.diag(std_values[:4])

    evaluation_scores = []  # 保存每一折的评估分数
    pred_list = []
    Y_list = []
    start_time = time.time()

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for i, (train_indices, test_indices) in enumerate(k_fold.split(X_data)):
        valid_indices = np.random.choice(train_indices, size=20, replace=False)
        train_indices_set = set(train_indices)
        valid_indices_set = set(valid_indices)
        train_indices = np.array(list(train_indices_set - valid_indices_set))


        # 对每条轨迹切片
        def create_window_slices(data):
            return [window_slice(trj, win_size=20, step=1) for trj in data]


        X_train_list = create_window_slices([X_data[i] for i in train_indices])
        X_valid_list = create_window_slices([X_data[i] for i in valid_indices])
        X_test_list = create_window_slices([X_data[i] for i in test_indices])

        X_train_list = np.concatenate(X_train_list, axis=0)
        X_valid_list = np.concatenate(X_valid_list, axis=0)
        X_test_list = np.concatenate(X_test_list, axis=0)
        # np.random.shuffle(X_train_list)
        # np.random.shuffle(X_valid_list)
        # np.random.shuffle(X_test_list)  # 破坏轨迹段顺序

        X_train, X_valid, X_test = torch.tensor(X_train_list).float(), torch.tensor(
            X_valid_list).float(), torch.tensor(
            X_test_list).float()
        """---------------------------"""
        lr = 5e-4
        model = IntentFlowNet(input_size_tcn, input_size, local_intent_size, output_size, concat_dim, input_length,
                              num_channels, kernel_size, d_model, dropout).to(device)
        awl = AutomaticWeightedLoss(2).cuda()
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters()}], lr=lr, weight_decay=0)  # L2正则化
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3,
        #                                                        verbose=True, threshold=0.0001, threshold_mode='rel',
        #                                                        cooldown=0, min_lr=0, eps=1e-08)

        best_vloss = 1e8
        lr_lower_bound = 1e-10
        vloss_list = []
        model_name = f"save_models/DMA_step1/no_decoder_{sample}min_K{fold}.pt"  # save_models/DMA_step1/
        early_stopping = EarlyStopping(patience=10, verbose=True)

        model = torch.load(open(model_name, "rb"))
        tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(X_test, plot=False, plot_distribution=False)

        # plot是绘制演化过程，plot_distribution是绘制分布图,二者独立
        print('-' * 89)
        print("K={}: ADE: {:.5f}nmi, FDE: {:.5f}nmi, COG_RMSE: {:.5f}°, SOG_RMSE: {:.5f}kn".format(fold, ADE, FDE,
                                                                                                   rmse_cog,
                                                                                                   rmse_sog))
        print('-' * 89)
        evaluation_score = np.array([ADE, FDE, rmse_cog, rmse_sog])
        evaluation_scores.append(evaluation_score)

        fold += 1

    end_time = time.time()
    print('-' * 89)
    print("Training time: {:.3f} s".format((end_time - start_time)))
    mean_evaluation_score = np.mean(evaluation_scores, axis=0)
    print("Mean evaluation score across all folds:", mean_evaluation_score)

    pred_3d_list = []
    target_3d_list = []
    for pred, target in zip(pred_list, Y_list):
        pred_3d_list.append(np.concatenate(pred.numpy()))
        target_3d_list.append(np.concatenate(target.numpy()))

    value_output = np.concatenate(pred_3d_list)
    value_target = np.concatenate(target_3d_list)

    dist = haversine_distance_2arrays(value_output[:, 2], value_output[:, 1], value_target[:, 2],
                                      value_target[:, 1])
    sorted_errors = np.sort(dist)
    cumulative_distribution = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    np.save(f"iTentformer_DMA_{sample}min_CDFerror_all.npy", sorted_errors)

    # 绘制累积误差分布图
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, cumulative_distribution, marker='o', linestyle='--')
    plt.xlabel('误差值')
    plt.ylabel('累积比例')
    plt.title('累积误差分布图')
    plt.grid(True)
    plt.show()
