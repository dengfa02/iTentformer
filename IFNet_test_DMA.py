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
from traj_prediction.Haversine_Loss import HaversineLoss
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
    pred_list = []
    Y_list = []
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

            if plot:
                # map = pd.read_pickle('../../data/bohai/net_data_bohai.pkl')
                src = src[:, :, :4] @ torch.from_numpy(transform_matrix).float().cuda() + torch.from_numpy(
                    mean_values[:4]).float().cuda()
                sum_attn_weights = attn_weights.sum(dim=1)  # 4个头加和  16*40*40
                for b in range(len(value_output)):
                    matrix = sum_attn_weights[b].cpu().numpy()  # 示例中使用的矩阵
                    # 计算最大值和最小值
                    max_val = matrix.max()
                    min_val = matrix.min()
                    # 使用最大值和最小值进行放缩到 [-1, 1]
                    normalized_matrix = 2 * (matrix - min_val) / (max_val - min_val) - 1
                    sns.heatmap(normalized_matrix, annot=False, cmap='viridis')
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.xlabel('Key', fontsize=14)
                    plt.ylabel('Query', fontsize=14)
                    plt.tight_layout()
                    # plt.title('{}'.format(count))
                    plt.show()

                    gdf = gpd.read_file('../../zi_paper_plot/gadm41_DNK_shp/gadm41_DNK_1.shp',
                                        geometry_precision=0.00001)  # ../../ship/dataset/map/gadm41_CHN_shp/gadm41_CHN_0.shp
                    gdf2 = gpd.read_file('../../zi_paper_plot/gadm41_SWE_shp/gadm41_swe_1.shp',
                                         geometry_precision=0.00001)
                    fig, ax = plt.subplots()
                    ax.set_xlim([10.7, 13])  # 12.75
                    ax.set_ylim([56, 56.9])
                    gdf.plot(ax=ax, facecolor='none', linewidth=0.5)
                    gdf2.plot(ax=ax, facecolor='none', linewidth=0.5)
                    for k in range(target_length):
                        dx = value_output[b, k, 3].cpu().numpy() * np.sin(
                            np.radians(value_output[b, k, 0].cpu().numpy())) / 150  # 计算向量在 x 轴上的分量
                        dy = value_output[b, k, 3].cpu().numpy() * np.cos(
                            np.radians(value_output[b, k, 0].cpu().numpy())) / 150  # 计算向量在 y 轴上的分量
                        plt.arrow(value_output[b, k, 1].cpu().numpy(), value_output[b, k, 2].cpu().numpy(), dx, dy,
                                  head_width=0.03, head_length=0.03, fc='#c58a93', ec='#c58a93')

                        dx_y = value_target[b, k, 3].cpu().numpy() * np.sin(
                            np.radians(value_target[b, k, 0].cpu().numpy())) / 150  # 计算向量在 x 轴上的分量
                        dy_y = value_target[b, k, 3].cpu().numpy() * np.cos(
                            np.radians(value_target[b, k, 0].cpu().numpy())) / 150  # 计算向量在 y 轴上的分量
                        plt.arrow(value_target[b, k, 1].cpu().numpy(), value_target[b, k, 2].cpu().numpy(), dx_y, dy_y,
                                  head_width=0.03, head_length=0.03, fc='#708ab9', ec='#708ab9')

                    plt.plot([], [], color='#c58a93', label='Predict vector', linewidth=1.5, linestyle='-', marker='>',
                             markersize=7)
                    plt.plot([], [], color='#708ab9', label='True vector', linewidth=1.5, linestyle='-', marker='>',
                             markersize=7)

                    plt.plot(src[b, :, 1].cpu().numpy(), src[b, :, 2].cpu().numpy(), '-.', color='grey', marker='o',
                             label='Observed', markersize=3, linewidth=1.5)
                    plt.plot(value_output[b, :, 1].cpu().numpy(), value_output[b, :, 2].cpu().numpy(), 'g-', marker='D',
                             label='Predict Trajectory', markersize=5)
                    plt.plot(value_target[b, :, 1].cpu().numpy(), value_target[b, :, 2].cpu().numpy(), '--',
                             color='k', marker='o', label='Ground Truth', markersize=5)

                    plt.legend(frameon=False, fontsize=13, loc='upper left')
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.xlabel('Longitude / (°)', fontsize=14)
                    plt.ylabel('Latitude / (°)', fontsize=14)
                    plt.tight_layout()
                    # plt.title('{}'.format(count))
                    plt.show()
                    count += 1

                # for k, item in enumerate(tgt_y):
                #     # for i, traj in enumerate(map):
                #     #     plt.plot(traj[:, 3], traj[:, 4], '-', color='#08316E', linewidth=0.2)
                #
                #     # plt.subplot(1, 3, 2)
                #     # plt.plot(src[k, :, 1].cpu().numpy(), src[k, :, 2].cpu().numpy(), 'g-', label='Observed', marker='o',
                #     #          alpha=1, markersize=5.5)
                #     # plt.plot(value_output[k, :, 1].cpu().numpy(), value_output[k, :, 2].cpu().numpy(), 'r-',
                #     #          label='Predicted', marker='s', alpha=0.65, markersize=5)
                #     # plt.plot(value_target[k, :, 1].cpu().numpy(), value_target[k, :, 2].cpu().numpy(), 'g--',
                #     #          label='Ground Truth', marker='^', alpha=1, markersize=5.5)
                #     #
                #     # plt.legend(['History(src)', 'Ground Truth(tgt_y)', 'pred(tgt)'])
                #     # plt.legend(frameon=False, fontsize=14)
                #     # plt.xticks(fontsize=14)
                #     # plt.yticks(fontsize=14)
                #     # plt.xlabel('Longitude / (°)', fontsize=14)
                #     # plt.ylabel('Latitude / (°)', fontsize=14)
                #     # plt.show()
                #     # if k <= 15:
                #     #     save_path = f"U:/exp_fig/itentformer/itentformer_{sample}_{k}.png"
                #     #     plt.savefig(save_path)
                #     #     plt.clf()
                #
                #     plt.plot(src[k, :, 0].cpu().numpy().tolist() + value_output[k, :, 0].cpu().numpy().tolist(), 'r',
                #              marker='o',
                #              label='Predicted')
                #     plt.plot(src[k, :, 0].cpu().numpy().tolist() + value_target[k, :, 0].cpu().numpy().tolist(), 'g',
                #              marker='o',
                #              label='Ground Truth')
                #     plt.plot(src[k, :, 0].cpu().numpy(), 'b', marker='o', label='Observed')
                #     plt.show()
                #     # plt.subplot(1, 3, 1)
                #     # plt.title('COG')
                #     # # plt.plot(src[k, :, 0].cpu().numpy(), 'b', marker='o')
                #     # plt.plot(tgt_y[k, :, 0].cpu().numpy(), 'k', marker='o')
                #     # plt.plot(tgt[k, :, 0].cpu().numpy(), 'r', marker='o')
                #     # plt.legend(['Ground Truth(tgt_y)', 'pred(tgt)'])
                #     # plt.subplot(1, 3, 3)
                #     # plt.title('SOG')
                #     # # plt.plot(src[k, :, 3].cpu().numpy(), 'b', marker='o')
                #     # plt.plot(tgt_y[k, :, 3].cpu().numpy(), 'k', marker='o')
                #     # plt.plot(tgt[k, :, 3].cpu().numpy(), 'r', marker='o')
                #     # plt.legend(['Ground Truth(tgt_y)', 'pred(tgt)'])

            total_loss += loss.item()
            sample += 1

        if plot_distribution:
            gdf = gpd.read_file('../../zi_paper_plot/gadm41_DNK_shp/gadm41_DNK_1.shp',
                                geometry_precision=0.00001)  # ../../ship/dataset/map/gadm41_CHN_shp/gadm41_CHN_0.shp
            gdf2 = gpd.read_file('../../zi_paper_plot/gadm41_SWE_shp/gadm41_swe_1.shp',
                                 geometry_precision=0.00001)
            fig, ax = plt.subplots()
            ax.set_xlim([10.7, 13])  # 12.75
            ax.set_ylim([56, 57.5])
            gdf.plot(ax=ax, facecolor='none', linewidth=0.5)
            gdf2.plot(ax=ax, facecolor='none', linewidth=0.5)

            pred_list = np.concatenate(pred_list, axis=0)
            for k, item in enumerate(pred_list):
                plt.plot(item[:, 1], item[:, 2], '-', color='#F5CF36', alpha=0.5, linewidth=0.5)

            turning_area = [(11.4, 56.85), (12.1, 56.85), (12.1, 56.5), (11.4, 56.5)]
            a2 = []
            lons2 = [point[0] for point in turning_area]
            lats2 = [point[1] for point in turning_area]
            a2.append(np.vstack((lats2, lons2)).T)
            plt.plot(lons2 + [lons2[0]], lats2 + [lats2[0]], 'k--', linewidth=1)  # 添加连线，并使首尾坐标相连
            plt.text(11.25, 56.9, "Turning area", ha='center', fontsize=16, color='k')

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Longitude / (°)', fontsize=14)
            plt.ylabel('Latitude / (°)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

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

        intent, output, _ = model(delta, src)

        # 序列one-hot不能直接用交叉熵，模型最后一层要激活
        value_output = output
        value_target = tgt_y

        # 计算数值部分的 MSE 损失
        mse_loss = criterion1(value_output, value_target)
        # 计算 one-hot 部分的交叉熵损失
        # ce_loss = criterion2(one_hot_output[:, -1, :], one_hot_target[:, -1, :])

        # intent_list = torch.concat(intent_list)
        intent = intent.reshape(-1, intent.size(-1))

        intent_y = intent_y.reshape(-1, intent_y.size(-1))
        loss_int = criterion1(intent, intent_y)  # 变为预测未来变化率

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

        total_loss += loss.item()
        sample += 1  # 测试的所有样本数量

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()

        if idx > 0 and idx % (100 * batch_size) == 0:
            cur_loss = total_loss / sample
            print("Epoch {:4d} | lr {:.9f} | loss {:.5f}".format(ep, lr, cur_loss))
            # ADE = torch.sum(torch.tensor(ADE_list)) / sample
            # print(" ADE: {:.5f}nmi-->{:.5f}m".format(ADE, ADE * 1852))
            # FDE = torch.sum(torch.tensor(FDE_list)) / sample
            # print(" FDE: {:.5f}nmi-->{:.5f}m".format(FDE, FDE * 1852))
            # rmse_cog = torch.sum(torch.tensor(rmse_cog_list)) / sample
            # print(" RMSE_COG: {:.5f}°".format(rmse_cog))
            # rmse_sog = torch.sum(torch.tensor(rmse_sog_list)) / sample
            # print(" RMSE_SOG: {:.5f}kn".format(rmse_sog))
            # ADE_list = []
            # FDE_list = []
            # rmse_cog_list = []
            # rmse_sog_list = []
            total_loss = 0.0
            sample = 0


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
    # pred_list = []
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

        # 训练
        # for ep in range(1, 301):
        #     train(ep, parallel_train=False)
        #     vloss, _, _, _, _ = evaluate(X_valid, name='Validation')
        #     tloss, _, _, _, _ = evaluate(X_test, name='Test')
        #     # 设置 EarlyStopping
        #     early_stopping(vloss, model)
        #
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        #
        #     if vloss < best_vloss:
        #         with open(model_name, "wb") as f:
        #             torch.save(model, f)
        #             print("Saved model!\n")
        #         best_vloss = vloss
        #
        #     if ep > 5 and vloss > max(vloss_list[-3:]) and lr > lr_lower_bound:
        #         lr /= 5.0
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr
        #     # scheduler.step(vloss)
        #
        #     vloss_list.append(vloss)

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
    # plot_loa = False
    # if plot_loa:
    #     pred_list = np.concatenate(pred_list, axis=0)
    #     Y_list = np.concatenate(Y_list, axis=0)
    #     pd.to_pickle(pred_list[:, :, 1:3], '../../data/bohai/error_data_plot/iTentformer_pred.pkl')
    #     pd.to_pickle(Y_list[:, :, 1:3], '../../data/bohai/error_data_plot/iTentformer_Y.pkl')
    #
    #     error_lon = abs(pred_list[:, :, 1] - Y_list[:, :, 1])
    #     error_lat = abs(pred_list[:, :, 2] - Y_list[:, :, 2])
    #
    #     mean_error_lon = np.mean(error_lon, axis=0)
    #     mean_error_lat = np.mean(error_lat, axis=0)
    #     std_error_lon = np.std(error_lon, axis=0)
    #     std_error_lat = np.std(error_lat, axis=0)
    #
    #     error_lon_min = np.min(error_lon, axis=0)
    #     error_lon_max = np.max(error_lon, axis=0)
    #     error_lat_min = np.min(error_lat, axis=0)
    #     error_lat_max = np.max(error_lat, axis=0)
    #
    #     q1_lon = np.percentile(error_lon, 25, axis=0)
    #     q3_lon = np.percentile(error_lon, 75, axis=0)
    #     q1_lat = np.percentile(error_lat, 25, axis=0)
    #     q3_lat = np.percentile(error_lat, 75, axis=0)
    #
    #     lower_bound_lon = mean_error_lon - 1.96 * std_error_lon
    #     upper_bound_lon = mean_error_lon + 1.96 * std_error_lon
    #     lower_bound_lat = mean_error_lat - 1.96 * std_error_lat
    #     upper_bound_lat = mean_error_lat + 1.96 * std_error_lat
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     # 绘制误差平均值和填充区域
    #     ax.fill_between(np.arange(len(mean_error_lon)), lower_bound_lon, upper_bound_lon, color='#387690', alpha=0.2
    #                     , label='95% confidence')
    #
    #     ax.fill_between(np.arange(len(mean_error_lat)) + 0.4, lower_bound_lat, upper_bound_lat, color='#F09B27',
    #                     alpha=0.2
    #                     , label='95% confidence')
    #
    #     # 绘制经度误差柱状图和误差线
    #     ax.bar(np.arange(len(mean_error_lon)), mean_error_lon, color='#387690', alpha=0.7, label='Longitude', width=0.4)
    #     # 绘制纬度误差柱状图和误差线
    #     ax.bar(np.arange(len(mean_error_lat)) + 0.4, mean_error_lat, color='#F09B27', alpha=0.7, label='Latitude',
    #            width=0.4)
    #     ax.errorbar(np.arange(len(mean_error_lon)), mean_error_lon,
    #                 yerr=[mean_error_lon - q1_lon, q3_lon - mean_error_lon], fmt='', ecolor='#387690',
    #                 capsize=5, capthick=1, markersize=5, linestyle='None')
    #     ax.errorbar(np.arange(len(mean_error_lat)) + 0.4, mean_error_lat,
    #                 yerr=[mean_error_lat - q1_lat, q3_lat - mean_error_lat], fmt='', ecolor='#F09B27',
    #                 capsize=5, capthick=1, markersize=5, linestyle='None')
    #
    #     ax.axhline(y=0, color='k', linestyle='-')
    #     ax.legend(frameon=False, loc='upper left', fontsize=12)
    #     ax.set_xlabel('Point Sequence', fontsize=14)
    #     ax.set_ylabel('Absolute Error / (°)', fontsize=14)
    #     ax.set_xticks(np.arange(len(mean_error_lon)) + 0.2, fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.xticks(fontsize=12)
    #     ax.set_xticklabels(np.arange(1, len(mean_error_lon) + 1))
    #
    #     plt.show()
