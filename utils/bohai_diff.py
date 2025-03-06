import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
import geopandas as gpd


def window_slice(data, win_size, step):
    """
    返回滑窗切分后的三维数据，包含单个船舶所有片段
    """
    num_rows, num_cols = data.shape
    indices = np.arange(0, num_rows - win_size + 1, step)
    # 切分数据
    result = [data[i:i + win_size] for i in indices]

    if len(result) >= 1:
        return np.array(result)


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.max_rows", 10000000)
    pd.set_option("display.width", 100000)
    pd.set_option("display.max_colwidth", 100000)
    plt.rcParams['savefig.dpi'] = 1200  # 图片像素
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    from d3_cog_divide_delta import cog_divide

    data = pd.read_pickle('./data/bohai/bohai_label.pkl')  # 1026条
    # 'MMSI','Length','Width','Draught','Heading','Course','Lon_d','Lat_d','SOG','vx','vy','UnixTime'
    random.seed(42)

    """
    差分
    """
    data_labels_window_list = []
    for w, item in enumerate(data):
        data_labels = cog_divide(item[:, 5], item, ranges=8, concat=True)
        data_labels_window_list.append(data_labels)
    # 'MMSI','Length','Width','Draught','Heading','Course','Lon_d','Lat_d','SOG','vx','vy','UnixTime',
    # 'Global intent'(4dim), 'Local intention(8dim)'

    save_data_list = []
    # 计算变化量，单独计算航向角变化量
    for w, item in enumerate(data_labels_window_list):
        delta_cog_list = []
        diff_result = np.diff(item[:, 6:11], axis=0)  # 沿着行进行差分，默认为-1
        diff_result = np.concatenate((np.zeros((1, diff_result.shape[-1])), diff_result), axis=0)

        cog_data = item[:, 5]
        for k in range(len(cog_data) - 1):
            delta_cog = cog_data[k + 1] - cog_data[k]
            if -180 <= delta_cog <= 180:
                delta_cog_list.append(delta_cog)
            elif delta_cog > 180:
                delta_cog_list.append(delta_cog - 360)
            elif delta_cog < -180:
                delta_cog_list.append(delta_cog + 360)
        delta_cog_list = np.concatenate(([0.0], np.array(delta_cog_list)), axis=0).reshape(-1, 1)
        diff_data = np.concatenate((item[:, [0, 1, 5, 6, 7, 8, 9, 10]], delta_cog_list, diff_result,
                                    item[:, 11:]), axis=-1)
        save_data_list.append(diff_data)

    """
    滑窗切分放在交叉验证，保证每条测试都是完整轨迹
    """
    num_slice = 0.0
    num_point = 0.0
    net_list = []
    format_list = []
    dtype_list = ['int32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32',
                  'float32', 'float32', 'float32', 'float32',
                  'int32', 'int32', 'int32', 'int32', 'int32', 'int32', 'int32',
                  'int32', 'int32', 'int32', 'int32']
    for k, trj in enumerate(save_data_list):
        if len(trj) >= 20 and k != 3:  # id为3的轨迹异常
            trj = trj.astype('float64')  # 转换格式
            format_list.append(trj)
    # for k, trj in enumerate(format_list):
    #     if len(trj) >= 20:
    #         slice = window_slice(trj, win_size=20, step=1)
    #         num_slice += slice.shape[0]
    #         num_point += slice.shape[0] + 20 - 1
    #         net_list.append(slice)
    #         if slice.shape[-1] != 27:
    #             print("错误", k, slice.shape[-1])

    check = np.concatenate(format_list, axis=0)
    print("轨迹段数", num_slice)
    print("重采后轨迹点数", num_point)
    pd.to_pickle(format_list, 'data/bohai/net_data_bohai.pkl')
