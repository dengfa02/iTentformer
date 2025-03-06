import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def cog_divide(angle_data, origin_data, ranges=8, concat=True):
    angle_ranges = [((i * 360.0 / ranges + 360.0 - 180.0 / ranges), ((i + 1) * 360.0 / ranges - 180.0 / ranges))
                    for i in range(ranges)]
    angle_ranges = [(x - 360 if x > 360 else x, y - 360 if y > 360 else y) for x, y in angle_ranges]
    labels = np.eye(ranges)
    data_labels_list = []
    for i, angle in enumerate(angle_data):
        for j, angle_range in enumerate(angle_ranges):
            if j == 0:
                if (angle_range[0] - 1e-3 <= angle < 360 + 1e-3) or (0 <= angle < angle_range[1] + 1e-3):
                    # 注意第一个337.5-22.5的比较要分段讨论
                    if concat:
                        data_with_labels = np.concatenate((origin_data[i], labels[j, :]))
                        data_labels_list.append(data_with_labels)
                        break
                    else:
                        data_labels_list.append(labels[j, :])
                        break
            else:
                if angle_range[0] - 1e-3 <= angle < angle_range[1] + 1e-3:
                    if concat:
                        data_with_labels = np.concatenate((origin_data[i], labels[j, :]))
                        data_labels_list.append(data_with_labels)
                        break
                    else:
                        data_labels_list.append(labels[j, :])
                        break

    data_labels = np.array(data_labels_list)
    return data_labels
