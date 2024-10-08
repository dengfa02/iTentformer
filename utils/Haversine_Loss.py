import torch
import torch.nn as nn
import numpy as np
import math
import sys

sys.path.append("../")


def haversine_distance_2vectors(latA, lonA, latB, lonB):
    r = 6371  # 地球半径,km
    latA, lonA, latB, lonB = latA * (torch.pi / 180.0), lonA * (torch.pi / 180.0), latB * (torch.pi / 180.0), lonB * (
        torch.pi / 180.0)
    dlat = latB - latA
    dlon = lonB - lonA
    distance_km = 2 * r * torch.arcsin(torch.sqrt(
        torch.sin(dlat / 2) ** 2 + torch.cos(latA) * torch.cos(latB) * torch.sin(dlon / 2) ** 2))
    distance_nm = distance_km / 1.852  # 转换成海里
    # distance_nm = torch.from_numpy(distance_nm).float()
    return distance_nm


class HaversineLoss(nn.Module):  # network
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Assuming y_pred and y_true are tensors of shape (batch_size, 10, 2)
        batch_size, sequence_length, _ = y_pred.size()
        y_pred = y_pred.reshape(-1, 2)
        y_true = y_true.reshape(-1, 2)

        lat_pred, lon_pred = y_pred[:, 1], y_pred[:, 0]
        lat_true, lon_true = y_true[:, 1], y_true[:, 0]

        distances = haversine_distance_2vectors(lat_pred, lon_pred, lat_true, lon_true)

        return distances


def haversine_distance_2arrays(latA, lonA, latB, lonB):
    r = 6371  # 地球半径,km
    latA, lonA, latB, lonB = latA * (np.pi / 180.0), lonA * (np.pi / 180.0), latB * (np.pi / 180.0), lonB * (
        np.pi / 180.0)
    dlat = latB - latA
    dlon = lonB - lonA
    distance_km = 2 * r * np.arcsin(np.sqrt(
        np.sin(dlat / 2 + 1e-10) ** 2 + np.cos(latA) * np.cos(latB) * np.sin(dlon / 2 + 1e-10) ** 2))
    distance_nm = distance_km / 1.852  # 转换成海里
    # distance_nm = torch.from_numpy(distance_nm).float()
    return distance_nm


class Haversinedist(nn.Module):  # svr
    def __init__(self):
        super(Haversinedist, self).__init__()

    def forward(self, y_pred, y_true):
        lat_pred, lon_pred = y_pred[:, 1], y_pred[:, 0]
        lat_true, lon_true = y_true[:, 1], y_true[:, 0]

        distances = haversine_distance_2arrays(lat_pred, lon_pred, lat_true, lon_true)

        return distances


def haversine_distance_2points(latA, lonA, latB, lonB):
    r = 6371  # 地球半径,km
    latA, lonA, latB, lonB = np.radians(latA), np.radians(lonA), np.radians(latB), np.radians(lonB)
    dlat = latB - latA
    dlon = lonB - lonA
    distance_km = 2 * r * math.asin(
        math.sqrt(math.sin(dlat / 2) ** 2 + math.cos(latA) * math.cos(latB) * math.sin(dlon / 2) ** 2))
    distance_nm = distance_km / 1.852  # 转换成海里
    return distance_nm


if __name__ == '__main__':
    a = haversine_distance_2vectors(31.1, 115.4, 32.4, 113.8)
    print(a)
    # rmse = torch.sqrt(torch.mean(distances ** 2))
