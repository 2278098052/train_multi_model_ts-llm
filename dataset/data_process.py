import h5py
import numpy as np
import pandas as pd


def read_and_sort_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        seq_data = f['seq_data'][:]  # 读取所有 seq_data
        id_data = f['data_ID'][:]    # 读取所有 data_ID

    # 将 seq_data 展平为二维数组
    seq_data_flat = seq_data.reshape(seq_data.shape[0], -1)

    # 将 id_data 和 seq_data_flat 组合成一个 DataFrame
    df = pd.DataFrame(seq_data_flat, columns=[f'seq_{i}' for i in range(seq_data_flat.shape[1])])
    df['id'] = id_data

    # 根据 id_data 进行排序
    sorted_df = df.sort_values(by='id').reset_index(drop=True)

    return sorted_df

# 示例用法
file_path = 'dataset/dataset_processing/data_merged.h5'
sorted_data = read_and_sort_h5(file_path)
print(sorted_data)