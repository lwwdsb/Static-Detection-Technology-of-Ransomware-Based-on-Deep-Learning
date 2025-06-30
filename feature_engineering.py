import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
# 设置为交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
headers_dataset = pd.read_csv('./Ransomware_headers.csv')
#print(headers_dataset)

# Divide all elements by the maximum possible
# 第4列之后（图像像素值）的所有特征进行归一化，数据预处理
for col in headers_dataset.columns[4:]:
    headers_dataset[col] = headers_dataset[col] / 255

#print(headers_dataset)

n_lines, n_columns = 32, 32

X = np.split(headers_dataset.columns[4:], n_lines)

for i in range(len(X)):
    X[i] = sorted(X[i], key=int)
    #print(X[i])

os.makedirs("Ransomware_Headers/ImagesGR", exist_ok=True)
os.chdir("Ransomware_Headers/ImagesGR")

# Image sample generation
ct = 0
for position_sample in range(headers_dataset.shape[0]):
    sample = headers_dataset.iloc[position_sample, 4:]

    sample_aux = np.zeros((n_lines, n_columns), dtype=float)

    for i in range(len(X)):
        sample_aux[i] = sample[X[i]]

    plt.figure(figsize=(256, 256), dpi=1)
    # Spectral_r cmap
    g = sns.heatmap(sample_aux, cmap='Spectral_r', vmin=0, vmax=1, xticklabels=False, yticklabels=False, cbar=False)

    # ransomware = rw and goodware = gw.
    # 1133 goodware samples, the remaining are ransomware.
    if position_sample < 1134:
        sample_path = f"./gw{headers_dataset.iloc[position_sample, 0]}.png"
    else:
        sample_path = f"./rw{headers_dataset.iloc[position_sample, 0]}.png"

    print(sample_path)
    plt.tight_layout(pad=0)
    plt.savefig(sample_path)
    plt.close()
    ct = ct + 1

# Confirm how many files are in the current directory
# It should be 2157 (number of samples)
current_directory = os.getcwd()
files = os.listdir(current_directory)
file_count = 0
for file in files:
    if os.path.isfile(os.path.join(current_directory, file)):
        file_count += 1

print(f"There are {file_count} files in the current directory.")

# Back to Ransomware_Headers directory
os.chdir("../")

# 创建映射表，记录每个样本对应的图像文件名和标签
csv_image_pointer = headers_dataset.iloc[:, :4].copy()
csv_image_pointer['img'] = ''

for i in range(csv_image_pointer.shape[0]):
    if csv_image_pointer.iloc[i, 2] == 0:
        csv_image_pointer.iloc[i, 4] = f'gw{csv_image_pointer.iloc[i, 0]}.png'
    else:
        csv_image_pointer.iloc[i, 4] = f'rw{csv_image_pointer.iloc[i, 0]}.png'

#print(csv_image_pointer)