import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, multiply, Activation, Conv2D, MaxPooling2D, Attention, Add
# 设置随机种子
np.random.seed(42)
# 定义通道注意力模块
def channel_attention(input_feature, ratio=16):
    channel_axis = -1
    filters = input_feature.shape[channel_axis]

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, filters))(avg_pool)
    avg_pool = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(avg_pool)
    avg_pool = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(avg_pool)

    max_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = Reshape((1, 1, filters))(max_pool)
    max_pool = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(max_pool)
    max_pool = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    return multiply([input_feature, channel_attention])

# 定义空间注意力模块
def spatial_attention(input_feature):
    kernel_size = 7
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_attention = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                               kernel_initializer='he_normal', use_bias=False)(concat)
    return multiply([input_feature, spatial_attention])

# 定义CBAM注意力机制模块
def cbam_block(input_feature, ratio=16):
    input_feature = channel_attention(input_feature, ratio)
    input_feature = spatial_attention(input_feature)
    return input_feature

# 读取数据并进行预处理
headers_dataset = pd.read_csv('./Ransomware_headers.csv')
for col in headers_dataset.columns[4:]:
    headers_dataset[col] = headers_dataset[col] / 255

# 生成图像数据（这里假设之前的处理逻辑已将数据转换为图像并保存）
n_lines, n_columns = 32, 32
image_dir = "./Ransomware_Headers/ImagesGR"

# 划分训练集（90%）和测试集（10%）
csv_image_pointer = headers_dataset.iloc[:, :4].copy()
csv_image_pointer['img'] = ''
for i in range(csv_image_pointer.shape[0]):
    if csv_image_pointer.iloc[i, 2] == 0:
        csv_image_pointer.iloc[i, 4] = f'gw{csv_image_pointer.iloc[i, 0]}.png'
    else:
        csv_image_pointer.iloc[i, 4] = f'rw{csv_image_pointer.iloc[i, 0]}.png'

csv_image_pointer = shuffle(csv_image_pointer)
training_data = csv_image_pointer.iloc[:-int(0.1 * len(csv_image_pointer))].copy()
test_data = csv_image_pointer.iloc[-int(0.1 * len(csv_image_pointer)):].copy()

training_data['GR'] = training_data['GR'].astype(str)
test_data['GR'] = test_data['GR'].astype(str)

# 数据生成器
idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
ffdf_trainng_data = idg.flow_from_dataframe(training_data, directory=image_dir,
                                            x_col="img", y_col="GR",
                                            class_mode="binary", shuffle=False, target_size=(256, 256), batch_size=16)
ffdf_test_data = idg.flow_from_dataframe(test_data, directory=image_dir,
                                         x_col="img", y_col="GR",
                                         class_mode="binary", shuffle=False, target_size=(256, 256), batch_size=16)

# 构建模型
base_model = Xception(weights=None, input_shape=(256, 256, 3), include_top=False)
x = base_model.output
x = cbam_block(x)
x = Attention()([x, x])
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# 编译模型，使用Adamax优化器
optimizer = Adamax(learning_rate=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', 'Precision', 'Recall'])

# 早停法回调
es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=10, min_delta=0.001)
start_time = time.time()
# 训练模型
history = model.fit(ffdf_trainng_data, epochs=35, callbacks=[es], verbose=1)
end_time = time.time()
training_time = end_time - start_time
avg_epoch_time = training_time / len(history.history['accuracy'])

# 保存数据到 CSV 文件
train_data = {
    'epoch': range(1, len(history.history['accuracy']) + 1),
    'accuracy': history.history['accuracy'],
    'loss': history.history['loss'],
    'avg_epoch_time': [avg_epoch_time] * len(history.history['accuracy'])
}
train_df = pd.DataFrame(train_data)


# 评估模型
test_predicted_array = np.round(model.predict(ffdf_test_data))



true_labels = ffdf_test_data.classes

# 计算混淆矩阵
cm = confusion_matrix(true_labels, test_predicted_array)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Xception")
plt.savefig("confusion_matrix_with_attention_xception.png")
plt.close()

# 计算其他评估指标
accuracy = accuracy_score(true_labels, test_predicted_array)
precision = precision_score(true_labels, test_predicted_array)
recall = recall_score(true_labels, test_predicted_array)
f_measure = f1_score(true_labels, test_predicted_array)

# 保存测试指标数据
test_data = {
    'epoch': [len(history.history['accuracy']) + 1],  # 假设测试结果epoch为训练epoch数+1
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'f_measure': [f_measure]
}
test_df = pd.DataFrame(test_data)

combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
combined_df.to_csv('xception_with_attention.csv', index=False)
