import pandas as pd
import matplotlib.pyplot as plt

# 读取四个模型的数据
model_files = [
    'xception_with_attention.csv',
    'InceptionResNetV2.csv',
    'EfficientNetV2S.csv',
    'Xception_original.csv'
]
model_names = [
    'Xception with Attention',
    'InceptionResNetV2',
    'EfficientNetV2S',
    'Xception (Original)'
]

# 用于存储每个模型最后一行的数据
last_rows = []
for file in model_files:
    df = pd.read_csv(file)
    last_row = df.iloc[-1]
    last_rows.append(last_row)

# 生成对比表格
comparison_df = pd.DataFrame(last_rows, index=model_names, columns=['accuracy', 'precision', 'recall', 'f_measure'])
print(comparison_df[['accuracy', 'precision', 'recall', 'f_measure']])

# 绘制训练准确率曲线
plt.figure(figsize=(10, 6))
for file, name in zip(model_files, model_names):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['accuracy'], label=name)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.savefig('training_accuracy_comparison.png')
plt.close()

# 绘制训练损失率曲线
plt.figure(figsize=(10, 6))
for file, name in zip(model_files, model_names):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['loss'], label=name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.savefig('training_loss_comparison.png')
plt.close()

# 绘制平均每轮训练时长曲线
plt.figure(figsize=(10, 6))
for file, name in zip(model_files, model_names):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['avg_epoch_time'], label=name)
plt.xlabel('Epochs')
plt.ylabel('Average Epoch Time (s)')
plt.title('Average Epoch Time Comparison')
plt.legend()
plt.savefig('average_epoch_time_comparison.png')
plt.close()
