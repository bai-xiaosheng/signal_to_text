import matplotlib.pyplot as plt


def save_accuracy_curve(accuracy_list, file_path, x_label='epoch'):
    """
    绘制并保存准确率曲线的函数，当没有具体的周期数据时使用。

    参数:
        accuracy_list (list): 每个周期对应的准确率列表。
        file_path (str): 保存准确率曲线图像的文件路径。
        x_label (str): x轴标签名称，默认为'epoch'。
    """
    # 使用accuracy_list的索引作为训练周期
    epoch_list = list(range(len(accuracy_list)))

    # 设置字体大小
    plt.rcParams.update({'font.size': 18})

    # 创建绘图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(epoch_list, accuracy_list, label='Accuracy over time')  # 绘制曲线
    # plt.title('Accuracy Curve')  # 设置图像标题
    # plt.xlabel(x_label)  # 设置x轴标签
    # plt.ylabel('Accuracy')  # 设置y轴标签
    # plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    # 保存图像
    plt.savefig(file_path)
    plt.close()  # 关闭图形，以释放内存

# 使用示例：
# accuracy_list = [0.6, 0.7, 0.75, 0.8, 0.82]
# save_accuracy_curve(accuracy_list, 'accuracy_curve.png')
