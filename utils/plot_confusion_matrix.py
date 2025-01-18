#函数形式
import string

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from scipy.io import savemat, loadmat
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(output, target, save_path=None, acc=None):
    """
    Plot and optionally save the confusion matrix.

    Parameters:
        output (array_like): Predicted labels.
        target (array_like): True labels.
        save_path (str, optional): File path to save the confusion matrix plot. Default is None.

    Returns:
        None
    """
    # 计算混淆矩阵
    cm = confusion_matrix(target, output, normalize='true')
    # 转换为numpy数组
    cm_np = np.array(cm)
    # 保存为mat文件
    savemat(save_path+acc+".mat", {"confusion_matrix": cm_np})
    # 绘制混淆矩阵
    plt.figure(figsize=(30, 24), dpi=60)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=range(50), yticklabels=range(49), cbar=False,
                annot_kws={"size": 8}, linewidths=.5)

    #旋转标签
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.tight_layout()  # 保证图不重叠

    plt.xlabel('Predicted labels', fontsize=16, fontname='Times New Roman')
    plt.ylabel('True labels', fontsize=16, fontname='Times New Roman')
    plt.title('Confusion Matrix', fontsize=20, fontname='Times New Roman')

    # 如果提供了保存路径，则保存混淆矩阵图
    if save_path:
        plt.savefig(save_path+acc+".png")
    plt.close()  # 关闭图形，以释放内存
    # 显示混淆矩阵图
    # plt.show()

# # 示例用法
# output = np.random.randint(0, 49, size=1000)  # 100个随机的预测结果
# target = np.random.randint(0, 49, size=1000)  # 100个随机的实际标签
# plot_confusion_matrix(output, target, save_path='confusion_matrix.png')


def plot_multiple_confusion_matrices(mat_file_paths,titles, num_classes=50, output_path = '../exp/visualizations/multiple_confusion_matrices_plot.png'):
    """
    Plot multiple confusion matrices in a grid layout similar to the provided example.

    Parameters:
        mat_file_paths (list of str): List of paths to .mat files containing confusion matrices.
        num_classes (int): Number of classes in the confusion matrices.
        grid_shape (tuple): Shape of the grid (rows, columns) for the subplot layout.
    """

    rows = 2
    cols = (len(mat_file_paths) + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.05)  # 缩小间距

    # Color map for consistency with the provided example image
    # 深蓝色 --> 深红色
    cmap = plt.cm.jet
    # 紫色 --> 黄色
    # cmap = plt.cm.viridis

    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # Plot each confusion matrix
    for i, (mat_file_path, title, ax) in enumerate(zip(mat_file_paths, titles, axs.flat)):
        # Load confusion matrix from .mat file
        mat_data = loadmat(mat_file_path)

        # Assuming the confusion matrix is stored under the key 'confusion_matrix'
        if 'confusion_matrix' not in mat_data:
            print(f"Error: 'confusion_matrix' key not found in {mat_file_path}.")
            ax.axis('off')  # Turn off axis if matrix is not found
            continue

        confusion_matrix = mat_data['confusion_matrix']

        # Plot the confusion matrix
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, norm=norm)
        # ax.set_title(title, fontsize=10)
        ax.set_xlabel('Predicted Classes \n('+string.ascii_uppercase[i]+")"+title, fontsize=14, fontname='Times New Roman')
        if i % cols == 0:
            ax.set_ylabel('True Classes', fontsize=14, fontname='Times New Roman')
        ax.set_xticks(np.arange(0, num_classes, 10))
        ax.set_yticks(np.arange(0, num_classes, 10))
        # 设置坐标轴字体
        ax.tick_params(axis='both', which='major', labelsize=14)  # 增大坐标轴字体大小
        # ax.set_xticklabels(ax.get_xticks(), fontsize=14, fontname='Times New Roman')  # 设置 x 轴刻度标签
        # ax.set_yticklabels(ax.get_yticks(), fontsize=14, fontname='Times New Roman')  # 设置 y 轴刻度标签
    # Turn off empty subplots if there are fewer matrices than grid cells
    for j in range(i + 1, rows * cols):
        axs.flat[j].axis('off')

    # Add a single colorbar on the right side of the grid
    # fraction: 控制颜色条占整个图形高度的比例，增大此值会使颜色条更宽。
    # pad: 控制颜色条与图形的距离，增大此值会使颜色条和图形之间的间距更大。
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.03, pad=0.05)  # 调整这两个参数
    cbar.set_label('Accuracy', fontsize=14)

    # Save the figure to the directory of the first .mat file

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multiple confusion matrices plot saved to: {output_path}")


def plot_class_accuracies(mat_file_paths, titles, class_names, samples, output_path="../exp/visualizations/class_accuracy_comparison.png"):
    """
    读取多个 .mat 文件，计算指定类别的识别准确率并绘制直方图。

    参数：
    - mat_file_paths: list，.mat 文件路径列表，每个文件包含模型预测数据。
    - titles: list，每个 .mat 文件对应的模型名称，用于图例。
    - class_names: list，类别名称列表（总共有50个类别）。
    - samples: list，指定的类别索引，表示要绘制的类别。
    - output_path: str，保存图像的路径（默认为 'class_accuracy_comparison.png'）。
    """
    num_classes = len(samples)  # 选择的类别数量
    num_methods = len(mat_file_paths)
    accuracy_data = {title: [] for title in titles}

    # 定义柔和颜色列表，突出最后一个方法
    base_colors = ['#4c72b0', '#55a868', '#8172b2', '#ccb974', '#64b5cd'][:num_methods - 1]
    highlight_color = "#e15759"  # 为最后一个方法定义对比暖色
    colors = base_colors + [highlight_color]

    # 设置字体
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']

    # 遍历每个方法的 .mat 文件，计算指定类别的准确率
    for idx, mat_path in enumerate(mat_file_paths):
        mat_data = loadmat(mat_path)
        confusion_matrix = mat_data['confusion_matrix'] # 假设真实标签存储在 "labels" 键中

        # 计算每个指定类别的准确率
        for class_idx in samples:
            # 对角线元素表示正确分类的数量，行的总和表示该类别的总数
            class_accuracy = confusion_matrix[class_idx, class_idx]
            accuracy_data[titles[idx]].append(class_accuracy)

    # 绘制图像
    x = np.arange(num_classes)  # 横坐标位置
    width = 0.15  # 每个柱的宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (title, color) in enumerate(zip(titles, colors)):
        ax.bar(x + i * width, accuracy_data[title], width, label=title, color=color)

    # 添加图形细节
    # ax.set_xlabel('Jamming Type/5dB', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Recognition Accuracy', fontsize=12)
    # ax.set_title('Comparison of recognition accuracy for different methods', fontsize=14)
    ax.set_xticks(x + width * (num_methods - 1) / 2)
    ax.set_xticklabels([class_names[i] for i in samples], rotation=0, ha="center", fontsize=14)

    # # 将图例放在图的上方
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(titles))
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=len(titles))
    # 将图例放在右侧
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Methods", fontsize=12, title_fontsize=12)
    # 保存图例单独的图像
    fig_legend = plt.figure(figsize=(8, 1))
    handles, labels = ax.get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc='center', ncol=len(titles), frameon=False)
    fig_legend.savefig("../exp/visualizations/tuli.png", bbox_inches='tight')
    plt.close(fig_legend)  # 关闭图例图形


    # 在柱子顶部标出准确率值

    for i, (title, color) in enumerate(zip(titles, colors)):
        gradient = np.linspace(0.9, 0.5, num_classes) if color != highlight_color else np.full(num_classes, 0.8)
        for j, acc in enumerate(accuracy_data[title]):
            bar = ax.bar(x[j] + i * width, acc, width, label=title if j == 0 else "", color=color, alpha=gradient[j])

            # 错开数值位置：奇数向上移动，偶数向下移动
            # offset = 0.02 if i % 2 == 0 else -0.02
            offset = 0
            ax.text(
                x[j] + i * width, acc + offset, f'{acc:.2f}',
                ha='center', va='bottom', fontsize=8,  rotation=0
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存图像
    plt.close(fig)  # 关闭图以释放内存

    print(f"图像已保存到 {output_path}")


if __name__ == "__main__":
    # Example usage:
    # You can replace this list with your own .mat file paths
    # 5dB情况
#     mat_file_paths = ["../exp/JNR_5dB_10M_29D_20H_0.9679/result/834_0.9678666666666667.mat",
#                       "../exp/JNR_5dB_07M_30D_09H_0.9411/result/752_0.9410666666666667.mat",
#                       "../exp/JNR_5dB_10M_03D_20H_0.9709/result/745_0.9708666666666667.mat",
#                       "../exp/JNR_5dB_10M_28D_21H_0.9687/result/611_0.9686666666666667.mat",
#                       "../exp/JNR_5dB_10M_25D_10H_0.9745/result/795_0.9744666666666667.mat",
#                       "../exp/JNR_5dB_07M_03D_19H_0.9827/result/434_0.9826666666666667.mat",
# ]
#     titles = ["ResNet18", "ViT", "VGG16", "ResNet50", "JRNet", "Ours"]
#     output_path = '../exp/visualizations/multiple_confusion_matrices_plot_5dB.png'
#     # 绘制多个算法的混淆矩阵对比图
#     plot_multiple_confusion_matrices(mat_file_paths, titles, num_classes=50, output_path = output_path)

    names = [
        'CSJ', 'DFTJ', 'ISFJ', 'ISRJ', 'LFM', 'MFPJ', 'MFPJ+DFTJ', 'MFPJ+ISFJ', 'MFPJ+ISRJ', 'MFPJ+MFTJ',
'MFPJ+SPFJ', 'MFTJ', 'NAMJ', 'NCJ', 'NCJ+DFTJ', 'NCJ+ISFJ', 'NCJ+ISRJ', 'NCJ+MFTJ', 'NCJ+SPFJ', 'NFMJ',
'NFMJ+DFTJ', 'NFMJ+ISFJ', 'NFMJ+ISRJ', 'NFMJ+MFTJ', 'NFMJ+SPFJ', 'NPJ', 'NPJ+DFTJ', 'NPJ+ISFJ', 'NPJ+ISRJ',
'NPJ+MFTJ', 'NPJ+SPFJ', 'SFJ1', 'SFJ1+DFTJ', 'SFJ1+ISFJ', 'SFJ1+ISRJ', 'SFJ1+MFTJ', 'SFJ1+SPFJ', 'SFJ2',
'SFJ2+DFTJ', 'SFJ2+ISFJ', 'SFJ2+ISRJ', 'SFJ2+MFTJ', 'SFJ2+SPFJ', 'SFJ3', 'SFJ3+DFTJ', 'SFJ3+ISFJ',
'SFJ3+ISRJ', 'SFJ3+MFTJ', 'SFJ3+SPFJ', 'SPFJ']
    sample = [10,18,24,30,36,42,48,49]
    print(names[i] for i in sample)
    for i in sample:
        print(names[i])
    # samples = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]
    # samples = [14, 15, 21, 19, 5]
    # output_path = "../exp/visualizations/class_accuracy_comparison_5dB.png"
    # # 绘制指定类别的直方图算法对比
    # plot_class_accuracies(mat_file_paths,titles,names,samples,output_path)

    # 0dB情况
    # mat_file_paths = ["../exp/JNR_5dB_10M_29D_20H_0.9679/result/834_0.9678666666666667.mat",
    #                   "../exp/JNR_5dB_07M_30D_09H_0.9411/result/752_0.9410666666666667.mat",
    #                   "../exp/JNR_5dB_10M_03D_20H_0.9709/result/745_0.9708666666666667.mat",
    #                   "../exp/JNR_5dB_10M_28D_21H_0.9687/result/611_0.9686666666666667.mat",
    #                   "../exp/JNR_5dB_10M_25D_10H_0.9745/result/795_0.9744666666666667.mat",
    #                   "../exp/JNR_5dB_07M_03D_19H_0.9827/result/434_0.9826666666666667.mat",
    #                   ]
    # titles = ["ResNet18", "ViT", "VGG16", "ResNet50", "JRNet", "Ours"]
    # output_path = '../exp/visualizations/multiple_confusion_matrices_plot_0dB.png'
    # # 绘制多个算法的混淆矩阵对比图
    # plot_multiple_confusion_matrices(mat_file_paths, titles, num_classes=50, output_path=output_path)
    #
    # names = [
    #     'csj', 'dftj', 'isfj', 'isrj', 'LFM', 'mfpj', 'mfpj_dftj', 'mfpj_isfj', 'mfpj_isrj', 'mfpj_mftj',
    #     'mfpj_spfj', 'mftj', 'namj', 'ncj', 'ncj_dftj', 'ncj_isfj', 'ncj_isrj', 'ncj_mftj', 'ncj_spfj', 'nfmj',
    #     'nfmj_dftj', 'nfmj_isfj', 'nfmj_isrj', 'nfmj_mftj', 'nfmj_spfj', 'npj', 'npj_dftj', 'npj_isfj', 'npj_isrj',
    #     'npj_mftj', 'npj_spfj', 'sfj1', 'sfj1_dftj', 'sfj1_isfj', 'sfj1_isrj', 'sfj1_mftj', 'sfj1_spfj', 'sfj2',
    #     'sfj2_dftj', 'sfj2_isfj', 'sfj2_isrj', 'sfj2_mftj', 'sfj2_spfj', 'sfj3', 'sfj3_dftj', 'sfj3_isfj',
    #     'sfj3_isrj', 'sfj3_mftj', 'sfj3_spfj', 'spfj']
    # # samples = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]
    # samples = [14, 15, 21, 19, 5]
    # output_path = "../exp/visualizations/class_accuracy_comparison_0dB.png"
    # # 绘制指定类别的直方图算法对比
    # plot_class_accuracies(mat_file_paths, titles, names, samples, output_path)