import matplotlib.pyplot as plt

def save_loss(losses, path, description):
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title(description + ' vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(path + description + '.jpg')  # 将图形保存为PNG文件
    plt.close()  # 关闭图形，以释放内存