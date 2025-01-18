import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from args import parse_args
from constants.text_constants import VOCAB
from data.rml_loader import load_radioml_dataset
from model.timeFreq_to_text import TimeFreqToText
from utils import plot_acc, plot_loss
from utils.plot_confusion_matrix import plot_confusion_matrix


def move_to_device(datas, device):
    """
    将数据移动到指定设备 (GPU/CPU) 上。
    支持 torch.Tensor 和 list 类型的输入。

    Args:
        datas: torch.Tensor 或 list 或 transformers.tokenization_utils_base.BatchEncoding（clip分词器的输出）
        device: 目标设备 (如 'cuda' 或 'cpu')

    Returns:
        移动后的数据，类型与输入保持一致
    """
    if len(datas) == 1:
        datas = datas[0]  # 假设 data 是一个元组或列表
    if isinstance(datas, torch.Tensor):
        # 如果是张量，直接移动到指定设备
        return datas.to(device)
    elif isinstance(datas, list) or isinstance(datas, tuple):
        # 如果是列表，对其中的每个元素递归调用 move_to_device
        return [move_to_device(item, device) for item in datas]
    # elif isinstance(datas, tuple):
    #     # 确保所有张量的形状正确
    #     return {
    #         key: value.to(device).squeeze(1) if value.dim() == 3 and value.size(1) == 1 else value.to(device)
    #         for key, value in datas.items()
    #     }
    else:
        raise TypeError(f"Unsupported data type: {type(datas)}. Expected torch.Tensor or list.")


def create_experiment_dirs(data_path: str):
    """
    Create experiment directories for saving model weights and results.
    :param data_path: The dataset path to construct the experiment name.
    """
    # Construct the experiment name based on the data path and current time
    path = f"exp/{data_path.split('/')[-1]}_{datetime.now().strftime('%mM_%dD_%HH')}_(current)"

    # Check if the experiment directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "weigh"))
        os.makedirs(os.path.join(path, "result"))
        print(f"Experiment directories created at: {path}")
    else:
        print(f"Experiment directory already exists: {path}")

    return path


def setup_logger(args, exp_name):
    """
    定义日志，保存训练参数信息。
    @param args: 训练时参数信息
    @param exp_name: 日志保存的地址文件夹

    @return:
    """
    # 确保日志文件包含具体文件名
    log_filename = f"{exp_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,  # 日志保存地址
        level=logging.INFO,  # 日志保存级别
        format="%(asctime)s - %(message)s",  # 日志保存格式 时间戳-信息
        datefmt="%Y-%m-%d %H:%M:%S"  # 日志时间戳格式
    )
    # 保存参数信息
    logging.info("========== Experiment Parameters ==========")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("===========================================")


if __name__ == '__main__':

    # 参数设置
    args = parse_args()

    # 设置GPU优化分配机制
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 加载模型(如果使用预处理模型，修改最后一层，固定之前的权重)
    model = TimeFreqToText().to(args.device)

    # 定义优化器
    # optimizer = optim.Adam(lr=lr, params=model.parameters(), weight_decay=0.005)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # ReduceLROnPlateau: 当监测的指标不再改善时减少学习率,mode='min': 当监测指标（如验证集损失）不再下降时减少学习率。factor: 学习率减少的比例。
    # patience: 等待多少个 epoch 无改善后才减少学习率。
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)

    # 定义损失函数
    cross_loss = nn.CrossEntropyLoss()

    # 实验指标
    valid_loss_min = np.Inf  # track change in validation loss
    best_accuracy = 0  # 最优准确率
    train_losses, val_losses, val_accuracies = [], [], []

    # 保存结果
    exp_name = create_experiment_dirs(args.data_path)
    setup_logger(args, exp_name)
    logging.info(str(model))  # 记录模型架构

    # 读取数据集
    train_loader, test_loader = load_radioml_dataset(args.data_path, batch_size=args.batch_size,
                                                     num_workers=args.num_workers)

    # 开始训练
    for epoch in tqdm(range(1, args.epochs + 1)):
        # keep track of training and validation loss
        epoch_losses = {"train_loss": 0.0, "valid_loss": 0.0}

        right_sample, total_sample = 0, 0

        ###################
        # 训练集的模型 #
        ###################
        model.train()  # 作用是启用batch normalization和drop out

        for data, target in train_loader:
            # 移动所有数据和标签到 GPU
            data = data.to(args.device)
            data = data.unsqueeze(1)  # 结果的形状将变为 (batch_size, 1, 2，128)
            target = target.to(args.device)

            # 训练逻辑
            # clear the gradients of all optimized variables（清除梯度）
            optimizer.zero_grad()
            # (正向传递：通过向模型传递输入来计算预测输出)
            feature = model(data)

            feature = feature.view(-1, feature.size(-1))
            # 将 captions 展平为一个 (batch_size * seq_len) 的向量
            target = target.view(-1)

            # 计算损失，detailed_losses包含每种损失的损失值，如果对应的损失函数没有使用则为0
            loss = cross_loss(feature, target)
            # （反向传递：计算损失相对于模型参数的梯度）
            loss.backward()
            # 执行单个优化步骤（参数更新）
            optimizer.step()
            # update training loss（更新损失）
            epoch_losses["train_loss"] += loss.item() * target.size(0)

        ######################
        # 验证集的模型#
        ######################
        # 反转 VOCAB 字典，得到 {index: word}
        index_to_word = {idx: word for word, idx in VOCAB.items()}

        model.eval()  # 验证模型
        with torch.no_grad():
            outputs, targets = [], []
            features = []
            for data, target in test_loader:
                data = data.to(args.device)
                data = data.unsqueeze(1)  # 结果的形状将变为 (batch_size, 1, 2，128)
                target = target.to(args.device)

                output_feature = model(data)

                # 展平输出和目标
                output_feature = output_feature.view(-1, output_feature.size(-1))
                target = target.view(-1)

                loss = cross_loss(output_feature, target)
                epoch_losses["valid_loss"] += loss.item() * target.shape[0]

                # 计算预测结果
                _, pred = torch.max(output_feature, 1)
                correct_tensor = pred.eq(target.data.view_as(pred))

                # 存储输出和目标
                targets.append(target.cpu())
                pred = pred.cpu()
                outputs.append(pred)

                # 存储特征
                features.append(output_feature)

                # 输出图像的描述（这里假设你已经有 vocab 和 pred_text 的映射关系）
                pred_text = [index_to_word[idx.item()] for idx in pred]
                # print("Predicted Caption: ", " ".join(pred_text))

                # 统计正确预测的数量
                total_sample += target.size(0)
                right_sample += correct_tensor.sum().item()

            # 计算准确率
            accuracy = right_sample / total_sample
            val_accuracies.append(accuracy)

            # 计算平均损失
            num_train_samples = len(train_loader.sampler)
            num_valid_samples = len(test_loader.sampler)

            epoch_losses["train_loss"] /= num_train_samples
            epoch_losses["valid_loss"] /= num_valid_samples

            # 显示训练集与验证集的指标
            print(
                f"Epoch: {epoch} \tAccuracy:{100 * accuracy:.2f}% \tTotal samples: {total_sample} \tTraining Loss: {epoch_losses['train_loss']:.6f} "
                f"\tValidation Loss: {epoch_losses['valid_loss']:.6f}")

            # 自适应更新优化器学习率
            scheduler.step(epoch_losses["valid_loss"])

            # 保存实验结果
            train_losses.append(epoch_losses["train_loss"])
            val_losses.append(epoch_losses["valid_loss"])
            # 构造完整日志内容
            log_message = (
                f"Epoch {epoch}: "
                f"Train Loss = {epoch_losses['train_loss']:.6f}, "
                f"Val Loss = {epoch_losses['valid_loss']:.6f}, "
                f"Accuracy = {100 * accuracy:.2f}%, "
            )

            # 记录日志
            logging.info(log_message)

            model_file = exp_name + "/weigh/"
            # 如果当前准确率比之前的最佳准确率高，保存模型和混淆矩阵
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"New best accuracy: {best_accuracy:.4f}. Saving model...")
                torch.save(model.state_dict(), model_file + "best_accuracy.pth")
                plot_confusion_matrix(torch.cat(outputs, -1).reshape(-1), torch.cat(targets, -1).reshape(-1),
                                      save_path=exp_name + "/result/", acc=str(epoch) + '_' + str(best_accuracy))
            # 如果验证集损失函数减少，就保存模型。
            if epoch_losses["valid_loss"] <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                epoch_losses[
                                                                                                    "valid_loss"]))
                torch.save(model.state_dict(), exp_name + "/weigh/valid_loss_min.pth")
                valid_loss_min = epoch_losses["valid_loss"]

        # 清空 GPU 上未使用的缓存内存，优化显存管理
        torch.cuda.empty_cache()

    # 保存最终模型
    print(f"Saving final model...")
    torch.save(model.state_dict(), exp_name + "/weigh/final_model.pth")

    # 绘制训练集和测试集损失函数曲线
    plot_loss.save_loss(train_losses, exp_name + "/result/", "train_loss")
    plot_loss.save_loss(val_losses, exp_name + "/result/", "val_loss")

    # 绘制测试集准确率曲线
    plot_acc.save_accuracy_curve(accuracy_list=val_accuracies, file_path=exp_name + "/result/" + "val_acc.png")

    new_exp_name = exp_name[:-9] + "_" + "{:.4f}".format(best_accuracy)
    if os.path.exists(exp_name):
        os.rename(exp_name, new_exp_name)
