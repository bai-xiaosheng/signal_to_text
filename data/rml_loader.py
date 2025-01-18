# 自定义数据集类
import pickle

import jieba
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

from constants.text_constants import TEXT


# 将文本转换为词索引序列
def text_to_sequence(text, vocab):
    # 遍历文本中的每个 token，并将其转换为词汇表中的索引
    return torch.tensor([vocab[token] if token in vocab else vocab["<unk>"] for token in text], dtype=torch.long)


# 文本预处理函数：分词
def preprocess_text(text):
    if not text.strip():
        # 如果文本为空（或仅包含空白字符），返回一个默认值
        return ["<pad>"]
    # 使用 jieba 进行分词
    words = jieba.cut(text)
    return list(words)


# 构建词汇表
def build_vocab(descriptions):
    # 使用 build_vocab_from_iterator 构建词汇表
    vocab = build_vocab_from_iterator(descriptions, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])  # 未知词用 <unk> 代替
    return vocab


# 将文本转换为词索引序列
def text_to_sequence(text, vocab):
    # 遍历文本中的每个 token，并将其转换为词汇表中的索引
    return torch.tensor([vocab[token] if token in vocab else vocab["<unk>"] for token in text], dtype=torch.long)


class RadioMLDataset(Dataset):
    def __init__(self, file_path, type="train", max_seq_len=15):
        """
        初始化数据集
        :param file_path: 数据集路径 (.pkl 文件)
        :param type: 训练还是测试
        :param max_seq_len: 文本序列最大长度
        """
        # 检查读取数据类型，只允许train或者test
        if type != "train" and type != "test":
            raise ValueError(f"type {type} not supported")
        # 加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')

        self.samples = []  # 存放样本
        self.labels = []  # 存放标签
        self.label_to_index = {}  # 用于保存标签与索引的映射关系

        # 遍历数据，生成样本和对应的标签
        for idx, ((mod, snr), samples) in enumerate(data.items()):
            for sample in samples:
                # 将 I/Q 数据转换为 tensor 并存储
                self.samples.append([torch.tensor(sample[0]), torch.tensor(sample[1])])
                # 生成标签（包括调制方式和信噪比）
                self.labels.append(f"信号调制方式为{mod}且对应信噪比为{snr}dB")  # 使用 mod 和 snr 的组合作为标签

        # 创建标签到索引的映射
        self.label_to_index = {label: idx for idx, label in enumerate(set(self.labels))}

        # 按标签对数据进行划分
        self.train_data = []
        self.test_data = []
        self.type = type
        self._split_data()

        # self.vocab = VOCAB  # 直接将词汇表保存为常量
        self.max_seq_len = max_seq_len
        # 构建词汇表
        # 2. 准备训练数据
        descriptions = TEXT
        # 3. 分词所有文本
        processed_descriptions = [preprocess_text(desc) for desc in descriptions]
        # 4. 构建词汇表
        self.vocab = build_vocab(processed_descriptions)
        # 打印词汇表
        print("词汇表：", self.vocab.get_stoi())  # 查看词汇表的映射

    def _split_data(self):
        # 按标签对数据进行分组
        class_data = {}
        for idx, label in enumerate(self.labels):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(idx)

        # 按 9:1 划分每个类别的数据
        for label, indices in class_data.items():
            train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
            self.train_data.extend([(idx, label) for idx in train_indices])
            self.test_data.extend([(idx, label) for idx in test_indices])

    def __len__(self):
        if self.type == "train":
            return len(self.train_data)
        elif self.type == "test":
            return len(self.test_data)
        else:
            raise ValueError(f"type {type} not supported")

    def __getitem__(self, idx):
        """
        获取指定索引的数据和标签
        :param idx: 索引
        :return: 时频图像，标签
        """
        if self.type == 'train':
            data_idx, label = self.train_data[idx]  # 获取训练数据索引和标签
        elif self.type == 'test':
            data_idx, label = self.test_data[idx]
        else:
            raise ValueError(f"type {self.type} is not supported")
        iq_data = self.samples[data_idx]  # 获取对应的 I/Q 数据
        # 将信号转为（2，128）
        signal = torch.stack(iq_data, dim=0)  # 结果形状为 (2, 128)

        # 处理标签文本
        # 4. 分词
        processed_descriptions = preprocess_text(label)
        # 5. 将文本转换为词索引序列
        sequences = text_to_sequence(processed_descriptions, self.vocab)  # 转换为词索引序列
        # sequences = torch.tensor(sequences, dtype=torch.long)  # 假设此处将文本转为词索引序列的结果是一个张量

        # 6. 文本填充（确保所有文本长度一致）
        padded_labels = F.pad(sequences, (0, self.max_seq_len - sequences.size(0)), value=self.vocab["<pad>"])
        return signal, padded_labels


def load_radioml_dataset(file_path, batch_size=64, num_workers=2):
    """
    加载 RadioML2016.10a 数据集并生成 DataLoader
    :param file_path: 数据集路径 (.pkl 文件)
    :param batch_size: 每批次样本数
    :param num_workers: 数据加载的并行线程数
    :return: 训练集和测试集的 DataLoader
    """

    train_dataset = RadioMLDataset(file_path, "train")

    # 获取训练集的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = RadioMLDataset(file_path, "test")

    # 获取训练集的 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    # 指定 RadioML 2016.10a 数据集的路径
    file_path = "./RML2016.10a_dict.pkl"  # 替换为实际路径

    # 创建训练和测试数据加载器
    train_loader, test_loader = load_radioml_dataset(file_path)

    # 显示一个批次的数据
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}")
        print(f"数据形状: {data.shape}")  # (batch_size, 1, time, freq)
        print(f"标签: {labels}")
        break
    for data, labels in test_loader:
        print(f"数据形状: {data.shape}")  # (batch_size, 1, time, freq)
        print(f"标签: {labels}")
        break
