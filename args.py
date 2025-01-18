import argparse
import json



def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    # 模型参数,# 默认值可以是 ModelName 枚举中的某一项,choices:限制可选项为枚举值
    parser.add_argument('--input_size', type=int, default=32, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=50, help="Number of classes")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')

    # 损失函数
    parser.add_argument('--temperature', type=int, default=0.7, help='sup con loss Temperature')
    parser.add_argument('--margin', type=int, default=0.3, help='triplet loss margin')


    # 优化器参数   # 学习策略（早停 + 学习率更新方式）
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--factor', type=int, default=0.5, help='Weight decay factor')
    parser.add_argument('--patience', type=int, default=20, help='当监测指标patience轮不下降时，衰减factor')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data/RML2016.10a_dict.pkl', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument("--str_lengths", type=int, default=25, help="Lengths of feature strings")

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()
