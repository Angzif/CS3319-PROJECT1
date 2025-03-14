import numpy as np
import os

def load_data(feature_path, label_path):
    """
    加载特征和标签数据
    :param feature_path: 特征文件路径
    :param label_path: 标签文件路径
    :return: 特征数据 (numpy数组), 标签数据 (numpy数组)
    """
    features = np.loadtxt(feature_path)
    labels = np.loadtxt(label_path)
    return features, labels


def split_data(features, labels, test_ratio=0.4, random_state=42):
    """
    划分训练集和测试集
    :param features: 特征数据
    :param labels: 标签数据
    :param test_ratio: 测试集比例
    :param random_state: 随机种子
    :return: 训练集特征, 测试集特征, 训练集标签, 测试集标签
    """
    np.random.seed(random_state)
    shuffle_index = np.random.permutation(len(labels))
    shuffled_features = features[shuffle_index]
    shuffled_labels = labels[shuffle_index]

    test_size = int(len(labels) * test_ratio)
    train_features = shuffled_features[:-test_size]
    test_features = shuffled_features[-test_size:]
    train_labels = shuffled_labels[:-test_size]
    test_labels = shuffled_labels[-test_size:]

    return train_features, test_features, train_labels, test_labels


def normalize_data(data):
    """
    标准化数据：均值为0，标准差为1
    :param data: 输入数据 (numpy数组)
    :return: 标准化后的数据
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


def scale_data(data, x_min=0, x_max=1):
    """
    归一化数据：缩放到指定范围 [x_min, x_max]
    :param data: 输入数据 (numpy数组)
    :param x_min: 最小值
    :param x_max: 最大值
    :return: 归一化后的数据
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val) * (x_max - x_min) + x_min
    return scaled_data


def save_data(data, save_path):
    """
    保存数据到文件
    :param data: 要保存的数据 (numpy数组)
    :param save_path: 保存路径
    """
    np.savetxt(save_path, data)


def data_preprocess(feature_path, label_path, output_dir, test_ratio=0.4, normalize=True, scale=False):
    """
    数据预处理主函数
    :param feature_path: 特征文件路径
    :param label_path: 标签文件路径
    :param output_dir: 输出目录
    :param test_ratio: 测试集比例
    :param normalize: 是否标准化数据
    :param scale: 是否归一化数据
    """
    # 加载数据
    features, labels = load_data(feature_path, label_path)
    print(f"Loaded {len(features)} samples with {features.shape[1]} features.")

    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = split_data(features, labels, test_ratio)
    print(f"Training set size: {len(train_features)}")
    print(f"Testing set size: {len(test_features)}")

    # 标准化
    if normalize:
        train_features = normalize_data(train_features)
        test_features = normalize_data(test_features)
        print("Data normalized (mean=0, std=1).")
    elif scale:
        train_features = scale_data(train_features)
        test_features = scale_data(test_features)
        print(f"Data scaled to [0, 1].")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据
    save_data(train_features, os.path.join(output_dir, "train_features.txt"))
    save_data(test_features, os.path.join(output_dir, "test_features.txt"))
    save_data(train_labels, os.path.join(output_dir, "train_labels.txt"))
    save_data(test_labels, os.path.join(output_dir, "test_labels.txt"))
    print(f"Processed data saved to {output_dir}.")

def data_preprocess_main():
    # 文件路径
    feature_path = "dataset/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
    label_path = "dataset/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
    output_dir = "dataset/processed_data"

    # 数据预处理
    data_preprocess(feature_path, label_path, output_dir, test_ratio=0.4, normalize=True, scale=False)

if __name__ == "__main__":
    data_preprocess_main()