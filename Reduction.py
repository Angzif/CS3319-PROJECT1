import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.svm import SVC
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import random

# 设置随机种子
random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)

# 数据目录
father_dir = "dataset/processed_data"  # 与 data_preprocess.py 中的 output_dir 一致
train_feature_path = os.path.join(father_dir, "train_features.txt")
train_label_path = os.path.join(father_dir, "train_labels.txt")
test_feature_path = os.path.join(father_dir, "test_features.txt")
test_label_path = os.path.join(father_dir, "test_labels.txt")

# 加载数据
train_feature = np.loadtxt(train_feature_path)
train_label = np.loadtxt(train_label_path)
test_feature = np.loadtxt(test_feature_path)
test_label = np.loadtxt(test_label_path)

def benchmark():
    """
    基准测试：使用原始特征进行 SVM 分类
    """
    svm = SVC(kernel='linear', random_state=random_state)
    svm.fit(train_feature, train_label)
    train_score = svm.score(train_feature, train_label)
    test_score = svm.score(test_feature, test_label)
    print(f"原始特征的性能 - 训练集准确率: {train_score:.4f}, 测试集准确率: {test_score:.4f}")
    return train_score, test_score

def genetic_reduction(n_components, population_size=20, num_generations=10):
    """
    使用遗传算法进行特征选择
    :param n_components: 目标特征数量
    :param population_size: 种群大小
    :param num_generations: 迭代次数
    :return: 选择的特征索引
    """
    print(f"使用遗传算法选择 {n_components} 个特征...")

    # 初始化种群
    num_features = train_feature.shape[1]
    population = [np.random.choice([0, 1], size=num_features, p=[0.5, 0.5]) for _ in range(population_size)]

    def fitness(individual):
        """
        适应度函数：使用 SVM 分类器的准确率
        :param individual: 个体（二进制编码）
        :return: 适应度值
        """
        selected_features = individual == 1
        if np.sum(selected_features) == 0:  # 如果没有选择任何特征，适应度为 0
            return 0
        X_train_selected = train_feature[:, selected_features]
        X_test_selected = test_feature[:, selected_features]
        svm = SVC(kernel='linear', random_state=random_state)
        svm.fit(X_train_selected, train_label)
        return svm.score(X_test_selected, test_label)

    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")

        # 计算适应度
        fitness_scores = [fitness(individual) for individual in population]

        # 选择（轮盘赌选择）
        fitness_scores = np.array(fitness_scores)
        fitness_scores = fitness_scores / np.sum(fitness_scores)  # 归一化
        selected_indices = np.random.choice(range(population_size), size=population_size, p=fitness_scores)
        selected_population = [population[i] for i in selected_indices]

        # 交叉（单点交叉）
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < population_size else selected_population[0]
            crossover_point = random.randint(1, num_features - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            new_population.extend([child1, child2])

        # 变异（随机翻转）
        for individual in new_population:
            if random.random() < 0.1:  # 变异概率
                mutation_point = random.randint(0, num_features - 1)
                individual[mutation_point] = 1 - individual[mutation_point]

        population = new_population

    # 选择最优个体
    best_individual = max(population, key=fitness)
    selected_features = best_individual == 1
    print("遗传算法特征选择完成！")
    return selected_features
def evaluate_genetic_reduction(n_components):
    """
    评估遗传算法降维
    :param n_components: 目标特征数量
    :return: 训练集和测试集准确率，以及耗时
    """
    start_time = time.time()
    selected_features = genetic_reduction(n_components)
    train_F = train_feature[:, selected_features]
    test_F = test_feature[:, selected_features]
    end_time = time.time()

    # 使用 SVM 评估降维后的特征
    svm = SVC(kernel='linear', random_state=random_state)
    svm.fit(train_F, train_label)
    train_score = svm.score(train_F, train_label)
    test_score = svm.score(test_F, test_label)
    print(f"遗传算法降维到 {n_components} 个特征 - 训练集准确率: {train_score:.4f}, 测试集准确率: {test_score:.4f}")
    return train_score, test_score, end_time - start_time

def pca_reduction(n_components):
    """
    使用 PCA 进行降维
    :param n_components: 目标维度
    :return: 降维后的训练集和测试集特征
    """
    print(f"使用 PCA 降维到 {n_components} 维...")
    start_time = time.time()
    pca = PCA(n_components=n_components, random_state=random_state)
    train_F = pca.fit_transform(train_feature)
    test_F = pca.transform(test_feature)
    end_time = time.time()
    print(f"PCA 降维完成！耗时: {end_time - start_time:.2f} 秒")
    return train_F, test_F, end_time - start_time

def tsne_reduction(n_components):
    """
    使用 t-SNE 进行降维
    :param n_components: 目标维度
    :return: 降维后的训练集和测试集特征
    """
    print(f"使用 t-SNE 降维到 {n_components} 维...")
    start_time = time.time()
    tsne = TSNE(n_components=n_components, method='exact', random_state=random_state)
    train_F = tsne.fit_transform(train_feature)
    test_F = tsne.fit_transform(test_feature)
    end_time = time.time()
    print(f"t-SNE 降维完成！耗时: {end_time - start_time:.2f} 秒")
    return train_F, test_F, end_time - start_time

def lle_reduction(n_components):
    """
    使用 LLE 进行降维
    :param n_components: 目标维度
    :return: 降维后的训练集和测试集特征
    """
    print(f"使用 LLE 降维到 {n_components} 维...")
    start_time = time.time()
    lle = LocallyLinearEmbedding(n_components=n_components, random_state=random_state)
    train_F = lle.fit_transform(train_feature)
    test_F = lle.fit_transform(test_feature)
    end_time = time.time()
    print(f"LLE 降维完成！耗时: {end_time - start_time:.2f} 秒")
    return train_F, test_F, end_time - start_time

def autoencoder_reduction(n_components):
    """
    使用 AutoEncoder 进行降维
    :param n_components: 目标维度
    :return: 降维后的训练集和测试集特征
    """
    print(f"使用 AutoEncoder 降维到 {n_components} 维...")

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.tensor(train_feature, dtype=torch.float32).to(device)
    test_tensor = torch.tensor(test_feature, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 训练 AutoEncoder
    model = Autoencoder(input_dim=train_feature.shape[1], hidden_dim=n_components).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1E-4)

    start_time = time.time()
    epochs = 20
    for epoch in range(epochs):
        for data in train_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 降维
    train_F = model.encoder(train_tensor).cpu().detach().numpy()
    test_F = model.encoder(test_tensor).cpu().detach().numpy()
    end_time = time.time()
    print(f"AutoEncoder 降维完成！耗时: {end_time - start_time:.2f} 秒")
    return train_F, test_F, end_time - start_time

def evaluate_reduction(method, n_components):
    """
    评估降维方法
    :param method: 降维方法（'pca', 'lle', 'autoencoder'）
    :param n_components: 目标维度
    :return: 训练集和测试集准确率，以及耗时
    """
    if method == 'pca':
        train_F, test_F, time_cost = pca_reduction(n_components)
    elif method == 'tsne':
        train_F, test_F, time_cost = tsne_reduction(n_components)
    elif method == 'lle':
        train_F, test_F, time_cost = lle_reduction(n_components)
    elif method == 'autoencoder':
        train_F, test_F, time_cost = autoencoder_reduction(n_components)
    elif method == 'genetic':
        train_F, test_F, time_cost = evaluate_genetic_reduction(n_components)
    else:
        raise ValueError("不支持的降维方法")

    # 使用 SVM 评估降维后的特征
    svm = SVC(kernel='linear', random_state=random_state)
    svm.fit(train_F, train_label)
    train_score = svm.score(train_F, train_label)
    test_score = svm.score(test_F, test_label)
    print(f"{method.upper()} 降维到 {n_components} 维 - 训练集准确率: {train_score:.4f}, 测试集准确率: {test_score:.4f}")
    return train_score, test_score, time_cost

def compare_time(methods, dim_list):
    """
    比较不同降维方法的时间性能
    :param methods: 降维方法列表（'pca', 'lle', 'autoencoder'）
    :param dim_list: 目标维度列表
    """
    time_results = {method: [] for method in methods}

    for method in methods:
        print(f"=== 正在评估 {method.upper()} ===")
        for dim in dim_list:
            _, _, time_cost = evaluate_reduction(method, dim)
            time_results[method].append(time_cost)
        print(f"{method.upper()} 的时间性能: {time_results[method]}")

    # 打印时间比较结果
    print("\n=== 时间性能比较 ===")
    for method in methods:
        print(f"{method.upper()} 的时间性能: {time_results[method]}")

if __name__ == "__main__":
    # 基准测试
    #benchmark()

    # 比较不同降维方法的时间性能
    methods = ['genetic']
    dim_list = [512, 256, 128, 64, 32, 16]  # 目标维度列表
    compare_time(methods, dim_list)