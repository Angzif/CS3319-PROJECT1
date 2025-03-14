import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os

def find_bestC(train_feature, train_label):
    """
    使用GridSearchCV寻找最优的C参数
    :param train_feature: 训练集特征
    :param train_label: 训练集标签
    :return: 最优的C参数
    """
    param_grid = [
        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
         'kernel': ['linear'], 
         'decision_function_shape': ['ovr']},
    ]

    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=3, n_jobs=4)
    grid_search.fit(train_feature, train_label)

    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    return grid_search.best_params_['C']


if __name__ == "__main__":
    # 数据目录（与 data_preprocess.py 中的 output_dir 一致）
    output_dir = "dataset/processed_data"
    train_label_path = os.path.join(output_dir, "train_labels.txt")
    train_feature_path = os.path.join(output_dir, "train_features.txt")

    # 检查文件路径
    print(f"Loading labels from: {train_label_path}")
    print(f"Loading features from: {train_feature_path}")

    # 加载数据
    train_label = np.loadtxt(train_label_path)
    train_feature = np.loadtxt(train_feature_path)

    # 寻找最优的C参数
    best_C = find_bestC(train_feature, train_label)
    print(f"Best C parameter: {best_C}")