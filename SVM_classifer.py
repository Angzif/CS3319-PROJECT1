import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
import os

def svm_classifier(train_features, train_labels, test_features, test_labels, C=1e-3, k_fold=5):
    """
    使用线性SVM进行分类
    :param train_features: 训练集特征
    :param train_labels: 训练集标签
    :param test_features: 测试集特征
    :param test_labels: 测试集标签
    :param C: SVM的正则化参数
    :param k_fold: K折交叉验证的折数
    :return: 测试集上的准确率
    """
    # 初始化线性SVM
    svm = SVC(kernel='linear', C=C)

    # K折交叉验证
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, train_features, train_labels, cv=kf, scoring='accuracy')
    print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # 训练模型
    svm.fit(train_features, train_labels)

    # 测试集预测
    test_predictions = svm.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(test_labels, test_predictions))

    return test_accuracy


if __name__ == "__main__":
    # 加载处理后的数据
    output_dir = "dataset/processed_data"
    train_features = np.loadtxt(os.path.join(output_dir, "train_features.txt"))
    train_labels = np.loadtxt(os.path.join(output_dir, "train_labels.txt"))
    test_features = np.loadtxt(os.path.join(output_dir, "test_features.txt"))
    test_labels = np.loadtxt(os.path.join(output_dir, "test_labels.txt"))

    # SVM分类
    svm_classifier(train_features, train_labels, test_features, test_labels, C=1e-3, k_fold=5)