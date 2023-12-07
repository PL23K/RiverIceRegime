import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


log_path = './runs/river_ice_experiment_svm'

# 加载数据集
def load_data_train(shuffle=True):
    """
    加载鸢尾花数据
    输入：
        - shuffle：是否打乱数据，数据类型为bool
    输出：
        - X：特征数据，shape=[150,4]
        - y：标签数据, shape=[150]
    """
    t_datas = np.loadtxt(fname='./dataset/ipc_ri_ids_train_modify_norm.csv', delimiter=",", dtype=np.float32)
    # 加载原始数据
    X = t_datas[:, (4, 5, 6, 9, 10)]
    y = t_datas[:, 1].astype(np.int32)
    y -= 1

    # 如果shuffle为True，随机打乱数据
    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]
    return X, y


def load_data_val():
    t_datas = np.loadtxt(fname='./dataset/ipc_ri_ids_val_modify_norm.csv', delimiter=",", dtype=np.float32)
    # 加载原始数据
    X = t_datas[:, (4, 5, 6, 9, 10)]
    y = t_datas[:, 1].astype(np.int32)
    y -= 1

    return X, y


def load_data_test():
    t_datas = np.loadtxt(fname='./dataset/ipc_ri_ids_test_modify_norm.csv', delimiter=",", dtype=np.float32)
    # 加载原始数据
    X = t_datas[:, (4, 5, 6, 9, 10)]
    y = t_datas[:, 1].astype(np.int32)
    y -= 1


    return X, y

def main():

    X_train, y_train = load_data_train(shuffle=True)
    print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)
    X_val, y_val = load_data_val()
    print("X_val shape: ", X_val.shape, "y_val shape: ", y_val.shape)
    X_test, y_test = load_data_test()
    print("X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)

    clf_linear = svm.SVC(kernel='linear', C=1.0)
    clf_linear.fit(X_train, y_train)

    print(f"linear Kernel 's score: {clf_linear.score(X_test, y_test)}")

    clf_poly = svm.SVC(kernel='poly', C=1.0, degree=5)
    clf_poly.fit(X_train, y_train)

    print(f"poly Kernel 's score: {clf_poly.score(X_test, y_test)}")

    # 创建一个RBF(高斯内核)的SVM模型，这里的效果不是很明显
    clf_rbf = svm.SVC(kernel='rbf', C=1.0)
    clf_rbf.fit(X_train, y_train)

    print(f"rbf Kernel 's score: {clf_rbf.score(X_test, y_test)}")

    clf_sigmoid = svm.SVC(kernel='sigmoid', C=1.0)
    clf_sigmoid.fit(X_train, y_train)

    print(f"sigmoid Kernel 's score: {clf_sigmoid.score(X_test, y_test)}")


if __name__ == '__main__':
    main()

# linear Kernel 's score: 0.8966535433070866
# poly Kernel 's score: 0.9645669291338582
# rbf Kernel 's score: 0.9189632545931758
