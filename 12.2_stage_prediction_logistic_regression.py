# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import numpy as np
import paddle
from visualdl import LogWriter
from nndl import op, metric, opitimizer, RunnerV2

log_path = './runs/river_ice_experiment_logistic_5'

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

    X = paddle.to_tensor(X)
    y = paddle.to_tensor(y)

    # 如果shuffle为True，随机打乱数据
    if shuffle:
        idx = paddle.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]
    return X, y


def load_data_val():
    t_datas = np.loadtxt(fname='./dataset/ipc_ri_ids_val_modify_norm.csv', delimiter=",", dtype=np.float32)
    # 加载原始数据
    X = t_datas[:, (4, 5, 6, 9, 10)]
    y = t_datas[:, 1].astype(np.int32)
    y -= 1

    X = paddle.to_tensor(X)
    y = paddle.to_tensor(y)

    return X, y


def load_data_test():
    t_datas = np.loadtxt(fname='./dataset/ipc_ri_ids_test_modify_norm.csv', delimiter=",", dtype=np.float32)
    # 加载原始数据
    X = t_datas[:, (4, 5, 6, 9, 10)]
    y = t_datas[:, 1].astype(np.int32)
    y -= 1

    X = paddle.to_tensor(X)
    y = paddle.to_tensor(y)

    return X, y


def main():
    logwriter = LogWriter(logdir=log_path)

    # 固定随机种子
    paddle.device.set_device('gpu')
    paddle.seed(102)

    X_train, y_train = load_data_train(shuffle=True)
    print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)
    X_val, y_val = load_data_val()
    print("X_val shape: ", X_val.shape, "y_val shape: ", y_val.shape)
    X_test, y_test = load_data_test()
    print("X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)

    # 输入维度
    input_dim = 5
    # 类别数
    output_dim = 5
    # 实例化模型
    model = op.model_SR(input_dim=input_dim, output_dim=output_dim)
    # 学习率
    lr = 5

    # 梯度下降法
    optimizer = opitimizer.SimpleBatchGD(init_lr=lr, model=model)
    # 交叉熵损失
    loss_fn = op.MultiCrossEntropyLoss()
    # 准确率
    metrics = metric.accuracy

    # 实例化RunnerV2
    runner = RunnerV2(model, optimizer, metrics, loss_fn)

    # 启动训练
    runner.train([X_train, y_train], [X_val, y_val], num_epochs=2000, log_epochs=10, save_path=os.path.join(log_path, 'best_model.pdparams'))

    # tloss = {'train': train_avg_loss, 'val': avg_loss}
    # tacc = {'train': train_avg_acc, 'val': avg_acc}
    # logwriter.add_scalars(main_tag="avg_loss", tag_scalar_dict=tloss, step=epoch)
    # logwriter.add_scalars(main_tag="avg_acc", tag_scalar_dict=tacc, step=epoch)

    from nndl import plot

    plot(runner, fig_name=os.path.join(log_path, 'linear-acc3.png'))

    print('end.')


def evaluate():
    # 固定随机种子
    paddle.device.set_device('gpu')
    paddle.seed(102)

    X_test, y_test = load_data_test()
    print("X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)

    # 输入维度
    input_dim = 5
    # 类别数
    output_dim = 5
    # 实例化模型
    model = op.model_SR(input_dim=input_dim, output_dim=output_dim)
    # 学习率
    lr = 2

    # 梯度下降法
    optimizer = opitimizer.SimpleBatchGD(init_lr=lr, model=model)
    # 交叉熵损失
    loss_fn = op.MultiCrossEntropyLoss()
    # 准确率
    metrics = metric.accuracy

    # 实例化RunnerV2
    runner = RunnerV2(model, optimizer, metrics, loss_fn)
    runner.load_model(os.path.join(log_path, 'best_model.pdparams'))

    avg_loss, avg_acc = runner.evaluate([X_test, y_test])
    print("evaluate: loss is: {}, acc is: {}".format(avg_loss, avg_acc))


if __name__ == '__main__':
    main()
    # evaluate()

# 2
# [Dev] epoch: 1940, loss: 0.32389336824417114, score: 0.8589614629745483
# best accuracy performence has been updated: 0.85896 --> 0.85930