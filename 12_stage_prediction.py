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

import cv2
import os
import numpy as np
from pathlib import Path
import time

import paddle
import paddle.io as io
from paddle import nn
from visualdl import LogWriter

class RiverIceDataset(io.Dataset):
    # data loader
    def __init__(self, mode='train'):
        super(RiverIceDataset, self).__init__()
        root_path = './dataset/'
        train_filename = os.path.join(root_path, 'ipc_ri_ids_train.csv')
        val_filename = os.path.join(root_path, 'ipc_ri_ids_val.csv')
        test_filename = os.path.join(root_path, 'ipc_ri_ids_test.csv')
        title = ['id', 'stage', 'video', 'sort', 'iceDensity', 'iceArea', 'motionIntensity', 'motionDensity',
                 'motionDivergence', 'maxVelocity', 'avgVelocity']

        if mode == 'train':
            datas = np.loadtxt(fname=train_filename, delimiter=",", dtype=np.float32)
            self.X = datas[:, (4, 5, 6, 9, 10)]
            self.y = datas[:, 1].astype(np.int32)
            self.y -= 1
            self.y = np.expand_dims(self.y, axis=1)
        elif mode == 'val':
            datas = np.loadtxt(fname=val_filename, delimiter=",", dtype=np.float32)
            self.X = datas[:, (4, 5, 6, 9, 10)]
            self.y = datas[:, 1].astype(np.int32)
            self.y -= 1
            self.y = np.expand_dims(self.y, axis=1)
        else:
            datas = np.loadtxt(fname=test_filename, delimiter=",", dtype=np.float32)
            self.X = datas[:, (4, 5, 6, 9, 10)]
            self.y = datas[:, 1].astype(np.int32)
            self.y -= 1
            self.y = np.expand_dims(self.y, axis=1)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# Feedforward Neural Network
class RiverIceMLP(nn.Layer):
    def __init__(self, input_size, output_size, hidden_size):
        super(RiverIceMLP, self).__init__()
        # fcn 1
        self.fc1 = nn.Linear(
            input_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
        )
        # fcn 2
        self.fc2 = nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
        )
        # act
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.act(outputs)
        outputs = self.fc2(outputs)
        return outputs


def main():
    logwriter = LogWriter(logdir='./runs/river_ice_experiment')

    train_dataset = RiverIceDataset(mode='train')
    val_dataset = RiverIceDataset(mode='val')
    test_dataset = RiverIceDataset(mode='test')

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1,
                                        drop_last=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=1,
                                      drop_last=True)

    print("length of train set: ", len(train_dataset))
    print("length of val set: ", len(val_dataset))
    print("length of test set: ", len(test_dataset))

    fnn_model = RiverIceMLP(input_size=5, output_size=5, hidden_size=10)

    # 设置迭代次数
    epochs = 1000

    # 设置优化器
    optim = paddle.optimizer.Adam(parameters=fnn_model.parameters())
    # 设置损失函数
    loss_fn = paddle.nn.CrossEntropyLoss()
    fnn_model.train()

    best_acc = 0.0
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]  # 训练数据
            y_data = data[1]  # 训练数据标签
            predicts = fnn_model(x_data)  # 预测结果

            # 计算损失 等价于 prepare 中loss的设置
            loss = loss_fn(predicts, y_data)

            # 计算准确率 等价于 prepare 中metrics的设置
            acc = paddle.metric.accuracy(predicts, y_data)

            # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
            # 反向传播
            loss.backward()

            # 更新参数
            optim.step()
            # 梯度清零
            optim.clear_grad()

        # if (epoch + 1) % 10 == 0:
        fnn_model.eval()
        # train
        accuracies = []
        losses = []
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]  # 测试数据
            y_data = data[1]  # 测试数据标签
            predicts = fnn_model(x_data)  # 预测结果
            loss = loss_fn(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            accuracies.append(acc.item())
            losses.append(loss.item())
        train_avg_acc, train_avg_loss = np.mean(accuracies), np.mean(losses)

        # val
        accuracies = []
        losses = []
        for batch_id, data in enumerate(val_loader()):
            x_data = data[0]  # 测试数据
            y_data = data[1]  # 测试数据标签
            predicts = fnn_model(x_data)  # 预测结果
            loss = loss_fn(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            accuracies.append(acc.item())
            losses.append(loss.item())
        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("epoch: {}, loss is: {}, acc is: {}".format(epoch + 1, avg_loss, avg_acc))
        if avg_acc > best_acc:
            paddle.save(fnn_model.state_dict(), "./output/best.pdparams")
            best_acc = avg_acc
            print("epoch: {}, best acc".format(epoch + 1))

        tloss = {'train': train_avg_loss, 'val': avg_loss}
        tacc = {'train': train_avg_acc, 'val': avg_acc}
        logwriter.add_scalars(main_tag="avg_loss", tag_scalar_dict=tloss, step=epoch)
        logwriter.add_scalars(main_tag="avg_acc", tag_scalar_dict=tacc, step=epoch)
        fnn_model.train()
    print('end.')


def evaluate():
    test_dataset = RiverIceDataset(mode='test')
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    print("length of test set: ", len(test_dataset))

    model_state_dict = paddle.load("./output/best.pdparams")
    fnn_model = RiverIceMLP(input_size=5, output_size=5, hidden_size=10)
    fnn_model.set_state_dict(model_state_dict)

    loss_fn = paddle.nn.CrossEntropyLoss()

    # test
    accuracies = []
    losses = []
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]  # 测试数据
        y_data = data[1]  # 测试数据标签
        predicts = fnn_model(x_data)  # 预测结果
        loss = loss_fn(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        accuracies.append(acc.item())
        losses.append(loss.item())
    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    print("evaluate: loss is: {}, acc is: {}".format(avg_loss, avg_acc))


if __name__ == '__main__':
    # main()
    evaluate()
