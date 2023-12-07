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
import paddle.nn.functional as f

log_path = './runs/river_ice_experiment_rate_0.01_2000'

def ms_get_int(filename: str):
    spl = filename.split('_')
    spl2 = spl[len(spl) - 1].split('.')
    return int(spl2[0])

class RiverIceDataset(io.Dataset):
    # data loader
    def __init__(self, mode='train'):
        super(RiverIceDataset, self).__init__()
        root_path = './dataset/'
        train_filename = os.path.join(root_path, 'ipc_ri_ids_train_modify_norm.csv')
        val_filename = os.path.join(root_path, 'ipc_ri_ids_val_modify_norm.csv')
        test_filename = os.path.join(root_path, 'ipc_ri_ids_test_modify_norm.csv')
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
    logwriter = LogWriter(logdir=log_path)

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



    # 记录训练过程中的损失函数变化情况
    train_scores = []
    val_scores = []
    train_losses = []
    val_losses = []
    g_best_acc = 0
    g_best_epoch = 0

    # 设置迭代次数
    epochs = 2000

    # 设置优化器
    # learning_rate = 0.001,
    # beta1 = 0.9,
    # beta2 = 0.999,
    # epsilon = 1e-8,
    # parameters = None,
    # weight_decay = None,
    # grad_clip = None,
    # lazy_mode = False,
    # multi_precision = False,
    # use_multi_tensor = False,
    # name = None,
    optim = paddle.optimizer.Adam(parameters=fnn_model.parameters(), learning_rate=0.1)
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
            paddle.save(fnn_model.state_dict(), os.path.join(log_path, 'best.pdparams'))
            best_acc = avg_acc
            g_best_acc = best_acc
            g_best_epoch = epoch + 1
            print("epoch: {}, best acc".format(epoch + 1))

        tloss = {'train': train_avg_loss, 'val': avg_loss}
        tacc = {'train': train_avg_acc, 'val': avg_acc}
        logwriter.add_scalars(main_tag="avg_loss", tag_scalar_dict=tloss, step=epoch)
        logwriter.add_scalars(main_tag="avg_acc", tag_scalar_dict=tacc, step=epoch)
        train_scores.append(train_avg_acc)
        val_scores.append(avg_acc)
        train_losses.append(train_avg_loss)
        val_losses.append(avg_loss)
        fnn_model.train()
    np.savetxt(os.path.join(log_path, 'train_scores.csv'), train_scores, delimiter=',',
               fmt='%10.4f')
    np.savetxt(os.path.join(log_path, 'val_scores.csv'), val_scores, delimiter=',',
               fmt='%10.4f')
    np.savetxt(os.path.join(log_path, 'train_losses.csv'), train_losses, delimiter=',',
               fmt='%10.4f')
    np.savetxt(os.path.join(log_path, 'val_losses.csv'), val_losses, delimiter=',',
               fmt='%10.4f')
    with open(os.path.join(log_path, 'best.txt'), 'w') as f:
        f.write('{} {}'.format(g_best_epoch, g_best_acc))
    print('end.')


def evaluate():
    test_dataset = RiverIceDataset(mode='test')
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    print("length of test set: ", len(test_dataset))

    model_state_dict = paddle.load(os.path.join(log_path, 'best.pdparams'))
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

    # evaluate: loss is: 0.017309415868483648, acc is: 0.9983595800524935


def prediction():
    # area: 4871.5786 32.1140
    # max velocity: 8.6385 0.0000
    # avg velocity: 3.9145 0.0000

    model_state_dict = paddle.load(os.path.join(log_path, 'best.pdparams'))
    fnn_model = RiverIceMLP(input_size=5, output_size=5, hidden_size=10)
    fnn_model.set_state_dict(model_state_dict)
    fnn_model.eval()

    area_max = 4871.5786
    area_min = 32.1140
    max_velocity_max = 8.6385
    max_velocity_min = 0.0
    avg_velocity_max = 3.9145
    avg_velocity_min = 0.0

    ids = [1, 6, 10, 6, 3]
    for idx_stage, idx_videos in enumerate(ids):
        for idx_video in range(idx_videos):
            segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{idx_stage+1}/{idx_video+1}'
            point_track_velocity_path = f'./dataset/RiverIceFixedCameraPointTrackVelocity/{idx_stage+1}/{idx_video+1}'
            stage_path = Path(f'./dataset/RiverIceFixedCameraSegmentation/{idx_stage+1}/{idx_video+1}/stage')
            stage_path.mkdir(exist_ok=True, parents=True)

            file_list = os.listdir(os.path.join(segmentation_path, 'density'))
            file_list = sorted(file_list, key=lambda item: ms_get_int(item))

            for idx, frame in enumerate(file_list):
                density_area_filename = os.path.join(segmentation_path, 'density', frame)
                motion_intensity_filename = os.path.join(segmentation_path, 'motion_intensity', frame)
                point_velocity_filename = os.path.join(point_track_velocity_path, frame)

                stage_filename = os.path.join(stage_path, frame)


                concentration = 0.0
                area = 0.0
                concentration, area = np.genfromtxt(density_area_filename, delimiter=' ', dtype=float)
                motion_intensity = 0.0
                motion_intensity, _, _ = np.genfromtxt(motion_intensity_filename, delimiter=' ',
                                                                                    dtype=float)
                motion_intensity *= 50
                if motion_intensity > 1.0:
                    motion_intensity = 1.0

                point_dict_list = []
                with open(point_velocity_filename, 'r') as point_velocity_file:
                    content = point_velocity_file.read()
                    point_dict_list = eval(content)

                max_velocity = 0.0
                avg_velocity = 0.0
                for pd in point_dict_list:
                    velocity = pd['velocity']
                    max_velocity = velocity if velocity > max_velocity else max_velocity
                    avg_velocity += velocity
                if len(point_dict_list) > 1:
                    avg_velocity /= len(point_dict_list)

                area = (area-area_min)/(area_max-area_min)
                max_velocity = (max_velocity-max_velocity_min)/(max_velocity_max-max_velocity_min)
                avg_velocity = (avg_velocity-avg_velocity_min)/(avg_velocity_max-avg_velocity_min)

                x_data = [concentration, area, motion_intensity, max_velocity, avg_velocity]  # 测试数据
                y_data = [idx_stage]  # 测试数据标签
                y_data = np.expand_dims(y_data, axis=1)

                x_data = paddle.to_tensor(x_data)

                predicts = fnn_model(x_data)  # 预测结果
                predicts = f.softmax(predicts)
                preds = paddle.argmax(predicts, axis=0, dtype='int32')
                preds = preds.item()
                bTrue = 0
                if preds == idx_stage:
                    bTrue = 1

                with open(stage_filename, 'w') as file:
                    file.write('{:d} {:d}'.format(preds, bTrue))
                    file.close()


if __name__ == '__main__':
    # main()
    # prediction()
    evaluate()
