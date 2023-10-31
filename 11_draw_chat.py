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
import math

import torch
import cv2
import os
import numpy as np
import csv
from pathlib import Path
from point_tracker.pips.tracker import PipsPointTracker
import point_tracker.utils.basic
import time
from matplotlib import cm
import matplotlib.pyplot as plt


def ms_get_int(filename: str):
    spl = filename.split('_')
    spl2 = spl[len(spl) - 1].split('.')
    return int(spl2[0])


def main():
    root_path = './dataset/'
    total_filename = os.path.join(root_path, 'ipc_ri_ids.csv')
    title = ['id', 'stage', 'video', 'sort', 'iceDensity', 'iceArea', 'motionIntensity', 'motionDensity',
     'motionDivergence', 'maxVelocity', 'avgVelocity']

    datas = np.loadtxt(fname=total_filename,
                    delimiter=",",  # 数据以什么符号分割，CSV为","分割
                    dtype="float64",  # 指定读取后的数据类型
                    skiprows=1,  # 要跳过前多少行数，0表示不跳过，比如设置为2表示跳过前两行
                    #usecols=(0, 1, 2, 3),  # 要读那几列，0为第一列
                    unpack=False,  # 是否要转置
                    ndmin=0,  # 指定最小维度
                    encoding=None,  # 指定编码类型
                    max_rows=None  # 最多读取多少行
                    )
    print(datas.shape)

    x = datas[:, 0]
    y = datas[:, 9]
    plt.plot(x, y)
    plt.show()
    # np.savetxt(os.path.join(root_path, 'ipc_ri_ids-id.csv'), X=x, fmt=['%d'], delimiter=',', newline=',')
    print(max(y))
    # np.savetxt(os.path.join(root_path, 'ipc_ri_ids-iceDensity.csv'), X=y, fmt=['%.4f'], delimiter=',', newline=',')
    # np.savetxt(os.path.join(root_path, 'ipc_ri_ids-iceArea.csv'), X=y, fmt=['%.4f'], delimiter=',', newline=',')
    # np.savetxt(os.path.join(root_path, 'ipc_ri_ids-motionIntensity.csv'), X=y, fmt=['%.4f'], delimiter=',', newline=',')
    # np.savetxt(os.path.join(root_path, 'ipc_ri_ids-maxVelocity.csv'), X=y, fmt=['%.4f'], delimiter=',', newline=',')
    y = datas[:, 10]
    # np.savetxt(os.path.join(root_path, 'ipc_ri_ids-avgVelocity.csv'), X=y, fmt=['%.4f'], delimiter=',', newline=',')
    print(max(y))
    print('end.')


if __name__ == '__main__':
    main()
