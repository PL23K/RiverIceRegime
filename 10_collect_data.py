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
import random

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


def ms_get_int(filename: str):
    spl = filename.split('_')
    spl2 = spl[len(spl) - 1].split('.')
    return int(spl2[0])


def main():
    root_path = './dataset/'
    total_filename = os.path.join(root_path, 'ipc_ri_ids.csv')
    total_file = open(total_filename, mode='w', newline='')
    csv_writer = csv.writer(total_file)
    csv_writer.writerow(['id', 'stage', 'video', 'sort', 'iceDensity', 'iceArea', 'motionIntensity', 'motionDensity',
                         'motionDivergence', 'maxVelocity', 'avgVelocity'])

    train_file = open(os.path.join(root_path, 'ipc_ri_ids_train.csv'), mode='w', newline='')
    train_csv_writer = csv.writer(train_file)

    val_file = open(os.path.join(root_path, 'ipc_ri_ids_val.csv'), mode='w', newline='')
    val_csv_writer = csv.writer(val_file)

    test_file = open(os.path.join(root_path, 'ipc_ri_ids_test.csv'), mode='w', newline='')
    test_csv_writer = csv.writer(test_file)

    tid = 1
    ids = [1, 6, 10, 6, 3]
    for idx_stage, idx_video in enumerate(ids):
        for idx_v in range(idx_video):
            stage = idx_stage + 1
            video_index = idx_v + 1

            density_area_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}/density'
            motion_intensity_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}/motion_intensity'
            velocity_path = f'./dataset/RiverIceFixedCameraPointTrackVelocity/{stage}/{video_index}'

            file_list = os.listdir(density_area_path)
            file_list = sorted(file_list, key=lambda c: ms_get_int(c))

            for idx, filename in enumerate(file_list):
                if idx < 20:
                    continue
                density_area_filename = os.path.join(density_area_path, filename)
                motion_intensity_filename = os.path.join(motion_intensity_path, filename)
                velocity_filename = os.path.join(velocity_path, filename)

                density, area = np.genfromtxt(density_area_filename, delimiter=' ', dtype=float)
                motion_intensity, motion_density, motion_divergence = np.genfromtxt(motion_intensity_filename,
                                                                                    delimiter=' ', dtype=float)
                motion_intensity *= 50
                if motion_intensity > 1.0:
                    motion_density = 1.0
                point_dict_list = []
                with open(velocity_filename, 'r') as point_velocity_file:
                    content = point_velocity_file.read()
                    point_dict_list = eval(content)
                max_velocity = 0.0
                avg_velocity = 0.0
                count = 0
                for pd in point_dict_list:
                    if len(pd['points']) > 5:  # filter
                        velocity = pd['velocity']
                        max_velocity = velocity if velocity > max_velocity else max_velocity
                        avg_velocity += velocity
                        count += 1
                if count > 1:
                    avg_velocity /= count

                data = [tid, stage, video_index, idx + 1, round(density, 4), round(area, 4), round(motion_intensity, 4),
                        round(motion_density, 4), round(motion_divergence, 4), round(max_velocity, 4),
                        round(avg_velocity, 4)]
                csv_writer.writerow(data)
                tid += 1

                ra = random.random()
                if ra > 0.8:
                    test_csv_writer.writerow(data)  # 20% into test dataset
                elif ra > 0.6:
                    val_csv_writer.writerow(data)  # 20% into val dataset
                else:
                    train_csv_writer.writerow(data)  # 60% into train dataset
    total_file.close()
    train_file.close()
    test_file.close()
    val_file.close()
    print('end.')


if __name__ == '__main__':
    main()
