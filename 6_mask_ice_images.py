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
    frame_stride = 1

    stage = 3
    video_index = 4
    # prepare data
    image_path = f'./dataset/RiverIceFixedCamera/{stage}/{video_index}/'
    segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}'
    ice_mask_path = Path(f'./dataset/RiverIceFixedCameraIceMask/{stage}/{video_index}')
    ice_mask_path.mkdir(exist_ok=True, parents=True)

    file_list = os.listdir(image_path)
    file_list = sorted(file_list, key=lambda item: ms_get_int(item))
    file_list = file_list[::frame_stride]

    for idx, frame in enumerate(file_list):
        frame_filename = os.path.join(image_path, frame)
        segmentation_filename = os.path.join(segmentation_path, 'pseudo_color_prediction', frame.split('.')[0] + '.png')

        ice_mask_filename = os.path.join(ice_mask_path, frame.split('.')[0] + '.png')

        img = cv2.imread(frame_filename)
        segmentation = cv2.imread(segmentation_filename)

        # mask
        for r in range(segmentation.shape[0]):
            for c in range(segmentation.shape[1]):
                if segmentation[r, c, 2] == 128:
                    segmentation[r, c] = [0, 0, 0]
        img = cv2.addWeighted(img, 1.0, segmentation, 0.1, 0)
        cv2.imwrite(ice_mask_filename, img)

    print('end.')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()