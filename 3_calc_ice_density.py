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
import cv2
from tqdm import tqdm
from datetime import datetime
from pathlib import Path


def main():
    ice_area_height_rate = 0.80  # ice_area_height / image_height
    ice_area_bottom_width = 20.0  # 20.0m
    ice_area_top_width = 45.5  # 45.5m
    river_width = 150.0  # 150.0m

    pixel_width_a = 0.0
    pixel_width_w = 0.0
    pixel_height = 0.0

    area_map = None

    ids = [1, 6, 10, 6, 3]
    for s, v in enumerate(ids):
        for i in range(v):
            segmentation_path = './dataset/RiverIceFixedCameraSegmentation/{}/{}/pseudo_color_prediction'.format(s + 1, i + 1)
            save_dir = Path('./dataset/RiverIceFixedCameraSegmentation/{}/{}/density'.format(s + 1, i + 1))
            save_dir.mkdir(exist_ok=True, parents=True)

            file_list = os.listdir(segmentation_path)
            for image_file in tqdm(file_list):
                segmentation_file = os.path.join(segmentation_path, image_file)
                img = cv2.imread(segmentation_file, 0)

                if image_file.split('.')[0].split('_')[2] == '0':
                    image_width = img.shape[1]
                    image_height = img.shape[0]
                    x_a = (ice_area_top_width - ice_area_bottom_width) / (float(image_height) * ice_area_height_rate)
                    pixel_width_a = float(image_height) * x_a / float(image_width) + ice_area_bottom_width / float(
                        image_width)
                    pixel_width_w = -(x_a / float(image_width))
                    pixel_height = river_width / (float(image_height) * ice_area_height_rate)
                    area_map = np.zeros([image_height, image_width], dtype=float)
                    for y in range(image_height):
                        for x in range(image_width):
                            pixel_width = pixel_width_a + pixel_width_w * y
                            area_map[y, x] = pixel_width * pixel_height
                ice = np.where(img == 75, img, 0)
                ice = np.where(ice == 75, 1, ice)
                ice_sum = float(np.sum(ice))
                water = np.where(img == 113, img, 0)
                water = np.where(water == 113, 1, water)
                water_sum = float(np.sum(water))
                density = round(ice_sum / (water_sum+ice_sum), 4)

                ice_area = np.where(ice == 1, area_map, 0.0)
                area = round(np.sum(ice_area), 4)
                density_file = os.path.join(save_dir, image_file.split('.')[0]+'.txt')
                with open(density_file, 'w') as file:
                    file.write(str(density) + ' ' + str(area))
                    file.close()
    print('end.')


if __name__ == '__main__':
    main()

# PPMobileSeg  68ms/step
