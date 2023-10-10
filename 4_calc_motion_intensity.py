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
import math


def main():
    max_std = 249.3489856940803
    ids = [1, 6, 10, 6, 3]
    for s, v in enumerate(ids):
        for i in range(v):
            # Through numerous experiments, the quantization value of motion intensity in Mpeg-7 and some testing
            # video is obtained.
            segmentation_path = ('./dataset/RiverIceFixedCameraSegmentation/{}/{}/pseudo_color_prediction'
                                 .format(s + 1, i + 1))
            seg_motion_path = ('./dataset/RiverIceFixedCameraSegMotion/{}/{}/'
                               .format(s + 1, i + 1))
            seg_motion_new_path = Path('./dataset/RiverIceFixedCameraSegMotionLimit/{}/{}/'
                                       .format(s + 1, i + 1))
            save_dir = Path('./dataset/RiverIceFixedCameraSegmentation/{}/{}/motion_intensity'
                            .format(s + 1, i + 1))
            save_dir.mkdir(exist_ok=True, parents=True)
            seg_motion_new_path.mkdir(exist_ok=True, parents=True)

            file_list = os.listdir(segmentation_path)
            for image_file in tqdm(file_list):
                segmentation_file = os.path.join(segmentation_path, image_file)
                segmentation = cv2.imread(segmentation_file, 0)

                seg_motion_file = os.path.join(seg_motion_path, image_file)
                seg_motion = cv2.imread(seg_motion_file, 0)
                # modify seg_motion based on segmentation
                seg_motion_new = np.where((segmentation == 75) & (seg_motion == 255), seg_motion, 0)  # 75 for ice
                seg_motion_new_file = os.path.join(seg_motion_new_path, image_file)
                # cv2.imwrite(seg_motion_new_file, seg_motion_new)

                # calc motion intensity
                motion_point_list = []
                motion_density = 0.0
                standard_deviation = 0.0
                motion_intensity = 0.0
                # motion_density
                seg_motion_new = np.where(seg_motion_new == 255, 1, seg_motion_new)
                motion_density = float(np.sum(seg_motion_new))/float(seg_motion_new.shape[0]*seg_motion_new.shape[1])
                # standard_deviation
                for h in range(seg_motion_new.shape[0]):
                    for w in range(seg_motion_new.shape[1]):
                        if seg_motion_new[h][w] == 1:
                            motion_point_list.append([[w, h]])
                if len(motion_point_list) > 0:
                    std = np.array(motion_point_list)
                    standard_deviation = np.std(std)
                    max_val = seg_motion_new.shape[1] if seg_motion_new.shape[1] > seg_motion_new.shape[0] \
                        else seg_motion_new.shape[0]
                    # max_std = math.sqrt(max_val**2)
                    standard_deviation /= max_val

                motion_intensity = motion_density * standard_deviation

                density_file = os.path.join(save_dir, image_file.split('.')[0] + '.txt')
                with open(density_file, 'w') as file:
                    file.write(str(round(motion_intensity, 4)) + ' ' + str(round(motion_density, 4)) + ' ' +
                               str(round(standard_deviation, 4)))
                    file.close()
    print('end.')


# 249.3489856940803
if __name__ == '__main__':
    main()

# PPMobileSeg  68ms/step
