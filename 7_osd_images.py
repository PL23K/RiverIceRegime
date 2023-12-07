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


import cv2
import os
import numpy as np
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from matplotlib import cm


def ms_get_int(filename: str):
    spl = filename.split('_')
    spl2 = spl[len(spl) - 1].split('.')
    return int(spl2[0])


def draw_points(image_frame, point_dict_list, cmap, maxdist, show_point_box, point_box_h):
    color_map = cm.get_cmap(cmap)
    for pd in point_dict_list:
        point_last = np.array(pd['points'][-1])
        for point_index in range(len(pd['points']) - 1):
            point = pd['points'][point_index]
            val = (np.sqrt(np.sum((np.array(point) - point_last) ** 2)) / maxdist).clip(0, 1)
            color = np.array(color_map(val)[:3]) * 255  # rgb
            pts = [point]
            pts += [pd['points'][point_index + 1]]
            pts = np.array([pts])
            cv2.polylines(image_frame, pts,
                          False, color, 2, cv2.LINE_AA)
        point = pd['points'][-1]
        # color = np.array(color_map(0.01)[:3]) * 255  # rgb
        cv2.circle(image_frame, (point[0], point[1]), 2, (92, 38, 248), -1, cv2.LINE_AA)
        if show_point_box:
            cv2.rectangle(image_frame, (point[0] - point_box_h, point[1] - point_box_h),
                          (point[0] + point_box_h, point[1] + point_box_h), (0, 204, 255), 1)
    # draw grid
    # for w in range(grid_factor - 1):
    #     cv2.line(image_frames_8[0], [roi_list[0, w, 2], 0], [roi_list[0, w, 2], image_height], [0, 255, 0],
    #              1)
    # for h in range(grid_factor - 1):
    #     cv2.line(image_frames_8[0], [0, roi_list[h, 0, 3]], [image_width, roi_list[h, 0, 3]], [0, 255, 0],
    #              1)


def main():
    point_box_h = 5

    frame_stride = 1
    frame_duration = 100  # 100ms

    show_point_box = True
    cmap = 'spring'
    maxdist = 200


    ids = [1, 6, 10, 6, 3]
    for idx_stage, idx_videos in enumerate(ids):
        print('stage {}'.format(idx_stage+1))
        for idx_video in range(idx_videos):
            print('\tvideo {}'.format(idx_video + 1))
            stage = idx_stage+1
            video_index = idx_video+1
            if stage != 4 or video_index != 3:
                continue
            # prepare data
            # image_path = f'./dataset/RiverIceFixedCamera/{stage}/{video_index}/'
            segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}'
            ice_mask_path = f'./dataset/RiverIceFixedCameraIceMask/{stage}/{video_index}'
            point_track_velocity_path = f'./dataset/RiverIceFixedCameraPointTrackVelocity/{stage}/{video_index}'
            osd_path = Path(f'./dataset/RiverIceFixedCameraOSD/{stage}/{video_index}')
            osd_path.mkdir(exist_ok=True, parents=True)

            file_list = os.listdir(ice_mask_path)
            file_list = sorted(file_list, key=lambda item: ms_get_int(item))
            file_list = file_list[::frame_stride]

            times_font_18 = ImageFont.truetype("times.ttf", 18)
            times_font_20 = ImageFont.truetype("times.ttf", 20)
            times_font_bd_24 = ImageFont.truetype("timesbd.ttf", 24)

            for idx, frame in enumerate(file_list):
                # frame_filename = os.path.join(image_path, frame)
                # segmentation_filename = os.path.join(segmentation_path, 'pseudo_color_prediction', frame.split('.')[0] + '.png')
                ice_mask_filename = os.path.join(ice_mask_path, frame)
                density_area_filename = os.path.join(segmentation_path, 'density', frame.split('.')[0] + '.txt')
                motion_intensity_filename = os.path.join(segmentation_path, 'motion_intensity', frame.split('.')[0] + '.txt')
                point_velocity_filename = os.path.join(point_track_velocity_path, frame.split('.')[0] + '.txt')
                stage_filename = os.path.join(segmentation_path, 'stage', frame.split('.')[0] + '.txt')

                osd_filename = os.path.join(osd_path, frame)

                # img = cv2.imread(frame_filename)
                # segmentation = cv2.imread(segmentation_filename)
                ice_mask = cv2.imread(ice_mask_filename)
                density = 0.0
                area = 0.0
                density, area = np.genfromtxt(density_area_filename, delimiter=' ', dtype=float)
                motion_intensity = 0.0
                motion_density = 0.0
                motion_divergence = 0.0
                motion_intensity, motion_density, motion_divergence = np.genfromtxt(motion_intensity_filename, delimiter=' ',
                                                                                    dtype=float)
                motion_intensity *= 50
                if motion_intensity > 1.0:
                    motion_intensity = 1.0

                ice_stage = -1
                ice_stage, _ = np.genfromtxt(stage_filename, delimiter=' ', dtype=float)

                point_dict_list = []
                with open(point_velocity_filename, 'r') as point_velocity_file:
                    content = point_velocity_file.read()
                    point_dict_list = eval(content)

                # 1 mask
                # for r in range(segmentation.shape[0]):
                #     for c in range(segmentation.shape[1]):
                #         if segmentation[r, c, 2] == 128:
                #             segmentation[r, c] = [0, 0, 0]
                # img = cv2.addWeighted(img, 1.0, segmentation, 0.1, 0)

                # 2 draw points
                draw_points(ice_mask, point_dict_list, cmap, maxdist, show_point_box, point_box_h)

                # 2 draw prediction
                color_map = [(181, 129, 62), (54, 83, 191), (54, 83, 191), (53, 130, 53), (168, 168, 168)]
                pts = np.array([[10, 6], [275, 6], [275, 46], [10, 46]], dtype=np.int32)
                cv2.fillPoly(ice_mask, [pts], (0, 187, 255))
                pts = np.array([[135, 9], [265, 9], [265, 43], [135, 43]], dtype=np.int32)
                cv2.fillPoly(ice_mask, [pts], color_map[int(ice_stage)])

                # write text
                max_velocity = 0.0
                avg_velocity = 0.0
                for pd in point_dict_list:
                    velocity = pd['velocity']
                    max_velocity = velocity if velocity > max_velocity else max_velocity
                    avg_velocity += velocity
                if len(point_dict_list) > 1:
                    avg_velocity /= len(point_dict_list)
                # Convert the image to RGB (OpenCV uses BGR)
                ice_mask_rgb = cv2.cvtColor(ice_mask, cv2.COLOR_BGR2RGB)
                pil_ice_mask = Image.fromarray(ice_mask_rgb)
                draw_ice_mask = ImageDraw.Draw(pil_ice_mask)

                ice_stage_string = 'Unknown'
                if ice_stage == 0:
                    ice_stage_string = 'Ice Frozen'
                elif ice_stage == 1:
                    ice_stage_string = 'Breakup Begin'
                elif ice_stage == 2:
                    ice_stage_string = 'Ice Drifting'
                elif ice_stage == 3:
                    ice_stage_string = 'Breakup End'
                elif ice_stage == 4:
                    ice_stage_string = 'Ice Free'
                draw_ice_mask.text((15, 12), 'Prediction:', font=times_font_bd_24, fill=(0, 0, 0))
                draw_ice_mask.text((140, 14), ice_stage_string, font=times_font_20, fill=(255, 255, 255))

                draw_ice_mask.text((15, 50), 'Ice velocity: max {:10.4f} m/s, avg {:10.4f} m/s'.format(max_velocity, avg_velocity),
                                   font=times_font_18, fill=(255, 221, 85), stroke_width=1, stroke_fill=(80, 80, 80))
                draw_ice_mask.text((15, 70), 'Ice area: {:10.4f} m^2'.format(area), font=times_font_18, fill=(255, 204, 34),
                                   stroke_width=1, stroke_fill=(80, 80, 80))
                draw_ice_mask.text((15, 90), 'Ice concentration: {:10.4f}'.format(density), font=times_font_18, fill=(255, 187, 0),
                                   stroke_width=1, stroke_fill=(80, 80, 80))
                draw_ice_mask.text((15, 110), 'Motion intensity: {:11.4f}'.format(motion_intensity), font=times_font_18,
                                   fill=(221, 170, 0), stroke_width=1, stroke_fill=(80, 80, 80))

                ice_mask = cv2.cvtColor(np.array(pil_ice_mask), cv2.COLOR_RGB2BGR)
                cv2.imwrite(osd_filename, ice_mask)
                # cv2.imshow('Display', ice_mask)
                # cv2.waitKey(25)

    print('end.')

if __name__ == '__main__':
    main()
