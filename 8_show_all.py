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

from PIL import ImageFont, ImageDraw, Image



def ms_get_int(filename: str):
    spl = filename.split('_')
    spl2 = spl[len(spl) - 1].split('.')
    return int(spl2[0])


def ms_get_int_2(filename: str):
    spl = filename.split('.')
    spl2 = spl[0].split('_')
    spl3 = spl2[-2] + spl2[-1]
    return int(spl3)


def plan_a():
    frame_stride = 1
    point_track_fully_stride = 8

    stage = 4
    video_index = 3
    # prepare data
    image_path = f'./dataset/RiverIceFixedCamera/{stage}/{video_index}/'
    segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}/added_prediction'
    motion_path = f'./dataset/RiverIceFixedCameraMotion/{stage}/{video_index}'
    seg_motion_path = f'./dataset/RiverIceFixedCameraSegMotion/{stage}/{video_index}'
    seg_motion_revision_path = f'./dataset/RiverIceFixedCameraSegMotionRevision/{stage}/{video_index}'
    point_track_fully_path = f'./dataset/RiverIceFixedCameraTrack8x_2/{stage}/{video_index}'
    point_grid_path = f'./dataset/RiverIceFixedCameraPointGrid/{stage}/{video_index}'
    osd_path = f'./dataset/RiverIceFixedCameraOSD/{stage}/{video_index}'

    merge_path = Path(f'./dataset/RiverIceFixedCameraMergeA/{stage}/{video_index}')
    merge_path.mkdir(exist_ok=True, parents=True)

    file_list = os.listdir(image_path)
    file_list = sorted(file_list, key=lambda item: ms_get_int(item))
    file_list = file_list[::frame_stride]

    point_track_fully_list = os.listdir(point_track_fully_path)
    point_track_fully_list = sorted(point_track_fully_list, key=lambda item: ms_get_int_2(item))

    times_font_18 = ImageFont.truetype("times.ttf", 18)
    times_font_20 = ImageFont.truetype("times.ttf", 20)
    times_font_bd_24 = ImageFont.truetype("timesbd.ttf", 24)

    point_track_fully_idx = 0
    small_width = 290
    small_height = 200
    for idx, frame in enumerate(file_list):
        frame_filename = os.path.join(image_path, frame)
        segmentation_filename = os.path.join(segmentation_path, frame.split('.')[0] + '.jpg')
        seg_motion_filename = os.path.join(seg_motion_path, frame.split('.')[0] + '.png')
        seg_motion_revision_filename = os.path.join(seg_motion_revision_path, frame.split('.')[0] + '.png')
        point_track_fully_idx = int(idx / 8) if int(idx / 8) < len(point_track_fully_list) else len(
            point_track_fully_list) - 1
        point_track_fully_filename = os.path.join(point_track_fully_path, point_track_fully_list[point_track_fully_idx])
        point_grid_filename = os.path.join(point_grid_path, frame.split('.')[0] + '.png')
        osd_filename = os.path.join(osd_path, frame.split('.')[0] + '.png')

        merge_filename = os.path.join(merge_path, frame.split('.')[0] + '.png')

        merge_image = np.zeros((720, 1600, 3), dtype=np.uint8)

        # 6 point track fully # draw first
        point_track_fully = cv2.imread(point_track_fully_filename)
        point_track_fully = cv2.resize(point_track_fully, (small_width + 40, small_height + 50),
                                       interpolation=cv2.INTER_LINEAR)
        merge_image[5:255, 300:630, :] = point_track_fully[:, :, :]
        cv2.rectangle(merge_image, (320, 30), (610, 230), (200, 200, 200), 1)

        # 1 img
        img = cv2.imread(frame_filename)
        img = cv2.resize(img, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[30:230, 10:300, :] = img[:, :, :]
        cv2.rectangle(merge_image, (10, 30), (300, 230), (200, 200, 200), 1)

        # 2 seg
        segmentation = cv2.imread(segmentation_filename)
        segmentation = cv2.resize(segmentation, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[250:450, 10:300, :] = segmentation[:, :, :]
        cv2.rectangle(merge_image, (10, 250), (300, 450), (200, 200, 200), 1)

        # 3 seg motion
        seg_motion = cv2.imread(seg_motion_filename)
        seg_motion = cv2.resize(seg_motion, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[470:670, 10:300, :] = seg_motion[:, :, :]
        cv2.rectangle(merge_image, (10, 470), (300, 670), (200, 200, 200), 1)

        # 4 seg motion revision
        seg_motion_revision = cv2.imread(seg_motion_revision_filename)
        seg_motion_revision = cv2.resize(seg_motion_revision, (small_width, small_height),
                                         interpolation=cv2.INTER_LINEAR)
        merge_image[470:670, 320:610, :] = seg_motion_revision[:, :, :]
        cv2.rectangle(merge_image, (320, 470), (610, 670), (200, 200, 200), 1)

        # 5 point grid selected
        point_selected = cv2.imread(point_grid_filename)
        point_selected = cv2.resize(point_selected, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[250:450, 320:610, :] = point_selected[:, :, :]
        cv2.rectangle(merge_image, (320, 250), (610, 450), (200, 200, 200), 1)



        # 7 osd image
        osd_image = cv2.imread(osd_filename)
        merge_image[30:690, 630:1590, :] = osd_image[:, :, :]
        cv2.rectangle(merge_image, (630, 30), (1590, 690), (220, 220, 220), 2)

        # write text
        # Convert the image to RGB (OpenCV uses BGR)
        merge_image_rgb = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)
        pil_merge_image = Image.fromarray(merge_image_rgb)
        draw_merge_image = ImageDraw.Draw(pil_merge_image)

        draw_merge_image.text((15, 12), 'Prediction:', font=times_font_bd_24, fill=(255, 255, 255))
        draw_merge_image.text((140, 14), 'Ice Drifting', font=times_font_20, fill=(255, 255, 255))

        merge_image = cv2.cvtColor(np.array(pil_merge_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(merge_filename, merge_image)
        cv2.imshow('Display', merge_image)
        cv2.waitKey(25)


def plan_b():
    frame_stride = 1
    point_track_fully_stride = 8

    stage = 4
    video_index = 3
    # prepare data
    image_path = f'./dataset/RiverIceFixedCamera/{stage}/{video_index}/'
    segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}/added_prediction'
    motion_path = f'./dataset/RiverIceFixedCameraMotion/{stage}/{video_index}'
    seg_motion_path = f'./dataset/RiverIceFixedCameraSegMotion/{stage}/{video_index}'
    seg_motion_revision_path = f'./dataset/RiverIceFixedCameraSegMotionRevision/{stage}/{video_index}'
    point_track_fully_path = f'./dataset/RiverIceFixedCameraTrack8x_2/{stage}/{video_index}'
    point_grid_path = f'./dataset/RiverIceFixedCameraPointGrid/{stage}/{video_index}'
    osd_path = f'./dataset/RiverIceFixedCameraOSD/{stage}/{video_index}'

    merge_path = Path(f'./dataset/RiverIceFixedCameraMergeB/{stage}/{video_index}')
    merge_path.mkdir(exist_ok=True, parents=True)

    file_list = os.listdir(image_path)
    file_list = sorted(file_list, key=lambda item: ms_get_int(item))
    file_list = file_list[::frame_stride]

    point_track_fully_list = os.listdir(point_track_fully_path)
    point_track_fully_list = sorted(point_track_fully_list, key=lambda item: ms_get_int_2(item))

    times_font_14 = ImageFont.truetype("times.ttf", 16)
    times_font_20 = ImageFont.truetype("times.ttf", 20)
    times_font_bd_24 = ImageFont.truetype("timesbd.ttf", 24)

    point_track_fully_idx = 0
    small_width = 290
    small_height = 200
    down_arrow_image = cv2.imread('./dataset/down.png')
    up_arrow_image = cv2.imread('./dataset/up.png')
    left_arrow_image = cv2.imread('./dataset/left.png')
    right_arrow_image = cv2.imread('./dataset/right.png')
    for idx, frame in enumerate(file_list):
        frame_filename = os.path.join(image_path, frame)
        segmentation_filename = os.path.join(segmentation_path, frame.split('.')[0] + '.jpg')
        seg_motion_filename = os.path.join(seg_motion_path, frame.split('.')[0] + '.png')
        seg_motion_revision_filename = os.path.join(seg_motion_revision_path, frame.split('.')[0] + '.png')
        point_track_fully_idx = int(idx / 8) if int(idx / 8) < len(point_track_fully_list) else len(
            point_track_fully_list) - 1
        point_track_fully_filename = os.path.join(point_track_fully_path, point_track_fully_list[point_track_fully_idx])
        point_grid_filename = os.path.join(point_grid_path, frame.split('.')[0] + '.png')
        osd_filename = os.path.join(osd_path, frame.split('.')[0] + '.png')

        merge_filename = os.path.join(merge_path, frame.split('.')[0] + '.png')

        merge_image = np.zeros((720, 1600, 3), dtype=np.uint8)

        # 6 point track fully # draw first7
        point_track_fully = cv2.imread(point_track_fully_filename)
        point_track_fully = cv2.resize(point_track_fully, (small_width + 40, small_height + 50),
                                       interpolation=cv2.INTER_LINEAR)
        merge_image[5:255, 970:1300, :] = point_track_fully[:, :, :] / 1.1
        cv2.rectangle(merge_image, (990, 30), (1280, 230), (200, 200, 200), 1)

        # 7 osd image
        osd_image = cv2.imread(osd_filename)
        merge_image[30:690, 10:970, :] = osd_image[:, :, :]
        merge_image[125:136, 971:989] = left_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (10, 30), (970, 690), (220, 220, 220), 2)

        # 1 img
        img = cv2.imread(frame_filename)
        img = cv2.resize(img, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[30:230, 1300:1590, :] = img[:, :, :] / 1.2
        merge_image[231:249, 1570:1581] = down_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (1300, 30), (1590, 230), (200, 200, 200), 1)

        # 2 seg
        segmentation = cv2.imread(segmentation_filename)
        segmentation = cv2.resize(segmentation, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[250:450, 1300:1590, :] = segmentation[:, :, :] / 1.2
        merge_image[451:469, 1570:1581] = down_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (1300, 250), (1590, 450), (200, 200, 200), 1)

        # 3 seg motion
        seg_motion = cv2.imread(seg_motion_filename)
        seg_motion = cv2.resize(seg_motion, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[470:670, 1300:1590, :] = seg_motion[:, :, :] / 1.2
        merge_image[565:576, 1281:1299] = left_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (1300, 470), (1590, 670), (200, 200, 200), 1)

        # 4 seg motion revision
        seg_motion_revision = cv2.imread(seg_motion_revision_filename)
        seg_motion_revision = cv2.resize(seg_motion_revision, (small_width, small_height),
                                         interpolation=cv2.INTER_LINEAR)
        merge_image[470:670, 990:1280, :] = seg_motion_revision[:, :, :] / 1.2
        merge_image[451:469, 1260:1271] = up_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (990, 470), (1280, 670), (200, 200, 200), 1)

        # 5 point grid selected
        point_selected = cv2.imread(point_grid_filename)
        point_selected = cv2.resize(point_selected, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[250:450, 990:1280, :] = point_selected[:, :, :] / 1.2
        merge_image[231:249, 1260:1271] = up_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (990, 250), (1280, 450), (200, 200, 200), 1)

        # write text
        # Convert the image to RGB (OpenCV uses BGR)
        merge_image_rgb = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)
        pil_merge_image = Image.fromarray(merge_image_rgb)
        draw_merge_image = ImageDraw.Draw(pil_merge_image)

        color = (255, 255, 255)
        draw_merge_image.text((1310, 230), '(a) input video', font=times_font_14, fill=color)
        draw_merge_image.text((1310, 450), '(b) semantic segmentation', font=times_font_14, fill=color)
        draw_merge_image.text((1310, 670), '(c) segmentation motion', font=times_font_14, fill=color)
        draw_merge_image.text((1000, 670), '(d) motion revision', font=times_font_14, fill=color)
        draw_merge_image.text((1000, 450), '(e) points selection', font=times_font_14, fill=color)
        draw_merge_image.text((1000, 230), '(f) points tracking', font=times_font_14, fill=color)
        draw_merge_image.text((10, 694), '(g) River Ice Regime Recognition: Surface Ice Concentration, Area, and Velocity', font=times_font_14, fill=color)

        merge_image = cv2.cvtColor(np.array(pil_merge_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(merge_filename, merge_image)
        # cv2.imshow('Display', merge_image)
        # cv2.waitKey(25)


def plan_c():
    frame_stride = 1
    point_track_fully_stride = 8

    stage = 2
    video_index = 1
    # prepare data
    image_path = f'./dataset/RiverIceFixedCamera/{stage}/{video_index}/'
    segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}/added_prediction'
    motion_path = f'./dataset/RiverIceFixedCameraMotion/{stage}/{video_index}'
    seg_motion_path = f'./dataset/RiverIceFixedCameraSegMotion/{stage}/{video_index}'
    seg_motion_revision_path = f'./dataset/RiverIceFixedCameraSegMotionRevision/{stage}/{video_index}'
    point_track_fully_path = f'./dataset/RiverIceFixedCameraTrack8x_2/{stage}/{video_index}'
    point_grid_path = f'./dataset/RiverIceFixedCameraPointGrid/{stage}/{video_index}'
    osd_path = f'./dataset/RiverIceFixedCameraOSD/{stage}/{video_index}'

    merge_path = Path(f'./dataset/RiverIceFixedCameraMergeC/{stage}/{video_index}')
    merge_path.mkdir(exist_ok=True, parents=True)

    file_list = os.listdir(image_path)
    file_list = sorted(file_list, key=lambda item: ms_get_int(item))
    file_list = file_list[::frame_stride]

    point_track_fully_list = os.listdir(point_track_fully_path)
    point_track_fully_list = sorted(point_track_fully_list, key=lambda item: ms_get_int_2(item))

    times_font_14 = ImageFont.truetype("times.ttf", 14)
    times_font_20 = ImageFont.truetype("times.ttf", 20)
    times_font_bd_24 = ImageFont.truetype("timesbd.ttf", 24)

    point_track_fully_idx = 0
    small_width = 290
    small_height = 200

    down_arrow_image = cv2.imread('./dataset/down.png')
    up_arrow_image = cv2.imread('./dataset/up.png')
    left_arrow_image = cv2.imread('./dataset/left.png')
    right_arrow_image = cv2.imread('./dataset/right.png')
    for idx, frame in enumerate(file_list):
        frame_filename = os.path.join(image_path, frame)
        segmentation_filename = os.path.join(segmentation_path, frame.split('.')[0] + '.jpg')
        seg_motion_filename = os.path.join(seg_motion_path, frame.split('.')[0] + '.png')
        seg_motion_revision_filename = os.path.join(seg_motion_revision_path, frame.split('.')[0] + '.png')
        point_track_fully_idx = int(idx / 8) if int(idx / 8) < len(point_track_fully_list) else len(
            point_track_fully_list) - 1
        point_track_fully_filename = os.path.join(point_track_fully_path, point_track_fully_list[point_track_fully_idx])
        point_grid_filename = os.path.join(point_grid_path, frame.split('.')[0] + '.png')
        osd_filename = os.path.join(osd_path, frame.split('.')[0] + '.png')

        merge_filename = os.path.join(merge_path, frame.split('.')[0] + '.png')

        merge_image = np.zeros((720, 1600, 3), dtype=np.uint8)

        # 6 point track fully # draw first
        point_track_fully = cv2.imread(point_track_fully_filename)
        point_track_fully = cv2.resize(point_track_fully, (small_width + 40, small_height + 50),
                                       interpolation=cv2.INTER_LINEAR)
        merge_image[5:255, 1280:1600, :] = point_track_fully[:, :small_width + 30, :] / 1.2
        merge_image[231:249, 1570:1581] = up_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (1300, 30), (1590, 230), (200, 200, 200), 1)

        # 1 img
        img = cv2.imread(frame_filename)
        img = cv2.resize(img, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[30:230, 10:300, :] = img[:, :, :] / 1.2
        merge_image[231:249, 280:291] = down_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (10, 30), (300, 230), (200, 200, 200), 1)

        # 2 seg
        segmentation = cv2.imread(segmentation_filename)
        segmentation = cv2.resize(segmentation, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[250:450, 10:300, :] = segmentation[:, :, :] / 1.2
        merge_image[451:469, 280:291] = down_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (10, 250), (300, 450), (200, 200, 200), 1)

        # 3 seg motion
        seg_motion = cv2.imread(seg_motion_filename)
        seg_motion = cv2.resize(seg_motion, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[470:670, 10:300, :] = seg_motion[:, :, :] / 1.2
        merge_image[671:689, 280:291] = down_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (10, 470), (300, 670), (200, 200, 200), 1)

        # 4 seg motion revision
        seg_motion_revision = cv2.imread(seg_motion_revision_filename)
        seg_motion_revision = cv2.resize(seg_motion_revision, (small_width, small_height),
                                         interpolation=cv2.INTER_LINEAR)
        merge_image[470:670, 1300:1590, :] = seg_motion_revision[:, :, :] / 1.2
        merge_image[671:689, 1570:1581] = up_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (1300, 470), (1590, 670), (200, 200, 200), 1)

        # 5 point grid selected
        point_selected = cv2.imread(point_grid_filename)
        point_selected = cv2.resize(point_selected, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        merge_image[250:450, 1300:1590, :] = point_selected[:, :, :] / 1.2
        merge_image[451:469, 1570:1581] = up_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (1300, 250), (1590, 450), (200, 200, 200), 1)



        # 7 osd image
        osd_image = cv2.imread(osd_filename)
        merge_image[30:690, 320:1280, :] = osd_image[:, :, :]
        merge_image[125:136, 1281:1299] = left_arrow_image[:, :, :] / 1.2
        cv2.rectangle(merge_image, (320, 30), (1280, 690), (220, 220, 220), 2)

        # write text
        # Convert the image to RGB (OpenCV uses BGR)
        merge_image_rgb = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)
        pil_merge_image = Image.fromarray(merge_image_rgb)
        draw_merge_image = ImageDraw.Draw(pil_merge_image)

        color = (212, 212, 212)
        draw_merge_image.text((20, 232), '(a) input video', font=times_font_14, fill=color)
        draw_merge_image.text((20, 452), '(b) semantic segmentation', font=times_font_14, fill=color)
        draw_merge_image.text((20, 672), '(c) segmentation motion', font=times_font_14, fill=color)
        draw_merge_image.text((1310, 672), '(d) motion revision', font=times_font_14, fill=color)
        draw_merge_image.text((1310, 452), '(e) points selection', font=times_font_14, fill=color)
        draw_merge_image.text((1310, 232), '(f) points tracking', font=times_font_14, fill=color)

        merge_image = cv2.cvtColor(np.array(pil_merge_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(merge_filename, merge_image)
        cv2.imshow('Display', merge_image)
        cv2.waitKey(25)


def main():
    plan_b()
    print('end.')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
