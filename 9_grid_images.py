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


def generate_grid(image, grid_factor):
    image_width = image.shape[1]
    image_height = image.shape[0]
    grid_width = int(image_width / grid_factor)
    grid_height = int(image_height / grid_factor)
    roi_list = []

    for h in range(grid_factor):
        for w in range(grid_factor):
            max_height = image_height if (h + 1) * grid_height > image_height else (h + 1) * grid_height
            max_width = image_width if (w + 1) * grid_width > image_width else (w + 1) * grid_width
            roi = [w * grid_width, h * grid_height, max_width, max_height]
            roi_list += roi
    roi_list = np.array(roi_list)
    roi_list = roi_list.reshape((grid_factor, grid_factor, 4))
    return image_width, image_height, grid_width, grid_height, roi_list


def is_points_in_roi(roi_box, point_dict_list):
    """
    Parameters
    ----------
    roi_box : [x, y, x2, y2]
        roi_box points
    point_dict_list : [{'id':0, 'points':[[p1x,p1y], [p2x,p2y],...]}]
        points list
    """
    for pd in point_dict_list:
        point = pd['points'][-1]
        if roi_box[0] < point[0] < roi_box[2] and roi_box[1] < point[1] < roi_box[3]:
            return True
    return False


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def find_new_point(seg_motion_frames, roi_list, point_dict_list, grid_factor, area_threshold):
    new_point_list = []
    for h in range(grid_factor - 1):
        for w in range(grid_factor - 1):
            # 1、判断该区域是否已经有点存在 有则跳出
            if is_points_in_roi(roi_list[h, w], point_dict_list):
                continue
            # find
            contours, hierarchy = cv2.findContours(seg_motion_frames[roi_list[h, w, 1]:roi_list[h, w, 3],
                                                   roi_list[h, w, 0]:roi_list[h, w, 2]], cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            # contours = imutils.grab_contours(contours)
            contours_count = len(contours)
            if contours_count > 0:
                contours = sorted(contours, key=cnt_area, reverse=True)
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                if area > area_threshold:
                    moments = cv2.moments(cnt)
                    c_x = int(moments["m10"] / moments["m00"]) + roi_list[h, w, 0]
                    c_y = int(moments["m01"] / moments["m00"]) + roi_list[h, w, 1]
                    # cv2.drawContours(image_frames_8[0][roi_list[h, w, 1]:roi_list[h, w, 3],
                    #                  roi_list[h, w, 0]:roi_list[h, w, 2]], [cnt], -1, (0, 0, 255), 2)
                    # cv2.circle(image_frames_8[0][roi_list[h, w, 1]:roi_list[h, w, 3],
                    #            roi_list[h, w, 0]:roi_list[h, w, 2]], (c_x, c_y), 2, (0, 255, 255), -1)

                    # has point nearby?
                    half_width = int((roi_list[h, w, 2] - roi_list[h, w, 0]) / 4)
                    half_height = int((roi_list[h, w, 3] - roi_list[h, w, 1]) / 4)
                    if not is_points_in_roi([c_x - half_width, c_y - half_height, c_x + half_width, c_y + half_height],
                                            point_dict_list):
                        new_point_list += [[c_x, c_y]]
    return new_point_list


def generate_point_id(point_dict_list):
    p_id = 0
    for i in range(100000):
        find = True
        for p in point_dict_list:
            if p['id'] == i:
                find = False
                break
        if find:
            p_id = i
            break
    return p_id


def compare_visibility(point_dict):
    return point_dict['visibility']


def remove_point_dict_in_same_roi(roi_list, point_dict_list, grid_factor):
    remove_point_dict_list = []
    for h in range(grid_factor - 1):
        for w in range(grid_factor - 1):
            roi_box = roi_list[h, w]
            same_point_dict_list = []
            for pd in point_dict_list:
                point = pd['points'][-1]
                if roi_box[0] < point[0] < roi_box[2] and roi_box[1] < point[1] < roi_box[3]:
                    same_point_dict_list.append(pd)
            if len(same_point_dict_list) > 1:
                same_point_dict_list = sorted(same_point_dict_list, key=compare_visibility)
                same_point_dict_list.pop()
                remove_point_dict_list += same_point_dict_list

    print('before remove: ' + str(len(point_dict_list)))
    for pd in remove_point_dict_list:
        print('remove point id:' + str(pd['id']))
        point_dict_list.remove(pd)
    print('after remove: ' + str(len(point_dict_list)))
    return point_dict_list


def track(model, input_images, query_points, device, point_dict_list, segmentation_frame, y_tune, tracker_length):
    # track new point
    with torch.no_grad():
        input_images = input_images.to(device)
        query_points = query_points.to(device)
        trajectories, visibilities = model.to(device).forward_once(input_images, query_points)

    trajectories_cpu = trajectories.cpu().numpy()
    visibilities_cpu = visibilities.cpu().numpy()

    remove_point_dict_list = []
    for idx, pd in enumerate(point_dict_list):
        if visibilities_cpu[0, 1, idx] > 0.5:
            new_x = trajectories_cpu[0, 1, idx, 0]
            new_y = trajectories_cpu[0, 1, idx, 1]
            # fine tune
            if y_tune:
                new_y = int(round((new_y - pd['points'][-1][1]) / 2 + pd['points'][-1][1]))
            pd['points'].append([int(round(new_x, 0)), int(round(new_y, 0))])
            pd['visibility'] = visibilities_cpu[0, 1, idx]
        else:
            remove_point_dict_list.append(pd)
    for pd in remove_point_dict_list:
        point_dict_list.remove(pd)
    # remove point in same roi
    # point_dict_list = remove_point_dict_in_same_roi(roi_list, point_dict_list, grid_factor)
    # remove point which track 30 frame
    remove_point_dict_list = []
    for pd in point_dict_list:
        if len(pd['points']) > tracker_length:
            print('remove ' + str(tracker_length) + ' frame point id:' + str(pd['id']))
            remove_point_dict_list.append(pd)
    for pd in remove_point_dict_list:
        point_dict_list.remove(pd)
    # remove point which is not in ice mask
    remove_point_dict_list = []
    for pd in point_dict_list:
        point = pd['points'][-1]
        if segmentation_frame[point[1], point[0]] != 75:
            remove_point_dict_list.append(pd)
    for pd in remove_point_dict_list:
        point_dict_list.remove(pd)
    return point_dict_list


def draw_output(point_dict_list, cmap, maxdist, image_frame, image_name, point_track_video, show_point_box,
                point_box_h, point_track_path):
    # draw points
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
    # save images and video
    cv2.imwrite(os.path.join(point_track_path, image_name.split('.')[0] + '.png'), image_frame)
    if point_track_video.isOpened():
        point_track_video.write(image_frame)


def clac_velocity(point_dict_list, velocity_txt, pixel_width_a, pixel_width_w, pixel_height,
                  frame_stride, frame_duration):
    for pd in point_dict_list:
        velocity = 0.0
        if len(pd['points']) > 1:
            now_point = pd['points'][-1]
            last_point = pd['points'][-2]
            pixel_width = pixel_width_a + last_point[1]*pixel_width_w
            velocity = math.sqrt(((now_point[0] - last_point[0])*pixel_width) ** 2 +
                                 ((now_point[1] - last_point[1])*pixel_height) ** 2)
            velocity = velocity / (frame_stride * frame_duration) * 1000.0
        pd['velocity'] = round(velocity, 4)
    with open(velocity_txt, 'w') as file:
        file.write(str(point_dict_list))
        file.close()


def main():
    grid_factor = 16
    area_threshold = 50
    frame_stride = 1

    stage = 3
    video_index = 4
    # prepare data
    image_path = f'./dataset/RiverIceFixedCamera/{stage}/{video_index}/'
    seg_motion_path = f'./dataset/RiverIceFixedCameraSegMotion/{stage}/{video_index}/'
    segmentation_path = f'./dataset/RiverIceFixedCameraSegmentation/{stage}/{video_index}/pseudo_color_prediction'

    grid_path = Path(f'./dataset/RiverIceFixedCameraPointGrid/{stage}/{video_index}')
    grid_path.mkdir(exist_ok=True, parents=True)

    file_list = os.listdir(image_path)
    file_list = sorted(file_list, key=lambda c: ms_get_int(c))
    file_list = file_list[::frame_stride]

    roi_list = []
    image_width = 0
    image_height = 0
    grid_width = 0
    grid_height = 0
    for idx, frame in enumerate(file_list):
        frame_filename = os.path.join(image_path, frame)
        seg_motion_filename = os.path.join(seg_motion_path, frame.split('.')[0] + '.png')
        segmentation_filename = os.path.join(segmentation_path, frame.split('.')[0] + '.png')

        img = cv2.imread(frame_filename)
        segmentation = cv2.imread(segmentation_filename, 0)

        seg_motion = cv2.imread(seg_motion_filename, 0)
        # modify seg_motion based on segmentation
        seg_motion_new = np.where((segmentation == 75) & (seg_motion == 255), seg_motion, 0)  # 75 for ice

        if idx == 0:
            image_width, image_height, grid_width, grid_height, roi_list = generate_grid(img, grid_factor)

        point_dict_list = []
        new_point_list = []
        for h in range(grid_factor - 1):
            for w in range(grid_factor - 1):
                # 1、判断该区域是否已经有点存在 有则跳出
                if is_points_in_roi(roi_list[h, w], point_dict_list):
                    continue
                # find
                contours, hierarchy = cv2.findContours(seg_motion_new[roi_list[h, w, 1]:roi_list[h, w, 3],
                                                       roi_list[h, w, 0]:roi_list[h, w, 2]], cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                # contours = imutils.grab_contours(contours)
                contours_count = len(contours)
                if contours_count > 0:
                    contours = sorted(contours, key=cnt_area, reverse=True)
                    cnt = contours[0]
                    area = cv2.contourArea(cnt)
                    if area > area_threshold:
                        moments = cv2.moments(cnt)
                        c_x = int(moments["m10"] / moments["m00"])
                        c_y = int(moments["m01"] / moments["m00"])
                        cv2.drawContours(img[roi_list[h, w, 1]:roi_list[h, w, 3],
                                         roi_list[h, w, 0]:roi_list[h, w, 2]], [cnt], -1, (57, 87, 203), 2)
                        cv2.circle(img[roi_list[h, w, 1]:roi_list[h, w, 3],
                                   roi_list[h, w, 0]:roi_list[h, w, 2]], (c_x, c_y), 2, (35, 188, 242), -1)
        # draw grid
        for w in range(grid_factor - 1):
            cv2.line(img, [roi_list[0, w, 2], 0], [roi_list[0, w, 2], image_height], [109, 216, 122],
                     2)
        for h in range(grid_factor - 1):
            cv2.line(img, [0, roi_list[h, 0, 3]], [image_width, roi_list[h, 0, 3]], [109, 216, 122],
                     2)
        # save images and video
        cv2.imwrite(os.path.join(grid_path, frame.split('.')[0] + '.png'), img)



    # ids = [1, 6, 10, 6, 3]
    # for s, v in enumerate(ids):
    #     for i in range(v):
    # image_path = './dataset/RiverIceFixedCamera/{}/{}'.format(s + 1, i + 1)
    # save_dir = Path('./dataset/RiverIceFixedCameraSegmentation/{}/{}'.format(s+1, i+1))
    # save_dir.mkdir(exist_ok=True, parents=True)

    print('end.')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
