import os
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
from pathlib import Path


def main():
    video_path = 'D:/Missionex/C051冰情数据/冰情视频_全天'
    video_filename = '20230331_20230331172815_20230331174011_172113.mp4'
    stage = 4
    index = 2

    output = './dataset/RiverIceFixedCameraOri/'+str(stage)+'/'+str(index)
    output_path = Path(output)
    output_path.mkdir(exist_ok=True, parents=True)

    print('Video filename: ' + video_filename)
    print('Output path: ' + output)

    video_file = os.path.join(video_path, video_filename)

    capture = cv2.VideoCapture(video_file)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    new_height = 720
    new_width = int(width * new_height / height + 0.5)
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = 1000 / fps
    print('Video prpo:  width: ' + str(width) + ' height: ' + str(height) + ' fps: ' + str(fps) +
          ' duration: ' + str(duration))

    time_ms = 0
    # video_time = 0
    capture.set(cv2.CAP_PROP_POS_MSEC, 360000)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            if video_file == 0:
                print("Opening camera is failed!!!")
            else:
                print("Video is over.")
            break
        # video_time += int(duration)
        # if video_time < 515000:
        #     continue

        new_frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
        new_filename = os.path.join(output, str(stage) + '_' + str(index) + '_' + str(time_ms) + '.jpg')
        cv2.imwrite(new_filename, new_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        time_ms += int(duration)
        if time_ms >= 60000:
            break


if __name__ == '__main__':
    main()
