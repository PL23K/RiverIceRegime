import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage import transform

def show_cmp_img(original_img, transform_img):
    plt.rcParams["font.sans-serif"] = ["Palatino Linotype", "times new roman"]
    plt.rcParams['axes.unicode_minus'] = False
    # label字体大小
    plt.rcParams['font.size'] = 8
    # label位置靠右
    plt.rcParams['legend.loc'] = 'upper right'
    # 分辨率
    plt.rcParams['figure.dpi'] = 500
    # 大小
    plt.rcParams['figure.figsize'] = (4, 4)

    _, axes = plt.subplots(1, 2)
    axes[0].imshow(original_img)
    axes[1].imshow(transform_img)

    axes[0].set_title("original image")
    axes[1].set_title("transform image")
    plt.show()


def main():
    image_path = './dataset/RiverIceFixedCamera/3/4/3_4_30700.jpg'
    save_path = './dataset'

    image = imread(image_path)


    # Source points
    src = np.array([270, 130,                    # top left
                    0, 660,                      # bottom left
                    685, 125,                    # top right
                    960, 660]).reshape((4, 2))   # bottom right

    # Destination points
    dst = np.array([
        [400, 500],  # top left
        [400, 2000],  # bottom left
        [600, 500],  # top right
        [600, 2000],  # bottom right
    ])

    tform = transform.estimate_transform('projective', src, dst)
    print(tform)

    tf_img = transform.warp(image, tform.inverse, output_shape=(2000, 1000))
    # fig, ax = plt.subplots(figsize=(20, 20))
    # ax.imshow(tf_img)
    # _ = ax.set_title('projective transformation')

    plt.rcParams["font.sans-serif"] = ["Palatino Linotype", "times new roman"]
    plt.rcParams['axes.unicode_minus'] = False
    # label字体大小
    plt.rcParams['font.size'] = 8
    # label位置靠右
    plt.rcParams['legend.loc'] = 'upper right'
    # 分辨率
    plt.rcParams['figure.dpi'] = 500
    # 大小
    plt.rcParams['figure.figsize'] = (4, 4)

    plt.xlabel('x/m')
    plt.ylabel('y/m')

    # show_cmp_img(image, tf_img)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(tf_img)
    _ = ax.set_title('Projective Transform')

    plt.savefig(os.path.join(save_path, os.path.basename(image_path)))


    # image_path = 'd:/aa.png'
    #
    # image = imread(image_path)
    #
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.imshow(image)
    # _ = ax.set_title('Original Image')
    # plt.xlabel('x/pixel')
    # plt.ylabel('y/pixel')
    # plt.savefig(os.path.join(save_path, os.path.basename(image_path)))

if __name__ == '__main__':
    main()
