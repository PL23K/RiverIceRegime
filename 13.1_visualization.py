import os.path

import matplotlib.pyplot as plt
import numpy as np
import os

log_path0001 = './runs/river_ice_experiment_rate_0.001_2000'
log_path001 = './runs/river_ice_experiment_rate_0.01_2000'
log_path01 = './runs/river_ice_experiment_rate_0.1_2000'
# 绘制训练集和验证集的损失变化以及验证集上的准确率变化曲线
def plot_training_loss_acc(fig_name,
                           fig_size=(10, 5),
                           sample_step=20,
                           loss_legend_loc="upper right",
                           acc_legend_loc="lower right",
                           train_color="#8E004D",
                           dev_color='#E20079',
                           fontsize='x-large',
                           train_linestyle="-",
                           dev_linestyle='--'):
    global dev_steps

    in_train_scores = np.loadtxt(fname=os.path.join(log_path0001, 'train_scores.csv'), delimiter=",", dtype=np.float32)
    in_val_scores = np.loadtxt(fname=os.path.join(log_path0001, 'val_scores.csv'), delimiter=",", dtype=np.float32)
    in_train_losses = np.loadtxt(fname=os.path.join(log_path0001, 'train_losses.csv'), delimiter=",", dtype=np.float32)
    in_val_losses = np.loadtxt(fname=os.path.join(log_path0001, 'val_losses.csv'), delimiter=",", dtype=np.float32)

    in_train_scores001 = np.loadtxt(fname=os.path.join(log_path001, 'train_scores.csv'), delimiter=",", dtype=np.float32)
    in_val_scores001 = np.loadtxt(fname=os.path.join(log_path001, 'val_scores.csv'), delimiter=",", dtype=np.float32)
    in_train_losses001 = np.loadtxt(fname=os.path.join(log_path001, 'train_losses.csv'), delimiter=",", dtype=np.float32)
    in_val_losses001 = np.loadtxt(fname=os.path.join(log_path001, 'val_losses.csv'), delimiter=",", dtype=np.float32)

    in_train_scores01 = np.loadtxt(fname=os.path.join(log_path01, 'train_scores.csv'), delimiter=",", dtype=np.float32)
    in_val_scores01 = np.loadtxt(fname=os.path.join(log_path01, 'val_scores.csv'), delimiter=",", dtype=np.float32)
    in_train_losses01 = np.loadtxt(fname=os.path.join(log_path01, 'train_losses.csv'), delimiter=",", dtype=np.float32)
    in_val_losses01 = np.loadtxt(fname=os.path.join(log_path01, 'val_losses.csv'), delimiter=",", dtype=np.float32)

    plt.figure(figsize=fig_size, dpi=500)
    plt.rcParams["font.sans-serif"] = ["Palatino Linotype", "times new roman"]

    plt.subplot(1, 2, 1)
    # train_steps = [i for i, v in enumerate(in_train_losses)]
    # train_losses = [v for i, v in enumerate(in_train_losses)]
    # plt.plot(train_steps, train_losses, color=train_color, linestyle=train_linestyle, label="Train loss")
    if len(in_val_losses) > 0:
        dev_steps = [i for i, v in enumerate(in_val_losses)]
        dev_losses = [v for i, v in enumerate(in_val_losses)]
        plt.plot(dev_steps, dev_losses, color=dev_color, linestyle=train_linestyle, label="Val loss (lr=0.001)")

        dev_steps001 = [i for i, v in enumerate(in_val_losses001)]
        dev_losses001 = [v for i, v in enumerate(in_val_losses001)]
        plt.plot(dev_steps001, dev_losses001, color='#3a00e2', linestyle=dev_linestyle, label="Val loss (lr=0.01)")

        dev_steps01 = [i for i, v in enumerate(in_val_losses01)]
        dev_losses01 = [v for i, v in enumerate(in_val_losses01)]
        plt.plot(dev_steps01, dev_losses01, color='#00e215', linestyle='dotted', label="Val loss (lr=0.1)")
    # 绘制坐标轴和图例
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.legend(loc=loss_legend_loc, fontsize=fontsize)

    # 绘制评价准确率变化曲线
    if len(in_val_scores) > 0:
        plt.subplot(1, 2, 2)
        # train_steps = [i for i, v in enumerate(in_train_scores)]
        # train_scores = [v for i, v in enumerate(in_train_scores)]
        # plt.plot(train_steps, train_scores, color=train_color, linestyle=train_linestyle, label="Train accuracy")

        dev_steps = [i for i, v in enumerate(in_val_scores)]
        val_scores = [v for i, v in enumerate(in_val_scores)]
        plt.plot(dev_steps, val_scores,
                 color=dev_color, linestyle=train_linestyle, label="Val accuracy (lr=0.001)")

        dev_steps001 = [i for i, v in enumerate(in_val_scores001)]
        val_scores001 = [v for i, v in enumerate(in_val_scores001)]
        plt.plot(dev_steps001, val_scores001,
                 color='#3a00e2', linestyle=dev_linestyle, label="Val accuracy (lr=0.01)")

        dev_steps01 = [i for i, v in enumerate(in_val_scores01)]
        val_scores01 = [v for i, v in enumerate(in_val_scores01)]
        plt.plot(dev_steps01, val_scores01,
                 color='#00e215', linestyle='dotted', label="Val accuracy (lr=0.1)")

        # 绘制坐标轴和图例
        plt.ylabel("Accuracy", fontsize=fontsize)
        plt.xlabel("Iteration", fontsize=fontsize)
        plt.legend(loc=acc_legend_loc, fontsize=fontsize)

    plt.savefig(fig_name)
    plt.show()


plot_training_loss_acc(os.path.join(log_path0001, '2000.png'))

