import matplotlib.pyplot as plt


# 新增绘制图像方法
def plot(runner, fig_name):
    plt.figure(figsize=(10, 5), dpi=500)
    plt.subplot(1, 2, 1)
    epochs = [i for i in range(len(runner.train_scores))]
    # 绘制训练损失变化曲线
    plt.plot(epochs, runner.train_loss, color='#8E004D', label="Train loss")
    # 绘制评价损失变化曲线
    plt.plot(epochs, runner.dev_loss, color='#E20079', linestyle='--', label="Val loss")
    # 绘制坐标轴和图例
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    # 绘制训练准确率变化曲线
    plt.plot(epochs, runner.train_scores, color='#8E004D', label="Train accuracy")
    # 绘制评价准确率变化曲线
    plt.plot(epochs, runner.dev_scores, color='#E20079', linestyle='--', label="Val accuracy")
    # 绘制坐标轴和图例
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
