import matplotlib.pyplot as plt
import os
import numpy as np


# 箱线图查看异常值分布
def boxplot(features):
    plt.rcParams["font.sans-serif"] = ["Palatino Linotype", "times new roman"]


    feature_names = ['ice_concentration', 'ice_area', 'motion_intensity', 'max_velocity', 'avg_velocity']

    # 连续画几个图片
    plt.figure(figsize=(4, 4), dpi=500)
    # 子图调整
    plt.subplots_adjust(wspace=0.6)
    # 每个特征画一个箱线图
    for i in range(5):
        plt.subplot(3, 3, i+1)
        # 画箱线图
        plt.boxplot(features[:, i],
                    showmeans=True,
                    whiskerprops={"color": "#E20079", "linewidth": 0.4, 'linestyle': "--"},
                    flierprops={"markersize": 0.4},
                    meanprops={"markersize": 1})
        # 图名
        # plt.title(feature_names[i], fontdict={"size": 5}, pad=2)
        # y方向刻度
        plt.yticks(fontsize=4, rotation=90)
        plt.tick_params(pad=0.5)
        # x方向刻度
        plt.xticks([])

    # 分辨率
    plt.rcParams['figure.dpi'] = 500
    plt.savefig('./ml-vis.png', )
    plt.show()


root_path = './dataset/'
train_filename = os.path.join(root_path, 'ipc_ri_ids_train_modify.csv')
val_filename = os.path.join(root_path, 'ipc_ri_ids_val_modify.csv')
test_filename = os.path.join(root_path, 'ipc_ri_ids_test.csv')
datas_train = np.loadtxt(fname=train_filename, delimiter=",", dtype=np.float32)
datas_val = np.loadtxt(fname=val_filename, delimiter=",", dtype=np.float32)
datas_test = np.loadtxt(fname=test_filename, delimiter=",", dtype=np.float32)
ice_features_train = datas_train[:, (4, 5, 6, 9, 10)]
ice_features_val = datas_val[:, (4, 5, 6, 9, 10)]
ice_features_test = datas_test[:, (4, 5, 6, 9, 10)]
ice_features = np.concatenate((ice_features_train, ice_features_val, ice_features_test), axis=0)

# boxplot(ice_features)

# Data Preprocessing

# max_values = np.max(ice_features_train, axis=0)
# print(max_values)
# max_values = np.max(ice_features_val, axis=0)
# print(max_values)
# max_values = np.max(ice_features_test, axis=0)
# print(max_values)
# max_values = np.max(ice_features, axis=0)
# print(max_values)
#
# iv = datas_train[:, 9]
# iv[iv > 10] = 8.6385
# max_values = np.max(datas_train, axis=0)
# print(max_values)
#
# iv = datas_train[:, 10]
# iv[iv > 5] = 3.9145
# max_values = np.max(datas_train, axis=0)
# print(max_values)
# np.savetxt(os.path.join(root_path, 'ipc_ri_ids_train_modify.csv'), datas_train, delimiter=',', fmt='%10.4f')
#
# iv = datas_val[:, 10]
# iv[iv > 5] = 3.9145
# max_values = np.max(datas_val, axis=0)
# print(max_values)
# np.savetxt(os.path.join(root_path, 'ipc_ri_ids_val_modify.csv'), datas_val, delimiter=',', fmt='%10.4f')

# Data normalization

features_max = ice_features.max(axis=0)
features_min = ice_features.min(axis=0)

print('area: {:.4f} {:.4f}'.format(features_max[1], features_min[1]))
print('max velocity: {:.4f} {:.4f}'.format(features_max[3], features_min[3]))
print('min velocity: {:.4f} {:.4f}'.format(features_max[4], features_min[4]))

#print('\nprocess train set...')
# area
# w_a = features_max[1]-features_min[1]
# datas_train[:, 5] = (datas_train[:, 5] - features_min[1]) / w_a
# max_values = np.max(datas_train, axis=0)
# print(max_values)
#
# # m_v
# w_mv = features_max[3]-features_min[3]
# datas_train[:, 9] = (datas_train[:, 9] - features_min[3]) / w_mv
# max_values = np.max(datas_train, axis=0)
# print(max_values)
#
# # a_v
# w_av = features_max[4]-features_min[4]
# datas_train[:, 10] = (datas_train[:, 10] - features_min[4]) / w_av
# max_values = np.max(datas_train, axis=0)
# print(max_values)
# #np.savetxt(os.path.join(root_path, 'ipc_ri_ids_train_modify_norm.csv'), datas_train, delimiter=',', fmt='%10.4f')
#
# print('\nprocess val set...')
# # area
# datas_val[:, 5] = (datas_val[:, 5] - features_min[1]) / w_a
# max_values = np.max(datas_val, axis=0)
# print(max_values)
#
# # m_v
# datas_val[:, 9] = (datas_val[:, 9] - features_min[3]) / w_mv
# max_values = np.max(datas_val, axis=0)
# print(max_values)
#
# # a_v
# datas_val[:, 10] = (datas_val[:, 10] - features_min[4]) / w_av
# max_values = np.max(datas_val, axis=0)
# print(max_values)
# np.savetxt(os.path.join(root_path, 'ipc_ri_ids_val_modify_norm.csv'), datas_val, delimiter=',', fmt='%10.4f')
#
# print('\nprocess test set...')
# # area
# datas_test[:, 5] = (datas_test[:, 5] - features_min[1]) / w_a
# max_values = np.max(datas_test, axis=0)
# print(max_values)
#
# # m_v
# datas_test[:, 9] = (datas_test[:, 9] - features_min[3]) / w_mv
# max_values = np.max(datas_test, axis=0)
# print(max_values)
#
# # a_v
# datas_test[:, 10] = (datas_test[:, 10] - features_min[4]) / w_av
# max_values = np.max(datas_test, axis=0)
# print(max_values)
# np.savetxt(os.path.join(root_path, 'ipc_ri_ids_test_modify_norm.csv'), datas_test, delimiter=',', fmt='%10.4f')
