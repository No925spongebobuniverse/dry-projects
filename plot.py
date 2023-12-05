"""
******************************************************
** Description: 数据后处理、画图函数
** Author:      黄森威
** Email:       15207847842@163.com
** date:        2022-11-11
******************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

sim_time_all = np.load("sim_time_all.npy")  # 仿真时间
base_position_w_all = np.load("base_position_w_all.npy")  # 躯干位置
base_orientation_w_all = np.rad2deg(np.load("base_orientation_w_all.npy"))  # 躯干姿态
base_linear_vel_all = np.load("base_linear_vel_all.npy")  # 机器人速度
torques_all = np.load("torques_all.npy")    # 关节力矩
joints_angle_all = np.rad2deg(np.load("joints_angle_all.npy"))  # 关节位置
foot_position_w_all = np.load("foot_position_w_all.npy")  # 世界系下 足端位置
foot_position_H_all = np.load("foot_position_H_all.npy")  # H系下 足端位置
foot_position_H_desired_all = np.load("foot_position_H_desired_all.npy")  # H系下 规划的 足端位置
foot_velocity_B_all = np.load("foot_velocity_B_all.npy")  # B系下 估计的 足端速度
foot_velocity_B_sensor_all = np.load("foot_velocity_B_sensor_all.npy")  # B系下 测量的 足端速度

# ------------------- 躯干位置 ---------------------
fig1 = plt.figure(num=1, figsize=(16, 8), dpi=80)
p1, = plt.plot(sim_time_all, base_position_w_all[:, 0], color='r', linewidth=2, linestyle='-', label="X")
p2, = plt.plot(sim_time_all, base_position_w_all[:, 1], color='g', linewidth=2, linestyle='--', label='Y')
p3, = plt.plot(sim_time_all, base_position_w_all[:, 2], color='b', linewidth=2, linestyle=':', label='Z')
plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
plt.ylabel('位置 [m]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p1, p2, p3], labels=['X', 'Y', 'Z'], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
plt.title('躯干位置曲线', fontproperties='FangSong', fontsize=24, fontweight='black')
plt.savefig(fname='躯干位置'+'.svg', bbox_inches='tight')

# ---------------- 躯干位置 ------------------------
fig2 = plt.figure(num=2, figsize=(16, 10), dpi=80)
p4, = plt.plot(base_position_w_all[:, 0], base_position_w_all[:, 1], color='r', linewidth=2, linestyle='-', label="sw")
plt.xlabel('X 位置 [m]', fontproperties='FangSong', fontsize=20)
plt.ylabel('Y 位置 [m]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p4], labels=['XY位置'], loc="upper right", prop={'family': 'FangSong', 'size': 20})
plt.title('躯干水平位置', fontproperties='FangSong', fontsize=24, fontweight='black')
plt.savefig(fname='躯干水平位置'+'.svg', bbox_inches='tight')

# ---------------- 躯干姿态 ------------------------
fig3 = plt.figure(num=3, figsize=(16, 12), dpi=80)
p1, = plt.plot(sim_time_all, base_orientation_w_all[:, 0], color='r', linewidth=2, linestyle='-', label="roll")
p2, = plt.plot(sim_time_all, base_orientation_w_all[:, 1], color='g', linewidth=2, linestyle='--', label='pitch')
p3, = plt.plot(sim_time_all, base_orientation_w_all[:, 2], color='b', linewidth=2, linestyle=':', label='yaw')
plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
plt.ylabel('姿态 [°]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p1, p2, p3], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
plt.title('躯干姿态曲线', fontproperties='FangSong', fontsize=24, fontweight='black')
plt.savefig(fname='躯干姿态'+'.svg', bbox_inches='tight')

# ----------------- 机器人速度 -----------------------
fig4 = plt.figure(num=4, figsize=(16, 6), dpi=80)
p1, = plt.plot(sim_time_all, base_linear_vel_all[:, 0], color='r', linewidth=2, linestyle='-', label=r"$v_x$")
p2, = plt.plot(sim_time_all, base_linear_vel_all[:, 1], color='g', linewidth=2, linestyle='--', label=r"$v_y$")
p3, = plt.plot(sim_time_all, base_linear_vel_all[:, 2], color='b', linewidth=2, linestyle=':', label=r"$v_z$")
plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
plt.ylabel('速度 [m/s]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p1, p2, p3], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
plt.title('机器人速度曲线', fontproperties='FangSong', fontsize=24, fontweight='black')
plt.savefig(fname='机器人速度'+'.svg', bbox_inches='tight')

# ----------------- 关节力矩 -----------------------
fig5 = plt.figure(num=5, figsize=(16, 6), dpi=80)
p1, = plt.plot(sim_time_all[0:1000], torques_all[0:1000, 0], color='r', linewidth=2, linestyle='-', label="roll")
p2, = plt.plot(sim_time_all[0:1000], torques_all[0:1000, 1], color='g', linewidth=2, linestyle='--', label='pitch')
p3, = plt.plot(sim_time_all[0:1000], torques_all[0:1000, 2], color='b', linewidth=2, linestyle=':', label='knee')
plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
plt.ylabel('力矩 [Nm]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p1, p2, p3], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
plt.title('关节力矩曲线', fontproperties='FangSong', fontsize=24, fontweight='black')
plt.savefig(fname='关节力矩'+'.svg', bbox_inches='tight')

# ---------------- 关节位置 ------------------------
fig6 = plt.figure(num=6, figsize=(16, 6), dpi=80)
p1, = plt.plot(sim_time_all[0:1000], joints_angle_all[0:1000, 0], color='r', linewidth=2, linestyle='-', label="roll")
p2, = plt.plot(sim_time_all[0:1000], joints_angle_all[0:1000, 1], color='g', linewidth=2, linestyle='--', label='pitch')
p3, = plt.plot(sim_time_all[0:1000], joints_angle_all[0:1000, 2], color='b', linewidth=2, linestyle=':', label='knee')
plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
plt.ylabel('位置 [°]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p1, p2, p3], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
plt.title('关节位置曲线', fontproperties='FangSong', fontsize=24, fontweight='black')
plt.savefig(fname='关节位置'+'.svg', bbox_inches='tight')

# ------------------- 足端位置(实际) 世界系下 ---------------------
fig7 = plt.figure(num=7, figsize=(16, 8), dpi=80)
p1, = plt.plot(sim_time_all, foot_position_w_all[:, 0], color='r', linewidth=2, linestyle='-', label="X")
p2, = plt.plot(sim_time_all, foot_position_w_all[:, 1], color='g', linewidth=2, linestyle='--', label='Y')
p3, = plt.plot(sim_time_all, foot_position_w_all[:, 2], color='b', linewidth=2, linestyle=':', label='Z')
plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
plt.ylabel('位置 [m]', fontproperties='FangSong', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(handles=[p1, p2, p3], labels=['X', 'Y', 'Z'], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
plt.title('足端位置曲线', fontproperties='FangSong', fontsize=24, fontweight='black')
# plt.savefig(fname='足端位置'+'.svg', bbox_inches='tight')

# 3D线图
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(xs=foot_position_w_all[:, 0], ys=foot_position_w_all[:, 1], zs=foot_position_w_all[:, 2], c="r")

# # ---------------- 足端位置(期望+实际) ------------------------
# fig8 = plt.figure(num=8, figsize=(16, 10), dpi=80)
# name_des = ["$x_{desire}$", '$y_{desire}$', '$z_{desire}$']
# name_act = ["$x_{actual}$", '$y_{actual}$', '$z_{actual}$']
# for i in range(3):
#     plt.subplot(3, 1, i + 1)
#     p1, = plt.plot(sim_time_all, foot_position_H_all[:, i], color='r', linewidth=2, linestyle='-', label=name_des[i])
#     p2, = plt.plot(sim_time_all, foot_position_H_desired_all[:, i], color='g', linewidth=2, linestyle=':', label=name_act[i])
#     plt.legend(handles=[p1, p2], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
#     if i == 1:
#         plt.ylabel('位置 [m/s]', fontproperties='FangSong', fontsize=20)
# plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
# plt.xticks(fontproperties='Times New Roman', size=20)
# plt.yticks(fontproperties='Times New Roman', size=20)
# plt.suptitle('足端位置曲线', fontproperties='FangSong', fontsize=24, fontweight='bold')
# fig8.tight_layout()
# # plt.savefig(fname='足端位置'+'.svg', bbox_inches='tight')
#
# # ---------------- 足端速度 ------------------------
# fig10 = plt.figure(num=10, figsize=(16, 10), dpi=80)
# name = ["$v_x$", '$v_y$', '$v_z$']
# name_s = ["$v_{xs}$", '$v_{ys}$', '$v_{zs}$']
# for i in range(3):
#     plt.subplot(3, 1, i + 1)
#     p1, = plt.plot(sim_time_all[0:500], foot_velocity_B_all[0:500, i], color='r', linewidth=2, linestyle='-', label=name[i])
#     p2, = plt.plot(sim_time_all[0:500], foot_velocity_B_sensor_all[0:500, i], color='g', linewidth=2, linestyle=':', label=name_s[i])
#     plt.legend(handles=[p1, p2], loc="upper right", prop={'family': 'Times New Roman', 'size': 20})
#     if i == 1:
#         plt.ylabel('速度 [m/s]', fontproperties='FangSong', fontsize=20)
# plt.xlabel('时间 [s]', fontproperties='FangSong', fontsize=20)
# plt.xticks(fontproperties='Times New Roman', size=20)
# plt.yticks(fontproperties='Times New Roman', size=20)
# plt.suptitle('足端速度曲线', fontproperties='FangSong', fontsize=24, fontweight='bold')
# fig10.tight_layout()
# plt.savefig(fname='足端速度'+'.svg', bbox_inches='tight')
#
# # plt.pause(2)  # 显示画布2秒后，继续运行下面程序
plt.show()