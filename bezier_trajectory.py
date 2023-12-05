"""
******************************************************
** Description: 贝塞尔曲线规划足端轨迹
** Author:      黄森威
** Email:       15207847842@163.com
** date:        2022-11-11
******************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb  # SciPy特殊函数——组合计算


class BezierTrajectory(object):
    """
    生成贝塞尔轨迹
    """

    def __init__(self):
        pass

    def bernstein_polynomial(self, n, i, u):
        """
        伯恩斯坦多项式
        n: Bézier曲线的阶数
        i: 控制点
        u: Bézier曲线的参数, u∈[0,1]
        """
        return comb(n, i) * (u ** i) * (1 - u) ** (n - i)

    def bezier_trajectory(self, control_points, t, t_max):
        """
        贝塞尔曲线方程
        control_points: Bézier曲线控制点的坐标
        对贝塞尔曲线引入时间参数：令u=at
        a: 贝塞尔曲线的时间系数，a = 1/t_max
        t: 贝塞尔曲线的时间参数，t∈[0,t_max]
        t_max: 从贝塞尔曲线的起点运动到终点所需的时间（即足端走过这段轨迹所需的时间）
        """
        a = 1 / t_max
        u = a*t
        n = control_points.shape[0] - 1  # Bézier曲线的阶数 = 控制点的个数 - 1
        b = np.zeros(n + 1)  # 位置: n阶伯恩斯坦多项式
        b_dot = np.zeros(n)  # 速度: n阶伯恩斯坦多项式的一阶导数(n-1阶伯恩斯坦多项式)
        b_2dot = np.zeros(n - 1)  # 加速度: n阶伯恩斯坦多项式的二阶导数(n-2阶伯恩斯坦多项式)
        # 位置
        for i in range(n + 1):  # 伯恩斯坦多项式计算
            b[i] = self.bernstein_polynomial(n, i, u)
        bezier_position = b @ control_points  # 贝塞尔曲线计算
        # 速度
        for i in range(n):
            b_dot[i] = self.bernstein_polynomial(n - 1, i, u)
        control_points_dot = control_points[1:] - control_points[:-1]
        bezier_velocity = n * a * b_dot @ control_points_dot
        # 加速度
        for i in range(n - 1):
            b_2dot[i] = self.bernstein_polynomial(n - 2, i, u)
        control_points_2dot = control_points_dot[1:] - control_points_dot[:-1]
        bezier_acceleration = n * (n - 1) * a * a * b_2dot @ control_points_2dot
        return bezier_position, bezier_velocity, bezier_acceleration

    def swing_trajectory(self, u, t_sw, lateral_displacement=0, l=0.4, clearance_height=0.1):
        """
        phase: current trajectory phase
        t_sw: 摆动相时间，trajectory period
        l: step length
        lateral_displacement: determines how lateral the movement is
        clearance_height: foot clearance height during swing phase
        """
        h = clearance_height  # ground clearance
        step = np.array(
            [-l / 2, -l / 2 * 1.4, -l / 2 * 1.5, -l / 2 * 1.5, -l / 2 * 1.5, 0.0, 0.0, 0.0, l / 2 * 1.5, l / 2 * 1.5,
             l / 2 * 1.4, l / 2])
        x = step * np.cos(lateral_displacement)
        y = step * np.sin(lateral_displacement)
        z = np.array([0.0, 0.0, h * 0.9, h * 0.9, h * 0.9, h * 0.9, h * 0.9, h * 1.1, h * 1.1, h * 1.1, 0.0, 0.0])
        print(x)
        print(y)
        print(z)
        control_points = np.zeros((len(x), 3))
        for i in range(len(x)):
            control_points[i][0] = x[i]
            control_points[i][1] = y[i]
            control_points[i][2] = z[i]
        # print("control_points: ", control_points)
        bezier_position, bezier_velocity, bezier_acceleration = self.bezier_trajectory(control_points, u, t_sw)
        return bezier_position, bezier_velocity, bezier_acceleration

    def swing_trajectory2(self, start, end, t, t_sw, clearance_height=0.3):
        """
        摆动相轨迹
        clearance_height: ground clearance
        """
        x_scalar = np.array([start[0], start[0], start[0], end[0], end[0], end[0]])
        y_scalar = np.array([start[1], start[1], start[1], end[1], end[1], end[1]])
        z_scalar = np.array([start[2], start[2], start[2], end[2], end[2], end[2]])
        h = np.array([0, 0, clearance_height, clearance_height, 0, 0])
        h_scalar = np.array([np.zeros(x_scalar.shape[0]), np.zeros(y_scalar.shape[0]), h]).T
        control_points = np.array([x_scalar, y_scalar, z_scalar]).T + h_scalar
        # print(("control points swing: ", control_points))
        bezier_position, bezier_velocity, bezier_acceleration = self.bezier_trajectory(control_points, t, t_sw)
        return bezier_position, bezier_velocity, bezier_acceleration, control_points

    def stance_trajectory(self, start, end, u, t_st):
        """
        支撑相轨迹：like cycloid
        """
        x_scalar = np.array([end[0], end[0], end[0], start[0], start[0], start[0]])
        y_scalar = np.array([end[1], end[1], end[1], start[1], start[1], start[1]])
        z_scalar = np.array([end[2], end[2], end[2], start[2], start[2], start[2]])
        control_points = np.array([x_scalar, y_scalar, z_scalar]).T
        # print(("control points stance: ", control_points))
        bezier_position, bezier_velocity, bezier_acceleration = self.bezier_trajectory(control_points, u, t_st)
        return bezier_position, bezier_velocity, bezier_acceleration, control_points

    def stance_trajectory_sine(self, phase, lateral_displacement=0, l=0.4, penetration_depth=0.00):
        """
        功能: Calculates the step coordinates for the Sinusoidal stance period
        参数：
            phase: current trajectory phase
            l: step length
            lateral_displacement: determines how lateral the movement is
            penetration_depth: foot penetration depth during stance phase
        returns: X,Y,Z Foot Coordinates relative to unmodified body
        """
        step = l / 2 - l * phase  # moves from +l/2 to -l/2
        x = step * np.cos(lateral_displacement)
        y = step * np.sin(lateral_displacement)
        if l != 0.0:
            z = -penetration_depth * np.cos((np.pi * (x + y)) / l)
        else:
            z = 0.0
        return x, y, z


if __name__ == '__main__':
    bezier_trajectory = BezierTrajectory()
    t_sw = 0.5  # 摆动相轨迹时间
    t_st = 0.5  # 支撑相轨迹时间
    foot_start = np.array([-0.2, 0, -0.6])
    foot_end = np.array([0.2, 0, -0.6])
    t = 0
    foot_pos_sw = []
    foot_pos_st = []
    for i in range(101):
        t += 0.01
        if t <= t_sw:
            foot_pos, foot_vel, foot_acc, control_point_sw = bezier_trajectory.swing_trajectory2(foot_start, foot_end,
                                                                                                 t, t_sw)
            foot_pos_sw.append(foot_pos)
        else:
            foot_pos, foot_vel, foot_acc, control_point_st = bezier_trajectory.stance_trajectory(foot_start, foot_end,
                                                                                                 t - t_sw, t_st)
            foot_pos_st.append(foot_pos)
    # print("control_point_sw: ", control_point_sw)
    # print("control_point_st: ", control_point_st)
    # print("foot_pos_sw: ", foot_pos_sw)
    # print("foot_pos_st: ", foot_pos_st)
    control_point_sw_x = []
    control_point_sw_z = []
    for i in range(control_point_sw.shape[0]):
        control_point_sw_x.append(control_point_sw[i][0])
        control_point_sw_z.append(control_point_sw[i][2])
    control_point_st_x = []
    control_point_st_z = []
    for i in range(control_point_st.shape[0]):
        control_point_st_x.append(control_point_st[i][0])
        control_point_st_z.append(control_point_st[i][2])
    foot_pos_sw_x = []
    foot_pos_sw_z = []
    foot_pos_st_x = []
    foot_pos_st_z = []
    for i in range(len(foot_pos_sw)):
        foot_pos_sw_x.append(foot_pos_sw[i][0])
        foot_pos_sw_z.append(foot_pos_sw[i][2])
        foot_pos_st_x.append(foot_pos_st[i][0])
        foot_pos_st_z.append(foot_pos_st[i][2])
    fig = plt.figure(num=1, figsize=(16, 8), dpi=80)
    plt.scatter(control_point_sw_x, control_point_sw_z, s=24, c='r', linewidths=5, alpha=1, marker='o')
    plt.scatter(control_point_st_x, control_point_st_z, s=24, c='g', linewidths=5, alpha=1, marker='o')
    p1, = plt.plot(foot_pos_sw_x, foot_pos_sw_z, color='r', linewidth=2, linestyle='-', label="sw")
    p2, = plt.plot(foot_pos_st_x, foot_pos_st_z, color='g', linewidth=2, linestyle='-', label="st")
    plt.legend(handles=[p1, p2], labels=['摆动相轨迹', '支撑相轨迹'], loc='upper center', prop={'family': 'FangSong', 'size': 24})
    plt.title('贝塞尔曲线-足端轨迹', fontproperties='FangSong', fontsize=32, fontweight='black')
    plt.xlabel('水平位置 [m]', fontproperties='FangSong', fontsize='24')
    plt.ylabel('垂直位置 [m]', fontproperties='FangSong', fontsize='24')
    plt.xticks(fontproperties='Times New Roman', size=24)
    plt.yticks(fontproperties='Times New Roman', size=24)
    # plt.pause(2)  # 显示画布2秒后，继续运行下面程序
    plt.show()
