"""
******************************************************
** Description: 单腿机器人控制
** Author:      黄森威，董冉祎
** Email:       15207847842@163.com
** date:        2022-11-11
******************************************************
"""
import time
import numpy as np
import pybullet as p
import pybullet_data as pd
import matplotlib.pyplot as plt
import kinematics
from bezier_trajectory import BezierTrajectory
from pybullet_interface import BulletInterface


class OneLeg(object):
    """
    单腿机器人
    """
    def __init__(self):
        # *************** pybullet设置 *************** #
        use_gui = True
        if use_gui:  # 连接物理引擎
            self.bullet_client = p.connect(p.GUI)
        else:
            self.bullet_client = p.connect(p.DIRECT)
        self.g = 9.80151
        p.setGravity(0, 0, -self.g)  # 设置重力值
        p.setAdditionalSearchPath(pd.getDataPath())  # 添加pybullet_data的文件路径
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 不显示GUI上的控件
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)  # 不让CPU上的集成显卡参与渲染工作
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 打开渲染
        p.setRealTimeSimulation(0)  # 关闭实时模拟
        self.sim_frequency = 100  # 仿真运行的频率 默认240Hz
        self.time_step = 1. / self.sim_frequency  # The time step of the simulation
        p.setTimeStep(self.time_step)
        # *************** 模型导入 *************** #
        self.control_mode = '2D'  # 选择控制模式: 'fixed' '1D' '2D' '3D'
        self.init_position = [0, 0, 1]  # 模型导入时 base连杆在世界坐标系下的初始位置 0.8
        self.init_orientation = [0, 0, 0, 1]  # 模型导入时 base连杆在世界坐标系下的初始姿态（欧拉角为[0, 0, 0]）
        flags = p.URDF_USE_IMPLICIT_CYLINDER   # 使用一个平滑的隐式圆柱体 | p.URDF_USE_INERTIA_FROM_FILE
        self.floor_id = p.loadURDF("plane.urdf", useMaximalCoordinates=False)  # 导入地面
        self.robot_id = p.loadURDF(fileName="OneLeg/urdf/OneLeg.urdf", basePosition=self.init_position,
                                   baseOrientation=self.init_orientation, useFixedBase=False, flags=flags)  # 导入机器人
        self.planar_id = p.loadURDF(fileName="PlanarConstraint/Planar.urdf", basePosition=self.init_position,
                                    baseOrientation=self.init_orientation, useFixedBase=True)  # 导入平面约束
        # self.robot_id = p.loadURDF(fileName="OneLeg2D/urdf/OneLeg2D.urdf", basePosition=self.init_position,
        #                            baseOrientation=self.init_orientation, useFixedBase=True, flags=flags)  # 带平面约束的机器人

        # *************** 模型信息获取 *************** #
        self.joints_id = []  # 转动和移动关节索引
        self.hip_joint_id = []
        self.kp_joint_id = []
        self.foot_id = []  # 足连杆索引
        self.joint_num_all = p.getNumJoints(self.robot_id)
        print("关节数量: ", self.joint_num_all)
        for joint_index in range(self.joint_num_all):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            print("关节信息: ", joint_info)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                self.joints_id.append(joint_index)
            if str(p.getJointInfo(self.robot_id, joint_index)[12].decode()) == 'Foot_Link':
                self.foot_id.append(joint_index)
            if "Roll" in joint_name:
                self.hip_joint_id.append(joint_index)
            if "Pitch" in joint_name:
                self.hip_joint_id.append(joint_index)
            if "Knee" in joint_name:
                self.kp_joint_id.append(joint_index)
        print('转动和移动关节ID: ', self.joints_id)
        print('髋关节关节ID: ', self.hip_joint_id)
        print('膝关节关节ID: ', self.kp_joint_id)
        self.joint_num = len(self.joints_id)  # 转动和移动关节数量
        active_joints_id = [i for i in range(self.joint_num_all) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        active_joints_name = [p.getJointInfo(self.robot_id, i)[1] for i in active_joints_id]
        print("活动关节ID: ", active_joints_id)
        print("活动关节名称: ", active_joints_name)
        self.bullet_interface = BulletInterface(self.planar_id, self.robot_id, self.joints_id, self.foot_id,
                                                self.control_mode, self.init_position, self.init_orientation)  # 创建bullet接口类对象
        self.bullet_interface.add_sliders()  # 在控制界面添加调试参数
        self.bullet_interface.set_constraint()  # 添加约束
        self.cam_dis = 1.2  # 相机的距离 m
        self.cam_yaw = 0    # 相机的偏航角 度
        self.cam_pitch = 0  # 相机的俯仰角 度
        self.camPosZ = 0.8  # 相机位置Z向位置 m
        self.bullet_interface.reset_camera(self.cam_dis, self.cam_yaw, self.cam_pitch, self.init_position)
        self.bullet_interface.print_joint_info()  # 打印关节信息

        # *************** 变量初始化 **************** #
        # 躯干
        self.base_position_w = self.init_position  # 初始 当前的 躯干位置
        self.base_orientation_w = p.getEulerFromQuaternion(self.init_orientation)  # 初始 当前的 躯干位置
        # 初始 当前 世界坐标系下 躯干的位置[x、y、z]、姿态[α,β,γ]、线速度[vx、vy、vz]、角速度[wx、wy、wz]
        self.body_state = np.array((self.base_position_w, self.base_orientation_w, [0, 0, 0], [0, 0, 0]))  # 初始 上一刻 躯干状态
        self.body_state_pre = self.body_state  # 初始 上一刻 躯干的状态
        self.base_linear_vel = np.zeros(3)  # 初始 躯干的线速度
        self.base_angular_vel = np.zeros(3)  # 初始 躯干的欧拉角变化率

        # 关节
        # self.q_init_IK = np.array([0, 0, 0])  # 初始 关节位置 (逆运动学)
        # self.q_init_FK = np.array([0, 90/180*np.pi, 0])  # 初始 关节位置 (正运动学) [0, 0, -0.7471]
        # self.q_init_IK = np.array([0, 0, 45/180*np.pi])  # 初始 关节位置 (逆运动学)
        # self.q_init_FK = np.array([0, 90/180*np.pi, -45/180*np.pi])  # 初始 关节位置 (正运动学) [0.24544, 0, -0.64544]
        self.q_init_IK = np.array([0, -25 / 180 * np.pi, 45 / 180 * np.pi])  # 初始 关节位置 (逆运动学)
        self.q_init_FK = np.array([0, 115 / 180 * np.pi, -45 / 180 * np.pi])  # 初始 关节位置 (正运动学) [0.00988, 0, -0.690458]
        # self.q_init_IK = np.array([0, -30/180*np.pi, 60/180*np.pi])  # 初始 关节位置 (逆运动学)
        # self.q_init_FK = np.array([0, 120/180*np.pi, -60/180*np.pi])  # 初始 关节位置 (正运动学)
        self.motor_angle = np.zeros(self.joint_num)  # 当前 电机位置
        self.motor_angle_pre = np.zeros(self.joint_num)  # 上一刻 电机位置
        self.motor_angle_desired = np.zeros(self.joint_num)  # 期望的 电机位置
        self.motor_angle_dot = np.zeros(self.joint_num)  # 当前 电机速度
        self.joint_angle = self.motor_angle + self.q_init_FK  # 初始 当前 关节位置
        self.joint_angle_pre = self.joint_angle  # 初始 上一刻 关节位置
        self.joint_angle_dot_pre = np.zeros(self.joint_num)  # 上一刻 当前关节速度
        # 验证 正运动学
        print("x, y, z: ", kinematics.foot_position_in_hr_frame(self.joint_angle[0], self.joint_angle[1],
                                                                self.joint_angle[2], l_hr_sign=0))
        print("x2, y2, z2: ", kinematics.foot_position_in_hr_frame2(self.joint_angle[0], self.joint_angle[1],
                                                                    self.joint_angle[2], l_hr_sign=0))
        # 验证 逆运动学
        kinematics.foot_position_in_hr_frame_to_joint_angle([0, 0, -0.7471])  # [-0.5196, 0, -0.5006]、[0, 0, -0.7471]
        kinematics.foot_position_in_hr_frame_to_joint_angle2([0, 0, -0.7471])  # [0.24544, 0, -0.64544]、[0.7471, 0, 0]

        # 腿
        self.l2 = kinematics.l_up  # 大腿长
        self.l3 = kinematics.l_low  # 小腿长
        self.l_vir_normal_s = 0.9 * (self.l2 + self.l3)  # 虚拟腿原长 前摆  0.7
        self.l_vir_normal = 0.9 * (self.l2 + self.l3)  # 虚拟腿原长 0.67239(0.9)  0.85
        self.l_vir = kinematics.get_virtual_leg_length(self.joint_angle[2])  # 初始 当前 虚拟腿长
        self.l_vir_dot = 0  # 伸长为正
        # 足端状态
        self.foot_position_w = np.asarray(p.getLinkState(self.robot_id, self.foot_id[0])[4])  # 当前 足端位置(在世界系)
        self.foot_position_w_pre = self.foot_position_w  # 初始 上一刻 足端位置状态(世界系)
        self.foot_position_H = self.foot_position_w - self.base_position_w # 初始 当前 足端位置(H系)
        self.foot_position_H_pre = self.foot_position_H  # 初始 上一刻 足端位置(H系)
        self.foot_position_B = self.foot_position_H  # 初始 当前 足端位置(B系)
        self.foot_position_B_pre = self.foot_position_B  # 初始 上一刻 足端位置(B系)
        self.foot_velocity_w = np.zeros(3)  # 初始 当前 足端速度(世界系)
        self.foot_velocity_H = np.zeros(3)  # 初始 当前 足端速度(H系)
        self.foot_velocity_B = np.zeros(3)  # 初始 当前 足端速度(B系)
        self.foot_contact = False  # 当前 足端是否触地（初始不触地）
        self.foot_contact_pre = False  # 上一刻 足端是否触地（初始不触地）
        # 矩阵
        self.R_H_B = None  # 从{B}系到{H}系的转换  {H}系原点与躯干坐标系{B}重合，方向与世界坐标系{W}平行
        self.R_B_H = None  # 从{H}系到{B}系的转换  {B}系原点与{hr}系重合，初始方向相同）
        self.JLeg = None  # 腿部雅克比矩阵
        self.JLeg_T = None  # 腿部雅克比矩阵的 转置

        # 定义数据实时显示变量
        self.torque = np.zeros(3)
        self.torque_g = np.zeros(3)  # 重力补偿力矩
        self.torque_pre = np.zeros(3)
        self.torque_all = np.zeros((self.sim_frequency, self.joint_num))  # 历史力矩数据
        # self.base_position_w_all = np.zeros((self.sim_frequency, self.joint_num))  # 历史躯干位置数据
        self.define_save_variables()  # 定义保存数据变量

        # SLIP模型的参数
        self.state_machine = 'FLIGHT'  # 初始 状态机的状态 THRUST FLIGHT
        self.velocity_desire = [1.5, 0, 0]  # 期望的速度
        self.add_velocity = 0.1  # 按一次键盘方向键的速度增量
        self.kSpring = 11000  # 11000
        self.cSpring = 60  # 30 60
        self.kp_pose = [3000, 3000, 0]  # [3000, 3000, 0]
        self.kd_pose = [60, 80, 0]  # [60, 80, 0]
        self.kp_foot = [4000, 4000, 1600]  # [5000, 5000, 0] [5500, 5500, 0] [4000, 4000, 1500]
        self.kd_foot = [60, 60, 4]  # [70, 70, 0] [60, 60, 4]
        self.kv_x = 0.01  # 0.05 0.03 水平速度控制(落足点估计)x方向速度增量系数，决定了加速所需时间，太大的加速系数会导致机体崩溃
        self.kv_y = 0.01  # 0.05 0.03 y方向速度增量系数
        # self.thrust = 1600 + self.velocity_desire[0] * 1000  # 1500 1800
        self.thrust = 1600 + 700 * (np.log(self.velocity_desire[0] + 1) / np.log(2.779))  # 补充推力，定义弹簧压缩为正
        self.add_thrust = 50

        # 基本变量
        self.sim_time = 0  # 仿真时间 从仿真启动开始的计时器
        self.sim_steps = 30000  # 仿真步数
        self.stance_time = 0  # 上一个 支撑相的持续时间
        self.flight_time = 0.3  # 上一个 腾空相的持续时间
        self.track_time = 0.2
        self.track_t = 0  # 轨迹跟随已进行的时间
        self.stance_start = 0
        self.flight_start = 0
        self.foot_start = self.foot_position_H  # 初始 腾空相足端轨迹起点
        self.foot_end = self.foot_placement_to_motor_angle(self.l_vir, get_foot_end=True)
        self.bezier_trajectory = BezierTrajectory()  # 创建 基于贝塞尔曲线生成足端轨迹的类对象
        # self.init_plot()  # 绘图
        self.robot_init()  # 机器人初始化

    # ********************** 基本函数 *****************************************************************
    def robot_init(self):
        """
        机器人参数初始化
        """
        # 重置关节状态
        for i in range(self.joint_num):
            p.resetJointState(self.robot_id, jointIndex=self.joints_id[i], targetValue=0.0, targetVelocity=0.0)
        # 添加关节角约束
        # p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=self.joints_id[0], jointLowerLimit=-1, jointUpperLimit=1)
        # p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=self.joints_id[1], jointLowerLimit=-1, jointUpperLimit=1)
        # p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=self.joints_id[2], jointLowerLimit=-1.5, jointUpperLimit=0)
        # 修改连杆颜色
        link_id = np.arange(-1, self.joint_num_all)
        color_id = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]]  # 黑 蓝 绿 绿 红
        for i, j in zip(link_id, color_id):
            p.changeVisualShape(objectUniqueId=self.robot_id, linkIndex=i, rgbaColor=j)
        # 修改连杆质量
        self.m_body = 120
        self.m2 = 6  # 大腿重 3
        self.m3 = 4  # 小腿重 2
        link_mass = [self.m_body, self.m2, self.m3]
        link_id = [-1, 1, 2]
        # for i, j in zip(link_id, link_mass):
        #     p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=i, mass=j)
        # 移除默认的关节阻尼
        for joint_index in range(self.joint_num):
            p.changeDynamics(self.robot_id, joint_index, linearDamping=0, angularDamping=0)
            p.changeDynamics(joint_index, -1, linearDamping=0, angularDamping=0)
        # 更改与地面的接触模型
        # p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=3, restitution=0.5, contactStiffness=10**8, contactDamping=10**5)
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=3, lateralFriction=0.7, restitution=0.6,
                         contactStiffness=35000, contactDamping=0.57)
        # p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=1, mass=10, lateralFriction=10, rollingFriction=10,
        #                  spinningFriction=10, restitution=0.5, contactStiffness=10**8, contactDamping=10**5)
        # 刚度系数为 2855，阻尼系数为 0.57，摩擦系数为 0.5，恢复系数为 0.6

        self.bullet_interface.print_joint_info()  # 打印关节信息
        self.bullet_interface.print_link_info()   # 打印连杆信息

    def define_save_variables(self):
        """
        定义保存的变量
        """
        self.sim_time_all = []
        self.base_position_w_all = []
        self.base_orientation_w_all = []
        self.base_linear_vel_all = []
        self.torques_all = []
        self.joints_angle_all = []
        self.foot_position_w_all = []
        self.foot_position_H_all = []
        self.foot_position_H_desired_all = []
        self.foot_velocity_B_all = []
        self.foot_velocity_B_sensor_all = []

    def update_save_variables(self):
        """
        更新保存的变量
        """
        self.sim_time_all.append(self.sim_time)
        self.base_position_w_all.append(self.base_position_w)
        self.base_orientation_w_all.append(self.base_orientation_w)
        self.base_linear_vel_all.append(self.base_linear_vel)
        self.torques_all.append(self.torque)
        self.joints_angle_all.append(self.joint_angle)
        self.foot_position_w_all.append(self.foot_position_w)
        self.foot_position_H_all.append(self.foot_position_H)
        self.foot_velocity_B_all.append(self.foot_velocity_B)
        self.foot_velocity_B_sensor_all.append(self.foot_velocity_B_sensor)

    def save_variables(self):
        """
        保存变量
        """
        np.save("sim_time_all.npy", np.asarray(self.sim_time_all))
        np.save("base_position_w_all.npy", np.asarray(self.base_position_w_all))
        np.save("base_orientation_w_all.npy", np.asarray(self.base_orientation_w_all))
        np.save("base_linear_vel_all.npy", np.asarray(self.base_linear_vel_all))
        np.save("torques_all.npy", np.asarray(self.torques_all))
        np.save("joints_angle_all.npy", np.asarray(self.joints_angle_all))
        np.save("foot_position_w_all.npy", np.asarray(self.foot_position_w_all))
        np.save("foot_position_H_all.npy", np.asarray(self.foot_position_H_all))
        np.save("foot_position_H_desired_all.npy", np.asarray(self.foot_position_H_desired_all))
        np.save("foot_velocity_B_all.npy", np.asarray(self.foot_velocity_B_all))
        np.save("foot_velocity_B_sensor_all.npy", np.asarray(self.foot_velocity_B_sensor_all))

    def trajectory_generation(self, foot_start, foot_end):
        """
        基于贝塞尔曲线生成足端轨迹
        """
        T = 1  # 轨迹周期 s
        t = self.sim_time % T  # 当前轨迹时刻: 0-t_sw, 0-t_st
        t_sw = T/2  # 摆动相轨迹时间
        t_st = T/2  # 支撑相轨迹时间
        if t <= t_sw:
            foot_pos, foot_vel, foot_acc, _ = self.bezier_trajectory.swing_trajectory2(foot_start, foot_end, t, t_sw)
        elif t > t_sw:
            foot_pos, foot_vel, foot_acc, _ = self.bezier_trajectory.stance_trajectory(foot_start, foot_end, t - t_sw, t_st)
        else:
            foot_pos, foot_vel, foot_acc = [0, 0, 0], [0, 0, 0], [0, 0, 0]
            print('Error: t out of range')
        return foot_pos, foot_vel, foot_acc

    def update_robot_state(self):
        """
        函数功能：机器人状态更新，包括读取传感器的数据以及一些状态估计
        """
        self.sim_time += self.time_step  # 仿真时间更新
        self.update_keyboard_input()  # 获取键盘输入 更新 期望速度
        # 状态更新
        body_state = self.bullet_interface.get_body_state()  # 获取新的 躯干状态 位置/姿态/线速度/角速度
        self.body_state_pre = self.body_state  # 更新上一刻的 躯干状态
        self.body_state = body_state  # 更新 当前的 躯干状态
        self.base_position_w = self.body_state[0]  # 更新 当前的 躯干位置
        self.base_orientation_w = self.body_state[1]  # 更新 当前的 躯干姿态
        self.base_linear_vel = self.body_state[2]  # 更新 当前的 躯干线速度（世界系）
        _, orientation_inv = p.invertTransform(position=[0, 0, 0], orientation=p.getQuaternionFromEuler(self.base_orientation_w))
        base_velocity_relative, _ = p.multiplyTransforms(positionA=[0, 0, 0], orientationA=orientation_inv,
                                                         positionB=self.base_linear_vel, orientationB=[0, 0, 0, 1])
        base_angular_vel = self.body_state[3]    # 更新 当前的 躯干角速度
        self.base_angular_vel = kinematics.angular_velocity_to_rpy_rate(self.base_orientation_w, self.body_state[3])  # 躯干角速度 转换成 欧拉角变化率
        print("self.body_state: ", self.body_state)
        print("躯干的位置{W}: ", self.base_position_w)

        self.motor_angle, self.motor_angle_dot = self.bullet_interface.get_joints_state()  # 更新 关节状态
        self.joint_angle = self.motor_angle + self.q_init_FK
        print("关节速度: ", self.motor_angle_dot)

        self.R_H_B = kinematics.rotation_matrix(self.base_orientation_w[0], self.base_orientation_w[1],
                                                self.base_orientation_w[2])  # 从{B}系到{H}系的转换
        self.R_B_H = self.R_H_B.T  # 从{H}系到{B}系的转换
        self.JLeg = kinematics.leg_jacobian_matrix(self.joint_angle[0], self.joint_angle[1],
                                                   self.joint_angle[2], l_hr_sign=0)  # 腿部雅克比矩阵
        self.JLeg_T = kinematics.matrix_transpose(self.JLeg)  # 腿部雅克比矩阵的 转置
        print("self.R_H_B: ", self.R_H_B)
        print("self.R_B_H: ", self.R_B_H)
        print("腿部雅可比矩阵: ", self.JLeg)
        print("腿部雅可比矩阵的转置: ", self.JLeg_T)

        foot_position_w, foot_velocity_w = self.bullet_interface.get_foot_link_state()  # 获取新的 足端坐标系 位置
        self.foot_position_w_pre = self.foot_position_w  # 更新 上一刻 足端位置（世界系）
        self.foot_velocity_w = (foot_position_w - self.foot_position_w_pre) / self.time_step  # 更新 足端速度（世界系）
        self.foot_position_w = foot_position_w  # 更新 当前 足端位置（世界系）

        foot_position_H = self.foot_position_w - self.base_position_w  # 获取新的 足端坐标系 位置 (H系)
        self.foot_position_H_pre = self.foot_position_H  # 更新 上一刻 足端位置(H系)
        self.foot_velocity_H = (foot_position_H - self.foot_position_H_pre) / self.time_step  # 更新 足端速度(H系)
        self.foot_position_H = foot_position_H  # 更新 当前的 足端位置(H系)

        foot_position_B = self.R_B_H @ foot_position_H  # 获取新的 足端坐标系 位置(B系)
        foot_velocity_B = foot_velocity_w - self.base_linear_vel
        self.foot_position_B_pre = self.foot_position_B  # 更新 上一刻 足端位置(B系)
        self.foot_velocity_B = (foot_position_B - self.foot_position_B_pre) / self.time_step  # 更新 足端速度(B系)
        self.foot_position_B = foot_position_B  # 更新 当前的 足端位置(B系)
        _, orientation_inv = p.invertTransform(position=[0, 0, 0], orientation=p.getQuaternionFromEuler(self.base_orientation_w))
        # 将世界系下的绝对速度变换成相对base系的速度(验证正确)
        foot_velocity_relative, _ = p.multiplyTransforms(positionA=[0, 0, 0], orientationA=orientation_inv,
                                                         positionB=foot_velocity_w, orientationB=[0, 0, 0, 1])
        self.foot_velocity_B_sensor = np.asarray(foot_velocity_relative)  # 更准确
        # 雅可比矩阵验证
        # v_foot = self.JLeg @ self.motor_angle_dot  # 在base系下表达的足端速度(验证正确)
        # print("v_foot: ", v_foot)

        foot_contact = self.bullet_interface.get_foot_contact_state()  # 获取新的 足端触地状态
        self.foot_contact_pre = self.foot_contact  # 更新上一次 足端触地状态
        self.foot_contact = foot_contact  # 更新 当前 足端触地状态
        l_vir = kinematics.get_virtual_leg_length(self.joint_angle[2])  # 获取新 虚拟腿长
        self.l_vir_dot = (l_vir - self.l_vir) / self.time_step  # 计算虚拟腿长速度  伸长为正 缩短为负
        self.l_vir = l_vir  # 更新 当前 虚拟腿长
        self.torque_g = self.leg_gravity_compensation(self.joint_angle)  # 更新 重力补偿力矩

        # 基本量 更新
        self.torque_all[:-1] = self.torque_all[1:]  # 更新力矩队列
        self.torque_all[-1] = self.torque
        self.torque_pre = self.torque
        self.update_last_phase_time()  # 更新 上一次支撑相时间
        self.update_finite_state_machine()  # 更新 有限状态机
        # self.update_plot()  # 更新绘图
        self.bullet_interface.plot_trajectory_in_bullet(self.body_state_pre[0], self.body_state[0],
                                                        self.foot_position_w_pre, self.foot_position_w)  # 躯干和足端轨迹显示
        sliders_value = self.bullet_interface.get_sliders_value()  # 更新控制滑块的值
        # p.getCameraImage(800, 600)
        # if remove_button != self.pre_remove_button:
        #     p.removeAllUserDebugItems()
        #     self.pre_remove_button = remove_button
        # if camera_image_flag != 0:
        #     self.bullet_interface.get_camera_image()
        self.bullet_interface.reset_camera(self.cam_dis, self.cam_yaw, self.cam_pitch,
                                           [round(self.base_position_w[0], 2), 0, self.camPosZ])
        self.update_save_variables()  # 更新保存变量

    # ********************** 画图函数 *****************************************************************
    def init_plot(self):
        """
        初始化画图
        """
        self.fig = plt.figure(num=1, figsize=(12, 9), dpi=80)
        motors_name = ['Torque hr [N/m]', 'Torque hp [N/m]', 'Torque kp [N/m]']
        self.line_torque = []
        self.time = np.arange(0, 1, self.time_step)
        for i in range(self.joint_num):
            plt.subplot(self.joint_num, 1, i + 1)
            line_torque, = plt.plot(self.time, self.torque_all[:, i], color='r', linewidth=1.5, linestyle='-')
            self.line_torque.append(line_torque)  # 获取曲线图对象，逗号不可少，如果没有逗号，得到的是元组。
            plt.ylabel('{}'.format(motors_name[i]), fontproperties='Times New Roman', fontsize='16')
            plt.ylim([-2000, 2000])
            plt.grid(True, linestyle=":", color="k", linewidth="1")
        plt.xlabel('Simulation time [ms]', fontproperties='Times New Roman', fontsize='16')
        plt.suptitle('关节力矩', fontproperties='SimHei', fontsize=20, fontweight='bold')
        self.fig.tight_layout()

    def update_plot(self):
        """
        更新绘图数据
        """
        for i in range(self.joint_num):
            self.line_torque[i].set_ydata(self.torque_all[:, i])
        plt.draw()
        # plt.pause(1 / 1000)

    # ********************** SLIP模型控制函数 *****************************************************************
    def update_last_phase_time(self):
        """
        函数功能：更新上次支撑相、腾空相时间
        """
        if self.foot_contact_pre is False and self.foot_contact is True:  # 进入支撑相
            self.stance_start = self.sim_time
            flight_end = self.sim_time
            self.flight_time = flight_end - self.flight_start
            self.track_time = 2*self.flight_time/3
        if self.foot_contact_pre is True and self.foot_contact is False:  # 进入摆动相
            stance_end = self.sim_time
            self.stance_time = stance_end - self.stance_start
            self.flight_start = self.sim_time
        print("上次支撑相时间: ", self.stance_time)
        print("上次腾空相时间: ", self.flight_time)

    def update_keyboard_input(self):
        """
        读取键盘: 增减 期望速度、补充推力、相机位置/距离/朝向
        """
        key_dict = p.getKeyboardEvents()
        if len(key_dict):
            if p.B3G_UP_ARROW in key_dict and key_dict[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED:
                self.velocity_desire[0] += self.add_velocity  # 上箭头 + x方向速度
            elif p.B3G_DOWN_ARROW in key_dict:
                self.velocity_desire[0] -= self.add_velocity  # 下箭头 - x方向速度
            elif p.B3G_LEFT_ARROW in key_dict:
                self.velocity_desire[1] += self.add_velocity  # 左箭头 + y方向速度
            elif p.B3G_RIGHT_ARROW in key_dict:
                self.velocity_desire[1] -= self.add_velocity  # 右箭头 - y方向速度
            elif p.B3G_F9 in key_dict:
                self.thrust += self.add_thrust  # 增加补充推力
            elif p.B3G_F10 in key_dict:
                self.thrust -= self.add_thrust  # 减小补充推力
            elif p.B3G_F1 in key_dict:
                self.camPosZ += 0.01  # 相机位置向上移动
            elif p.B3G_F2 in key_dict:
                self.camPosZ -= 0.01  # 相机位置向下移动
            elif p.B3G_F3 in key_dict:
                self.cam_dis += 0.01  # 相机距离增大
            elif p.B3G_F4 in key_dict:
                self.cam_dis -= 0.01  # 相机距离减小
            elif p.B3G_F5 in key_dict:
                self.cam_yaw += 1  # 相机偏航角增大
            elif p.B3G_F6 in key_dict:
                self.cam_yaw -= 1  # 相机偏航角减小
            elif p.B3G_F7 in key_dict:
                self.cam_pitch += 1  # 相机俯仰角增大
            elif p.B3G_F8 in key_dict:
                self.cam_pitch -= 1  # 相机俯仰角减小

    def update_finite_state_machine(self):
        """
        *函数功能：更新 有限状态机
        *----------------------------------------------------
        * |  名称   |     符号        | 触发条件 |
        *----------------------------------------------------
        * |  触地   |     LOADING     | 足底触地 |
        * | 压缩腿  |   COMPRESSION   | 腿长小于阈值 |
        * | 伸长腿  |    THRUST      | 腿长导数大于0 |
        * |  离地  |   UNLOADING    | 腿长大于阈值 |
        * |  飞行  |    FLIGHT     | 足底离地 |
        *----------------------------------------------------
        """
        r_threshold = 0.9  # 状态机阈值
        if self.state_machine == 'LOADING':  # 触地
            if self.l_vir < self.l_vir_normal:
                # if self.l_vir < self.l_vir_normal * r_threshold:
                self.state_machine = 'COMPRESSION'  # 虚拟弹簧长度小于阈值，切换至压缩状态
        elif self.state_machine == 'COMPRESSION':  # 压缩 or self.l_vir < self.l_vir_normal*0.6
            if self.l_vir_dot > 0:
                self.state_machine = 'THRUST'  # 虚拟弹簧长度开始增加，切换至伸展状态
        elif self.state_machine == 'THRUST':  # 伸展
            if self.l_vir > self.l_vir_normal * r_threshold:
                self.state_machine = 'UNLOADING'  # 虚拟弹簧长度大于阈值，切换至离地状态
        elif self.state_machine == 'UNLOADING':  # 离地
            if not self.foot_contact:
                self.state_machine = 'FLIGHT'  # 足端离地，切换至腾空状态
        elif self.state_machine == 'FLIGHT':  # 腾空
            # if abs(self.motor_angle_dot[1]) < 0.001 and self.base_linear_vel[2] < 0:
            if self.track_t >= self.track_time:
                self.state_machine = 'RETRACTION'
        elif self.state_machine == 'RETRACTION':
            if self.foot_contact:
                self.state_machine = 'LOADING'  # 足端触地，切换至触地状态
        print('状态机状态: ', self.state_machine)

    def leg_mimic_spring(self, l_vir_normal):
        """
        函数功能：控制膝关节扭矩，使腿部模拟弹簧，虚拟腿压缩为正（推力为正）
        l_vir_normal: 弹簧原长
        """
        F_vir = self.kSpring * (l_vir_normal - self.l_vir) - self.cSpring * self.l_vir_dot  # 虚拟弹簧力
        print("虚拟弹簧力: ", F_vir)
        if self.state_machine == 'THRUST':
            F_vir += self.thrust  # 补充推力、调整跳跃高度
        # if self.state_machine == 'UNLOADING':
        #     F_vir += self.thrust  # 补充推力、调整跳跃高度
        joint_torque = self.spring_force_to_joint_torque(F_vir)
        return joint_torque[2]

    def spring_force_to_joint_torque(self, F_vir):
        """
        函数功能：将虚拟腿弹簧力转换到关节力矩
        参数: F_vir -- 虚拟腿弹簧力（触地时，为虚拟腿受到来自地面的力），压缩为正
        """
        theta_vir_up = kinematics.get_theta_vir_up(self.joint_angle[2], self.l_vir)  # 虚拟腿与大腿的夹角
        theta_vir = self.joint_angle[1] + theta_vir_up  # 虚拟腿俯仰角
        GRFx = -F_vir * np.cos(theta_vir)
        GRFy = F_vir * np.sin(theta_vir) * np.sin(self.joint_angle[0])
        GRFz = F_vir * np.sin(theta_vir) * np.cos(self.joint_angle[0])
        GRF_xyz = np.array([GRFx, GRFy, GRFz])
        print("地面反力: ", GRF_xyz)
        foot_force_H = -GRF_xyz
        foot_force_B = self.R_B_H @ foot_force_H  # 转换到躯干坐标系{B}
        joint_torque = self.JLeg_T @ foot_force_B
        print("腿部模拟弹簧的关节力矩: ", joint_torque)
        return joint_torque

    def body_attitude_control(self):
        """
        函数功能: 躯干姿态控制
        """
        body_torque = self.kp_pose * (0 - self.base_orientation_w) + self.kd_pose * (0 - self.base_angular_vel)
        joint_torque = -body_torque
        print("支撑相姿态控制力矩: ", joint_torque)
        return joint_torque[0:2]

    def foot_placement_to_motor_angle(self, l_vir, get_foot_end=False):
        """
        函数功能: 在{H}系下规划期望落足点，逆解到髋关节角 再转换到电机角度，{H}系原点与躯干坐标系{B}重合，方向与世界坐标系平行
        l_vir: 虚拟腿长
        """
        x_f = self.base_linear_vel[0] * self.stance_time / 2 - self.kv_x * (self.velocity_desire[0] - self.base_linear_vel[0])
        y_f = self.base_linear_vel[1] * self.stance_time / 2 - self.kv_y * (self.velocity_desire[1] - self.base_linear_vel[1])
        z_f = -np.sqrt(l_vir ** 2 - x_f ** 2 - y_f ** 2)
        foot_position_H = [x_f-0.001, y_f, z_f]  # - 0.01  - 0.0252
        print("期望落足点{H}: ", foot_position_H)
        if get_foot_end:
            return foot_position_H
        foot_position_B = self.R_B_H @ foot_position_H  # 转换到躯干坐标系{B}
        print("期望落足点{B}: ", foot_position_B)
        joint_angle_desired = kinematics.foot_position_in_hr_frame_to_joint_angle(foot_position_B)
        motor_angle_desired = joint_angle_desired + self.q_init_IK
        return motor_angle_desired

    def leg_gravity_compensation(self, joint_angle):
        """
        函数功能：腿部重力补偿
        """
        theta_hr, theta_hp, theta_kp = joint_angle[0], joint_angle[1], joint_angle[2]
        M_hr = 1 / 2 * self.m2 * self.g * self.l2 * np.sin(theta_hr) * np.sin(theta_hp) + \
               self.m3 * self.g * (self.l2 * np.sin(theta_hr) * np.sin(theta_hp) +
                                   1 / 2 * self.l3 * np.sin(theta_hr) * np.sin(theta_hp + theta_kp))
        M_hp = -1 / 2 * self.m2 * self.g * self.l2 * np.cos(theta_hr) * np.cos(theta_hp) - \
               self.m3 * self.g * (self.l2 * np.cos(theta_hr) * np.cos(theta_hp) +
                                   1 / 2 * self.l3 * np.cos(theta_hr) * np.cos(theta_hp + theta_kp))
        M_kp = -1 / 2 * self.m3 * self.g * self.l3 * np.cos(theta_hr) * np.cos(theta_hp + theta_kp)
        torque_g = -np.asarray([M_hr, M_hp, M_kp])
        print("重力补偿力矩: ", torque_g)
        return torque_g

    def set_motors(self, motor_ids, mode, *args):
        """
        施加控制: 同时控制多个电机
        """
        if mode == p.POSITION_CONTROL:
            position = args[0]
            max_force = 2000 * np.ones(len(motor_ids))
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=motor_ids, controlMode=mode,
                                        targetPositions=position, forces=max_force)
        elif mode == p.TORQUE_CONTROL:
            torques = args[0]
            torques = np.clip(torques, -1500., 1500.)  # 裁剪防止超出范围
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=motor_ids, controlMode=mode, forces=torques)
        else:
            print('Error: mode is unknown')

    def set_motor(self, motor_id, mode, *args):
        """
        施加控制: 控制单个电机
        """
        torque = args[0]
        p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=motor_id[0], controlMode=mode, force=torque)

    def run(self):
        """
        主循环： 执行控制
        """
        logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "OneLeg-{}.mp4".format(
            time.strftime("%Y-%m-%d-%H-%M", time.gmtime())))  # 开始录制仿真视频
        for i in range(self.sim_steps):
            self.update_robot_state()  # 更新状态
            if self.control_mode == 'fixed':
                foot_start = np.array([-0.2, 0, -0.6])
                foot_end = np.array([0.2, 0, -0.6])
                foot_position_H_desired, foot_velocity_H_desired, foot_acceleration_H = self.trajectory_generation(foot_start, foot_end)
                foot_position_B_desired = self.R_B_H @ foot_position_H_desired
                foot_velocity_B_desired = self.R_B_H @ foot_velocity_H_desired
                foot_acceleration_B_desired = self.R_B_H @ foot_acceleration_H
                joint_angle_desired = kinematics.foot_position_in_hr_frame_to_joint_angle(foot_position_B_desired)
                self.motor_angle_desired = joint_angle_desired + self.q_init_IK
                motor_angle_dot_desired = np.linalg.inv(self.JLeg) @ foot_velocity_B_desired  # np.zeros(3)
                motor_angle_2dot_desired = self.JLeg_T @ foot_acceleration_B_desired  # np.zeros(3)
                # 足端PD控制
                # kp = np.array([5000, 5000, 5000])  # [0, 3000, 3000]
                # kd = np.array([20, 20, 20])  # [0, 3, 5]
                # foot_force = kp * (foot_position_B_desired - self.foot_position_B) + kd * (0 - self.foot_velocity_B)
                # foot_force = kp * (foot_position_B_desired - self.foot_position_B) + kd * (foot_velocity_B_desired - self.foot_velocity_B)  # 用估计的足端速度，效果较差
                # foot_force = kp * (foot_position_B_desired - self.foot_position_B) + kd * (foot_velocity_B_desired - self.foot_velocity_B_sensor)  # 用传感器测量的足端速度，较好
                # self.torque = self.JLeg_T @ foot_force
                # 关节PD控制
                kp = np.array([0, 3000, 2000])  # [0, 4000, 3500] [0, 4000, 2500]
                kd = np.array([0, 20, 6])  # [0, 15, 3] [0, 15, 5]
                self.torque = kp * (self.motor_angle_desired - self.motor_angle) + kd * (0 - self.motor_angle_dot) + 0
                # 足端PD + 关节PD
                # self.torque += kp * (self.motor_angle_desired - self.motor_angle) + kd * (0 - self.motor_angle_dot) + 0

                self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)  # 力控

            else:
                # 触地/离地
                if self.state_machine == 'LOADING' or self.state_machine == 'UNLOADING':
                    # ① 腿部模拟弹簧: 控制膝关节扭矩
                    self.torque[2] = self.leg_mimic_spring(self.l_vir_normal)
                    # ② 选择1: 控制 横滚、俯仰髋关节扭矩为0
                    # self.torque[0:2] = np.zeros(2)
                    # ② 选择2: 调整 躯干姿态
                    self.torque[0:2] = self.body_attitude_control()
                    # ③ 确定足端轨迹的起点
                    if self.state_machine == 'UNLOADING':
                        self.foot_start = self.foot_position_H
                        self.track_t = 0

                    self.foot_position_H_desired_all.append([0, 0, 0])
                    self.torque += self.torque_g
                    self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)
                # 压缩/伸展
                if self.state_machine == 'COMPRESSION' or self.state_machine == 'THRUST':
                    # ① 腿部模拟弹簧: 控制膝关节扭矩
                    self.torque[2] = self.leg_mimic_spring(self.l_vir_normal)
                    # ② 躯干姿态控制: 调整 横滚、俯仰髋关节
                    self.torque[0:2] = self.body_attitude_control()

                    self.foot_position_H_desired_all.append([0, 0, 0])
                    self.torque += self.torque_g
                    self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)
                # 腾空
                if self.state_machine == 'FLIGHT' or self.state_machine == 'RETRACTION':
                    self.foot_end = self.foot_placement_to_motor_angle(self.l_vir_normal, get_foot_end=True)
                    foot_trajectory = True  # True False
                    if foot_trajectory:
                        if self.state_machine == 'FLIGHT':
                            self.track_t += self.time_step
                            # if self.track_t <= self.track_time:
                            self.foot_pos_H_desired, _, _, _ = self.bezier_trajectory.swing_trajectory2(self.foot_start, self.foot_end,
                                                                                                            self.track_t, self.track_time)
                            foot_position_B_desired = self.R_B_H @ self.foot_pos_H_desired
                            joint_angle_desired = kinematics.foot_position_in_hr_frame_to_joint_angle(foot_position_B_desired)
                            motor_angle_desired = joint_angle_desired + self.q_init_IK
                            self.torque = self.kp_foot * (motor_angle_desired - self.motor_angle) + self.kd_foot * (0 - self.motor_angle_dot)  # 力控
                            # self.set_motors(self.joints_id, p.POSITION_CONTROL, motor_angle_desired)  # 位控

                            self.foot_position_H_desired_all.append(self.foot_pos_H_desired)
                            self.torque += self.torque_g
                            self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)

                        if self.state_machine == 'RETRACTION':
                            # 腿部模拟弹簧
                            self.torque[2] = self.leg_mimic_spring(self.l_vir_normal_s)
                            # 落足点估计 —— 水平速度控制
                            motor_angle_desired = self.foot_placement_to_motor_angle(self.l_vir)
                            torque = self.kp_foot * (motor_angle_desired - self.motor_angle) + self.kd_foot * (0 - self.motor_angle_dot)
                            self.torque[0:2] = torque[0:2]

                            self.foot_position_H_desired_all.append([0, 0, 0])
                            self.torque += self.torque_g
                            self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)
                            # self.set_motor(self.kp_joint_id, p.TORQUE_CONTROL, self.torque[2])
                    else:
                        # 腿部模拟弹簧
                        self.torque[2] = self.leg_mimic_spring(self.l_vir_normal)
                        # 落足点估计 —— 水平速度控制
                        motor_angle_desired = self.foot_placement_to_motor_angle(self.l_vir)
                        torque = self.kp_foot * (motor_angle_desired - self.motor_angle) + self.kd_foot * (0 - self.motor_angle_dot)  # 力控
                        # self.set_motors(self.hip_joint_id, p.POSITION_CONTROL, motor_angle_desired[0:2])  # 位控
                        self.torque[0:2] = torque[0:2]

                        self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)

            # self.torque += self.torque_g  # 重力补偿
            # self.set_motors(self.joints_id, p.TORQUE_CONTROL, self.torque)  # 力控
            # self.set_motors(self.joints_id, p.POSITION_CONTROL, self.motor_angle_desired)  # 位控
            print("torque: ", self.torque)
            p.stepSimulation()
            # time.sleep(1 / 100.)

        self.save_variables()  # 保存变量
        p.stopStateLogging(logging_id)  # 结束视频录制
        p.disconnect()  # 断开引擎连接


if __name__ == '__main__':
    one_leg = OneLeg()
    one_leg.run()
