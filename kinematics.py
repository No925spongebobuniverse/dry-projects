"""
***************************************************************************************
** Description: BigCat四足机器人和OneLeg单腿机器人的正逆运动学、矩阵运算文件
** Author:      黄森威
** Email:       15207847842@163.com
** date:        2022-11-11
***************************************************************************************
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import typing
import pybullet as p


l1 = l_hr = 0.107  # roll连杆的长度，roll坐标系与pitch坐标系沿y轴的偏置
l2 = l_up = 0.4  # 大腿长度
l3 = l_low = 0.3471  # 小腿长度
l_spM = 0.55  # MSpine脊柱连杆的长度
l_spFH = 0.38715  # FSpine、HSpine的长度
L = 1.068  # 前后腿髋关节的距离
W = 0.564  # 左右腿髋关节的距离
H = 0.288  # 腰部质心与髋关节的垂直距离
COM_OFFSET = -np.array([0.0, 0.0, 0.0])  # base连杆坐标系与MSpine坐标系的偏置，现在MSpine即为base连杆
Spine_OFFSETS = np.array([[l_spM / 2, 0., 0.], [l_spM / 2, 0., 0.],
                          [-l_spM / 2, 0., 0.], [-l_spM / 2, 0., 0.]]) + COM_OFFSET  # MSpine坐标系与FSpine、MSpine坐标系的偏置
Spine2Roll_OFFSET_y = 0.175  # FSpine、MSpine坐标系与roll坐标系沿y轴的偏置，x、z轴需要根据脊柱转角计算
Roll2Pitch_OFFSET_y = 0.107  # roll坐标系与pitch坐标系沿y轴的偏置，x、z轴为0
# 腿部顺序 A1: FR-FL-HR-HL  BigCat: FL-FR-HR-HL


def get_virtual_leg_length(theta_kp):
    """
    计算虚拟腿长（指的是大腿l_up和小腿l_low所形成的虚拟腿） 余弦定理表达式: c² = a² + b² -2abcosθ
    """
    theta_knee = np.pi + theta_kp  # 大腿和小腿所夹的角: theta_knee、膝关节的转角: theta_kp
    l_vir = np.sqrt(l_up ** 2 + l_low ** 2 - 2 * l_up * l_low * np.cos(theta_knee))
    return l_vir


def get_theta_vir_up(theta_kp, l_vir):
    """
    虚拟腿相对于大腿的夹角：逆时针为负，顺时针为正
    """
    return np.arcsin(l_low * np.sin(theta_kp) / l_vir)


def foot_position_in_hr_frame(theta_hr, theta_hp, theta_kp, l_hr_sign=np.array([1, -1, 1, -1])):
    """
    函数功能：正运动学求{hr}坐标系下的足端位置----几何方法(已验证正确)，角度计算的起点是大腿和小腿都水平向右，顺时针为正，逆时针为负。
    """
    l_vir = get_virtual_leg_length(theta_kp)
    theta_vir_up = get_theta_vir_up(theta_kp, l_vir)  # 虚拟腿与大腿的夹角
    theta_virp = theta_hp + theta_vir_up  # 虚拟腿俯仰角
    x_vir = l_vir * np.cos(theta_virp)
    y_vir = l_hr * l_hr_sign
    z_vir = -l_vir * np.sin(theta_virp)
    x = x_vir
    y = np.cos(theta_hr) * y_vir - np.sin(theta_hr) * z_vir
    z = np.sin(theta_hr) * y_vir + np.cos(theta_hr) * z_vir
    if l_hr_sign == 0 or len(l_hr_sign) == 1:  # 单腿仿真 或 计算一条腿时有用
        foot_position_hr = np.asarray([x, y, z])
    else:
        foot_position_hr = np.zeros((len(theta_hr), 3))
        for leg_id in range(len(theta_hr)):
            foot_position_hr[leg_id] = [x[leg_id], y[leg_id], z[leg_id]]
    return foot_position_hr


def foot_position_in_hr_frame2(theta_hr, theta_hp, theta_kp, l_hr_sign=np.array([1, -1, -1, 1])):
    """
    函数功能：正运动学求{hr}坐标系下的足端位置----解析法（已验证正确），角度计算的起点是大腿和小腿都水平向右，顺时针为正，逆时针为负。
    """
    x = np.cos(theta_hp) * l2 + np.cos(theta_hp + theta_kp) * l3
    y = l_hr_sign*np.cos(theta_hr)*l1 + np.sin(theta_hr) * (np.sin(theta_hp) * l2 + np.sin(theta_hp + theta_kp) * l3)
    z = l_hr_sign*np.sin(theta_hr)*l1 - np.cos(theta_hr) * (np.sin(theta_hp) * l2 + np.sin(theta_hp + theta_kp) * l3)
    if l_hr_sign == 0 or len(l_hr_sign) == 1:
        foot_position_hr = np.array([x, y, z])
    else:
        foot_position_hr = np.zeros((len(theta_hr), 3))
        for leg_id in range(len(theta_hr)):
            foot_position_hr[leg_id] = [x[leg_id], y[leg_id], z[leg_id]]
    return foot_position_hr


def foot_position_in_hr_frame_to_joint_angle(foot_position, leg_id=0):
    """
    函数功能：运动学逆解----几何方法（已验证在一个方向上正确），角度计算的起点是大腿和小腿都垂直向下，顺时针为正，逆时针为负。
    l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l1 = 0  # 单腿仿真时用
    l_hr_sign = [1, -1, -1, 1]
    l_hr = l1 * l_hr_sign[leg_id]
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    l_vir_square = x ** 2 + y ** 2 + z ** 2 - l_hr ** 2
    theta_knee = np.arccos((l_vir_square - l_up ** 2 - l_low ** 2) / (-2 * l_up * l_low))
    theta_kp = -(np.pi - theta_knee)  # theta_knee: 大腿和小腿所夹的角、theta_kp: 膝关节的转角
    # theta_kp = -np.arccos((l_vir_square - l_up ** 2 - l_low ** 2) / (2 * l_up * l_low))  # 或者直接这样
    l_vir = get_virtual_leg_length(theta_kp)
    # theta_hp = np.arcsin(-x / l_vir) + theta_kp / 2  # 当 l_up = l_low 时，可以这样求
    theta_virp = np.arcsin(-x / l_vir)  # 虚拟腿俯仰角
    theta_vir_up = get_theta_vir_up(theta_kp, l_vir)  # 虚拟腿与大腿的夹角
    theta_hp = theta_virp - theta_vir_up
    s1 = l_vir * np.cos(theta_hp + theta_vir_up) * y + l_hr * z
    c1 = l_hr * y - l_vir * np.cos(theta_hp + theta_vir_up) * z
    theta_hr = np.arctan2(s1, c1)
    print("theta_kp: ", np.rad2deg(theta_kp))
    print("theta_hp: ", np.rad2deg(theta_hp))
    print("theta_hr: ", np.rad2deg(theta_hr))
    return np.array([theta_hr, theta_hp, theta_kp])


def foot_position_in_hr_frame_to_joint_angle2(foot_position, leg_id=1, theta_hp_pre=90 ):
    """
    函数功能：运动学逆解----解析法（已验证正确，但有多解问题），l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_hr_sign = [1, -1, -1, 1]
    l1 = l_hr * l_hr_sign[leg_id]
    l1 = 0  # 单腿仿真时用
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    l_vir_square = x ** 2 + y ** 2 + z ** 2

    theta_kp = -np.arccos((l_vir_square - l1 ** 2 - l_up ** 2 - l_low ** 2) / (2 * l_up * l_low))
    theta_hp1 = -(np.arctan((l2 + l3 * np.cos(theta_kp)) / (l3 * np.sin(theta_kp)))
                  - np.arctan(x / np.sqrt(l2 ** 2 + l3 ** 2 + 2*l2*l3*np.cos(theta_kp) - x ** 2)))
    theta_hp = -(np.arctan((l2 + l3 * np.cos(theta_kp)) / (l3 * np.sin(theta_kp)))
                 - np.arctan(x / -np.sqrt(l2 ** 2 + l3 ** 2 + 2*l2*l3*np.cos(theta_kp) - x ** 2)))
    # theta_hp3 = -np.pi/2 - np.arctan(x/z) - np.arcsin((l3*np.sin(np.pi-theta_kp)/np.sqrt(l_vir_square)))
    # theta_hr1 = np.arctan(y/z)
    theta_hr = np.arctan((l1*z + (l2*np.sin(theta_hp) + l3*np.sin(theta_hp + theta_kp))*y) /
                         (l1*y - (l2*np.sin(theta_hp) + l3*np.sin(theta_hp + theta_kp))*z))
    print("theta_kp: ", np.rad2deg(theta_kp))
    print("theta_hp1: ", np.rad2deg(theta_hp1))
    print("theta_hp: ", np.rad2deg(theta_hp))
    # print("theta_hp3: ", np.rad2deg(theta_hp3))
    # print("theta_hr1: ", np.rad2deg(theta_hr1))
    print("theta_hr: ", np.rad2deg(theta_hr))
    return np.array([theta_hr, theta_hp, theta_kp])

def joint_angles_from_link_position(robot: typing.Any, link_position: typing.Sequence[float], link_id: int,
                                    joint_ids: typing.Sequence[int],
                                    base_translation: typing.Sequence[float] = (0, 0, 0),
                                    base_rotation: typing.Sequence[float] = (0, 0, 0, 1)):
    """Uses Inverse Kinematics to calculate joint angles.
  Args:
    robot: A robot instance.
    link_position: The (x, y, z) of the link in the body frame. This local frame is transformed relative to the COM frame
                   using a given translation and rotation.
    link_id: The link id as returned from loadURDF.
    joint_ids: The positional index of the joints. This can be different from the joint unique ids.
    base_translation: Additional base translation.
    base_rotation: Additional base rotation.
  Returns:
    A list of joint angles.
  """
    # Projects to local frame.
    base_position, base_orientation = robot.GetBasePosition(), robot.GetBaseOrientation()
    base_position, base_orientation = robot.pybullet_client.multiplyTransforms(base_position, base_orientation,
                                                                               base_translation, base_rotation)

    # Projects to world space.
    world_link_pos, _ = robot.pybullet_client.multiplyTransforms(base_position, base_orientation, link_position,
                                                                 (0, 0, 0, 1))
    ik_solver = 0
    all_joint_angles = robot.pybullet_client.calculateInverseKinematics(robot.quadruped, link_id, world_link_pos,
                                                                        solver=ik_solver)

    # Extract the relevant joint angles.
    joint_angles = [all_joint_angles[i] for i in joint_ids]
    return joint_angles


def link_position_in_base_frame(robot: typing.Any, link_id: int, ):
    """Computes the link's local position in the robot frame.
  Args:
    robot: A robot instance.
    link_id: The link to calculate its relative position.
  Returns:
    The relative position of the link.
  """
    base_position, base_orientation = robot.GetBasePosition(), robot.GetBaseOrientation()
    inverse_translation, inverse_rotation = robot.pybullet_client.invertTransform(base_position, base_orientation)

    link_state = robot.pybullet_client.getLinkState(robot.quadruped, link_id)
    link_position = link_state[0]
    link_local_position, _ = robot.pybullet_client.multiplyTransforms(
        inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))
    link_local_position = [0] * 3
    return np.array(link_local_position)


def foot_positions_in_Spine_frame(theta_hr, theta_hp, theta_kp, spine_angles):
    FSpine_angles = spine_angles[0]
    MSpine_angles = spine_angles[1]
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hr_frame(theta_hr, theta_hp, theta_kp)
    Spine2Roll_OFFSET = np.zeros((4, 3))
    for i in range(4):
        if i < 2:
            offset_sign_x = 1
        else:
            offset_sign_x = -1
        offset_sign_y = (-1) ** (i + 1)
        Spine2Roll_OFFSET[i] = [offset_sign_x * np.cos(FSpine_angles) * l_spFH, offset_sign_y * Spine2Roll_OFFSET_y,
                                -np.sin(FSpine_angles) * l_spFH]
    return foot_positions + Spine2Roll_OFFSET


def foot_positions_in_base_frame(angles):
    '''
    函数功能：正运动学, {B}坐标系下的足端坐标
    '''
    foot_angles = angles.reshape((4, 3))
    spine_angles = angles
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_positions_in_Spine_frame(foot_angles, spine_angles)
    return foot_positions + Spine_OFFSETS


def leg_jacobian_matrix(theta_hr, theta_hp, theta_kp, l_hr_sign=np.array([1, -1, -1, 1])):
    """
    计算腿部雅可比矩阵：解析法(验证正确)
    l_hip_sign: whether it's a left (1) or right (-1) leg.
    """
    l1 = l_hr * l_hr_sign
    if l_hr_sign == 0:  # 单腿仿真时用
        theta_hr = np.asarray([theta_hr])
        theta_hp = np.asarray([theta_hp])
        theta_kp = np.asarray([theta_kp])
    Jleg = []
    for i in range(len(theta_hr)):
        J = np.zeros((3, 3))
        J[0][0] = 0
        J[1][0] = -l1 * np.sin(theta_hr[i]) + \
                  np.cos(theta_hr[i]) * (l2 * np.sin(theta_hp[i]) + l3 * np.sin(theta_hp[i] + theta_kp[i]))
        J[2][0] = l1 * np.cos(theta_hr[i]) + \
                  np.sin(theta_hr[i]) * (l2 * np.sin(theta_hp[i]) + l3 * np.sin(theta_hp[i] + theta_kp[i]))

        J[0][1] = -l2 * np.sin(theta_hp[i]) - l3 * np.sin(theta_hp[i] + theta_kp[i])
        J[1][1] = np.sin(theta_hr[i]) * (l2 * np.cos(theta_hp[i]) + l3 * np.cos(theta_hp[i] + theta_kp[i]))
        J[2][1] = -np.cos(theta_hr[i]) * (l2 * np.cos(theta_hp[i]) + l3 * np.cos(theta_hp[i] + theta_kp[i]))

        J[0][2] = -l3 * np.sin(theta_hp[i] + theta_kp[i])
        J[1][2] = l3 * np.sin(theta_hr[i]) * np.cos(theta_hp[i] + theta_kp[i])
        J[2][2] = -l3 * np.cos(theta_hr[i]) * np.cos(theta_hp[i] + theta_kp[i])
        print("J: ", J)
        Jleg.append(J)
    Jleg = np.squeeze(np.asarray(Jleg))
    print("雅可比矩阵: ", Jleg)
    print("雅可比矩阵的数据类型: ", type(Jleg))
    print("雅可比矩阵的形状: ", Jleg.shape)
    return Jleg


def compute_jacobian(robot: typing.Any, link_id: int):
    """Computes the Jacobian matrix for the given link.
  Args:
    robot: A robot instance.
    link_id: The link id as returned from loadURDF.
  Returns:
    The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
    robot. For a quadruped, the first 6 columns of the matrix corresponds to
    the CoM translation and rotation. The columns corresponds to a leg can be
    extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
  """
    zero_vec = [0] * len(robot.joint_angle)
    print(robot.joint_angle)
    jv, _ = p.calculateJacobian(bodyUniqueId=robot.robot_id, linkIndex=3,
                                                    localPosition=(0, 0, 0), objPositions=robot.joint_angle,
                                                    objVelocities=zero_vec, objAccelerations=zero_vec)
    jacobian = np.array(jv)
    return jacobian


def leg_jacobian_matrix_transpose(matrix):
    """
    计算四条腿的雅可比矩阵的转置: 四足时用
    """
    Jleg_T = []
    for i in range(len(matrix)):
        # print("雅可比矩阵的转置: ", matrix[i].T)
        Jleg_T.append(matrix[i].T)
    return np.asarray(Jleg_T)


def matrix_transpose(matrix):
    """
    矩阵转置: 单腿时用
    """
    row = matrix.shape[0]
    column = matrix.shape[1]
    matrix_T = np.zeros((column, row))
    for i in range(column):
        for j in range(row):
            matrix_T[i][j] = matrix[j][i]
    return matrix_T


def matrix_inverse(matrix):
    """
    矩阵求逆
    """
    row = matrix.shape[0]
    column = matrix.shape[1]
    if row != column:
        return -1
    matrix_I = np.linalg.inv(matrix)
    return matrix_I


def angular_velocity_to_rpy_rate(rpy, w):
    """
    角速度转欧拉角变化率
    """
    R = np.array([[np.cos(rpy[2]) / np.cos(rpy[1]), np.sin(rpy[2]) / np.cos(rpy[1]), 0.],
                  [-np.sin(rpy[2]), np.cos(rpy[2]), 0.],
                  [np.cos(rpy[2]) * np.tan(rpy[1]), np.sin(rpy[2]) * np.tan(rpy[1]), 0.]])
    rpy_rate = R @ w.T
    return rpy_rate


def rotation_matrix(roll, pitch, yaw):
    """
    旋转变换矩阵: Rz(yaw) * Ry(pitch) * Rx(roll), RPY变换，绕定系旋转：左乘
    """
    Rx = matrix_Rx(roll)
    Ry = matrix_Ry(pitch)
    Rz = matrix_Rz(yaw)
    matrix_RPY = Rz @ Ry @ Rx
    return matrix_RPY


def matrix_Rx(angle):
    """
    绕x轴旋转的旋转矩阵Rx(roll)
    """
    matrix = np.array([[1., 0., 0.],
                       [0., np.cos(angle), -np.sin(angle)],
                       [0., np.sin(angle), np.cos(angle)]])
    return matrix


def matrix_Ry(angle):
    """
    绕y轴旋转的旋转矩阵Ry(pitch)
    """
    matrix = np.array([[np.cos(angle), 0., np.sin(angle)],
                       [0., 1., 0.],
                       [-np.sin(angle), 0., np.cos(angle)]])
    return matrix


def matrix_Rz(angle):
    """
    绕z轴旋转的旋转矩阵Rz(yaw)
    """
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0.],
                       [np.sin(angle), np.cos(angle), 0.],
                       [0., 0., 1.]])
    return matrix
