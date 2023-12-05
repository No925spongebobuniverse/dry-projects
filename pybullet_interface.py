"""
******************************************************
** Description: 与pybullet仿真环境交互的接口
** Author:      黄森威
** Email:       15207847842@163.com
** date:        2022-11-11
******************************************************
"""
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import generate_terrain


class BulletInterface(object):
    """
    与bullet的交互接口函数
    """
    def __init__(self, planar_id, robot_id, joints_id, foot_id, control_mode, init_position, init_orientation):
        self.planar_id = planar_id  # 地面id
        self.robot_id = robot_id  # 机器人id
        self.joint_num_all = p.getNumJoints(self.robot_id)  # 所有关节数量
        self.joints_id = joints_id  # 转动和移动关节索引
        self.joint_num = len(self.joints_id)  # 转动和移动关节数量
        self.foot_id = foot_id
        self.control_mode = control_mode  # 控制模式: 'fixed' '1D' '2D' '3D'
        self.init_position = init_position  # 模型导入时 base连杆在世界坐标系下的初始位置
        self.init_orientation = init_orientation  # 模型导入时 base连杆在世界坐标系下的初始姿态

    def set_constraint(self, force=100000):
        """
        函数功能: 添加约束
        """
        # 运动约束
        if self.control_mode == 'fixed':
            fixed_id = p.createConstraint(parentBodyUniqueId=self.robot_id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                          childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=self.init_position,
                                          childFrameOrientation=self.init_orientation)
            p.changeConstraint(fixed_id, maxForce=force)
        if self.control_mode == '1D':
            one_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 1])
            p.changeConstraint(one_id, maxForce=force)
        if self.control_mode == '2D':
            two_id = p.createConstraint(self.robot_id, -1, self.planar_id, 0, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                        [0, 0, 0])
            p.changeConstraint(two_id, maxForce=force)
        # if self.control_mode == '3D':
        # 创建两面墙
        # self.create_wall_body(length=120, width=0.4, height=6, position=[0, 0.6, 3])
        # self.create_wall_body(length=120, width=0.4, height=6, position=[0, -0.6, 3], color=[1, 1, 1, 0])

        # 创建墙体障碍物
        step = 0.1
        h = 0.1
        # for i in range(2, 20, 4):
        #     self.create_wall_body(length=0.02, width=2, height=h, position=[i, 0, 0.1], color=[0, 1, 1, 1])
        #     h += step
        # self.create_wall_body(length=0.05, width=1, height=0.2, position=[5, 0, 0.1], color=[0, 1, 1, 1])
        # terrain.upstair_terrain(stepwidth=0.4, mode="stair-fix")  # 创建楼梯
        # terrain.upstair_terrain(stepwidth=0.4, mode="stair-var")  # 创建楼梯
        # terrain.upstair_terrain(stepwidth=0.4, mode="downstair")  # 创建楼梯
        generate_terrain.upstair_terrain(stepwidth=0.4, slope=0.14, mode="slope")  # 创建斜坡

    def create_wall_body(self, length, width, height, position, mass=1000, color=[1, 1, 1, 1]):
        """
        函数功能: 创建墙体
        position: 墙体几何中心在世界系的位置
        """
        wall_collison_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2])
        wall_visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2],
                                             rgbaColor=color)
        wall_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=wall_collison_id,
                                    baseVisualShapeIndex=wall_visual_id, basePosition=position)
        fixed_id = p.createConstraint(parentBodyUniqueId=wall_id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=position)
        p.changeConstraint(fixed_id, maxForce=100000)

    # ********************** 机器人状态函数 *****************************************************************
    def get_body_state(self):
        """
        返回 世界坐标系下 躯干的位置[x、y、z]、姿态[α,β,γ]、线速度[vx、vy、vz]、角速度[wx、wy、wz]
        """
        body_position, body_orientation = p.getBasePositionAndOrientation(self.robot_id)  # 世界坐标系下 base质心的位姿
        body_orientation = np.asarray(p.getEulerFromQuaternion(body_orientation))  # 将四元数[x、y、z、w]姿态转换成用欧拉角表示
        body_linear_vel, body_angular_vel = p.getBaseVelocity(self.robot_id)
        return np.asarray(body_position), body_orientation, np.asarray(body_linear_vel), np.asarray(body_angular_vel)

    def get_joints_state(self):
        """
        返回: 关节实际的 位置、速度
        """
        q = np.zeros(self.joint_num)
        q_dot = np.zeros(self.joint_num)
        for i in range(self.joint_num):
            q[i], q_dot[i], _, _ = p.getJointState(self.robot_id, self.joints_id[i])
        return q, q_dot

    def get_foot_link_state(self):
        """
        返回: 世界坐标系、init系的 足端坐标系的位置
        """
        foot_state = p.getLinkState(self.robot_id, self.foot_id[0], computeLinkVelocity=1)
        foot_com_position_w = np.asarray(foot_state[0])  # 足端质心的位置（世界坐标系）
        foot_frame_position_w = np.asarray(foot_state[4])  # 足端坐标系的位置（世界坐标系）
        foot_velocity_w = np.asarray(foot_state[6])
        # 在base连杆初始位姿下(init系) 世界坐标系的位姿
        init_position_inv, init_orientation_inv = p.invertTransform(position=self.init_position,
                                                                    orientation=self.init_orientation)
        # 转换到base连杆初始位姿坐标系（init系）下表示: R_init_w = R_w_init的逆  P_init = R_init_w × P_w
        foot_com_position_init, _ = p.multiplyTransforms(positionA=foot_com_position_w, orientationA=[0, 0, 0, 1],
                                                         positionB=init_position_inv, orientationB=init_orientation_inv)
        return foot_frame_position_w, foot_velocity_w

    def get_foot_contact_state(self):
        """
        获取足端碰撞状态
        """
        all_contacts = p.getContactPoints(bodyA=self.robot_id)
        foot_contact = False
        for contact in all_contacts:
            if contact[2] == self.robot_id:  # 忽略自碰撞
                continue
            try:
                foot_contact = True
            except ValueError:
                continue
        print("足端是否触地: ", foot_contact)
        return foot_contact

    # ********************** 仿真环境设置函数 *****************************************************************
    def add_sliders(self):
        """
        函数功能: 在控制界面添加调试参数
        """
        self.CameraDistanceSlider = p.addUserDebugParameter('CameraDistanceSlider', 0, 3, 1.2)  # 视像头的距离 m
        self.CameraYawSlider = p.addUserDebugParameter('CameraYaw', -180, 180, 0)  # 视像头的偏航角 度
        self.CameraPitchSlider = p.addUserDebugParameter('CameraPitchSlider', -180, 180, 0)  # 视像头的俯仰角 度
        self.VelocityXSlider = p.addUserDebugParameter('X Velocity', -3, 3, 0)  # X向速度 m/s
        self.ThrustSlider = p.addUserDebugParameter('Thrust', 0, 6000, 2000)  # 推力 N
        self.CameraImageButton = p.addUserDebugParameter("getCameraImage", 1, 0, 0)  # 打开摄像头图像
        self.RemoveButton = p.addUserDebugParameter("RemoveAllUserDebugItems", 1, 0, 0)
        self.pre_remove_button = p.readUserDebugParameter(self.RemoveButton)

    def get_sliders_value(self):
        """
        读取滑块数据
        """
        cam_distance = p.readUserDebugParameter(self.CameraDistanceSlider)
        cam_yaw = p.readUserDebugParameter(self.CameraYawSlider)
        cam_pitch = p.readUserDebugParameter(self.CameraPitchSlider)
        velocity_x = p.readUserDebugParameter(self.VelocityXSlider)
        thrust = p.readUserDebugParameter(self.ThrustSlider)
        camera_image_flag = p.readUserDebugParameter(self.CameraImageButton)
        remove_button = p.readUserDebugParameter(self.RemoveButton)
        return cam_distance, cam_yaw, cam_pitch, velocity_x, thrust, camera_image_flag, remove_button

    def plot_trajectory_in_bullet(self, body_position_w_pre, body_position_w, foot_position_w_pre, foot_position_w):
        """
        函数功能: 在仿真环境中显示躯干质心和足端坐标系的位置轨迹
        """
        p.addUserDebugLine(lineFromXYZ=body_position_w_pre, lineToXYZ=body_position_w,
                           lineColorRGB=[1, 0, 0], lineWidth=2)  # show body trajectory
        p.addUserDebugLine(lineFromXYZ=foot_position_w_pre, lineToXYZ=foot_position_w,
                           lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=30)  # show foot trajectory

    def reset_camera(self, cam_dis, cam_yaw, cam_pitch, cam_pos):
        """
        设置相机的距离、朝向、位置
        cam_dis: 相机的距离 m
        cam_yaw: 相机的偏航角 度
        cam_pitch: 相机的俯仰角 度
        cam_pos: 相机位置 m
        """
        p.resetDebugVisualizerCamera(cameraDistance=cam_dis, cameraYaw=cam_yaw,
                                     cameraPitch=cam_pitch, cameraTargetPosition=cam_pos)

    def get_camera_image(self):
        """
        获取RGB图像、深度图和语义分割图
        """
        w, h, rgb_pixel, depth_pixel, seg_pixel = p.getCameraImage(800, 600)
        plt.figure(figsize=[12, 9])
        plt.subplot(2, 2, 1)
        plt.imshow(rgb_pixel)  # plt.imshow画出原有的一张图片
        plt.title("RGB Image")  # RGB原图
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.imshow(depth_pixel, cmap=plt.cm.gray)  # cmap为调整显示颜色, gray为黑白色
        plt.title("Depth Image")  # 深度图
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.imshow(seg_pixel)
        plt.title("Segmentation Mask Buffer")  # 语义分割图
        plt.axis("off")
        plt.show()

    def first_order_filter(self, nowIn, preOut):
        """
        函数功能：一阶滤波
        """
        filter_coefficient = 0.5  # 滤波系数
        return filter_coefficient * nowIn + (1 - filter_coefficient) * preOut

    def print_joint_info(self):
        """
        函数功能: 打印关节信息
        """
        print("关节信息：")
        for joint_index in range(self.joint_num_all):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            print(f"\
                    [0]关节索引: {joint_info[0]}\n\
                    [1]关节名称: {joint_info[1]}\n\
                    [2]关节类型: {joint_info[2]}\n\
                    [3]此主体的位置状态变量中的第一个位置索引: {joint_info[3]}\n\
                    [4]在这个物体的速度状态变量中的第一个速度索引: {joint_info[4]}\n\
                    [5]保留参数: {joint_info[5]}\n\
                    [6]关节阻尼大小: {joint_info[6]}\n\
                    [7]关节摩擦系数: {joint_info[7]}\n\
                    [8]平动或转动关节的位置下限: {joint_info[8]}\n\
                    [9]平动或转动关节的位置上限: {joint_info[9]}\n\
                    [10]关节最大力矩: {joint_info[10]}\n\
                    [11]关节最大速度: {joint_info[11]}\n\
                    [12]连杆名称: {joint_info[12]}\n\
                    [13]在当前连杆坐标系中表示的移动或转动的关节轴: {joint_info[13]}\n\
                    [14]在父连杆坐标系中表示的关节位置: {joint_info[14]}\n\
                    [15]在父连杆坐标系中表示的关节姿态(四元数x、y、z、w): {joint_info[15]}\n\
                    [16]父连杆的索引，若是base连杆则返回-1: {joint_info[16]}\n\n")

    def print_link_info(self):
        """
        函数功能: 打印连杆信息
        """
        print("连杆信息：")
        for link_index in range(-1, self.joint_num_all):
            link_info = p.getDynamicsInfo(self.robot_id, link_index)
            print(f"\
                    [0]质量: {link_info[0]}\n\
                    [1]横向摩擦系数(lateral friction): {link_info[1]}\n\
                    [2]主惯性矩: {link_info[2]}\n\
                    [3]惯性坐标系在局部关节坐标系中的位置: {link_info[3]}\n\
                    [4]惯性坐标系在局部关节坐标系中的姿态: {link_info[4]}\n\
                    [5]恢复系数: {link_info[5]}\n\
                    [6]滚动摩擦系数(rolling friction): {link_info[6]}\n\
                    [7]扭转摩擦系数(spinning friction): {link_info[7]}\n\
                    [8]接触阻尼(-1表示不可用): {link_info[8]}\n\
                    [9]接触刚度(-1表示不可用): {link_info[9]}\n\
                    [10]物体属性: 1=刚体，2=多刚体，3=软体: {link_info[10]}\n\
                    [11]碰撞边界: {link_info[11]}\n\n")
