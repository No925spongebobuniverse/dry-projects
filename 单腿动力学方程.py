import sympy as sp

# 输出设置
sp.init_printing(use_latex=True)
# --------------------------------
# 变量定义
# 杆件长度
l1, l2, l3 = sp.symbols('l1 l2 l3')
# 杆件质心
lc1, lc2, lc3 = sp.symbols('lc1 lc2 lc3')
# 杆件宽度
w1, w2, w3 = sp.symbols('w1 w2 w3')
# 杆件厚度
h1, h2, h3 = sp.symbols('h1 h2 h3')
# 杆件质量
m1, m2, m3 = sp.symbols('m1 m2 m3')
m = [m1, m2, m3]
# 足端力
Fx, Fy, Fz = sp.symbols('Fx Fy Fz')
F = sp.Matrix([[Fx], [Fy], [Fz]])
# 关节角位置/速度/加速度
theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3')
theta1_v, theta2_v, theta3_v = sp.symbols('theta1_v theta2_v theta3_v')
theta1_a, theta2_a, theta3_a = sp.symbols('theta1_a theta2_a theta3_a')
q = [theta1, theta2, theta3]
q_dot1 = [theta1_v, theta2_v, theta3_v]
q_dot2 = [theta1_a, theta2_a, theta3_a]
# 重力加速度向量
g = sp.Symbol('g')
g_T = sp.Matrix([0, 0, -g]).T
# 符号变量 左腿 k=1 右腿 k=-1
k = sp.Symbol('k')


def Rx(theta):
    """
    旋转矩阵 x
    """
    return sp.Matrix([[1,             0,              0],
                      [0, sp.cos(theta), -sp.sin(theta)],
                      [0, sp.sin(theta), sp.cos(theta)]])


def Ry(theta):
    """
    旋转矩阵 y
    """
    return sp.Matrix([[sp.cos(theta),  0, sp.sin(theta)],
                      [0,              1,             0],
                      [-sp.sin(theta), 0, sp.cos(theta)]])


def Rz(theta):
    """
    旋转矩阵 z
    """
    return sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                      [sp.sin(theta), sp.cos(theta),  0],
                      [0,                         0,  1]])


def rotation_matrix_x(theta):
    """
    齐次变换矩阵 旋转 x
    """
    return sp.Matrix([[1,             0,              0, 0],
                      [0, sp.cos(theta), -sp.sin(theta), 0],
                      [0, sp.sin(theta), sp.cos(theta), 0],
                      [0,             0,              0, 1]])


def rotation_matrix_y(theta):
    """
    齐次变换矩阵 旋转 y
    """
    return sp.Matrix([[sp.cos(theta),  0, sp.sin(theta), 0],
                      [0,              1,             0, 0],
                      [-sp.sin(theta), 0, sp.cos(theta), 0],
                      [0,             0,              0, 1]])


def rotation_matrix_z(theta):
    """
    齐次变换矩阵 旋转 z
    """
    return sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0, 0],
                      [sp.sin(theta), sp.cos(theta),  0, 0],
                      [0,                         0,  1, 0],
                      [0,             0,              0, 1]])


def translation_matrix(lx, ly, lz):
    """
    齐次变换矩阵 平移 x y z
    """
    return sp.Matrix([[1, 0, 0, lx],
                      [0, 1, 0, ly],
                      [0, 0, 1, lz],
                      [0, 0, 0,  1]])


#  齐次变换矩阵 到杆件的末端
T00 = sp.Matrix([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
T01 = rotation_matrix_x(theta1) @ translation_matrix(0, k*l1, 0)
T12 = rotation_matrix_y(theta2) @ translation_matrix(l2, 0, 0)
T23 = rotation_matrix_y(theta3) @ translation_matrix(l3, 0, 0)
T02 = sp.simplify(T01 @ T12)
T03 = sp.simplify(T01 @ T12 @ T23)
print("T01", T01)
print("T12", T12)
print("T23", T23)
print("T03", T03)
#  齐次变换矩阵 到杆件的质心
T01c = rotation_matrix_x(theta1) @ translation_matrix(0, k*lc1, 0)
T12c = rotation_matrix_y(theta2) @ translation_matrix(lc2, 0, 0)
T23c = rotation_matrix_y(theta3) @ translation_matrix(lc3, 0, 0)
T02c = sp.simplify(T01 @ T12c)
T03c = sp.simplify(T01 @ T12 @ T23c)
print("T01c", T01c)
print("T02c", T02c)
print("T03c", T03c)

# 坐标系之间的旋转变换矩阵
R01 = T01[:3, :3]
R12 = T12[:3, :3]
R23 = T23[:3, :3]
R02 = T02[:3, :3]
R03 = T03[:3, :3]

# 雅可比矩阵 偏微分法
J_v1 = sp.Matrix([[sp.diff(T01c[0, 3], theta1), sp.diff(T01c[0, 3], theta2), sp.diff(T01c[0, 3], theta3)],
                  [sp.diff(T01c[1, 3], theta1), sp.diff(T01c[1, 3], theta2), sp.diff(T01c[1, 3], theta3)],
                  [sp.diff(T01c[2, 3], theta1), sp.diff(T01c[2, 3], theta2), sp.diff(T01c[2, 3], theta3)]])

J_v2 = sp.Matrix([[sp.diff(T02c[0, 3], theta1), sp.diff(T02c[0, 3], theta2), sp.diff(T02c[0, 3], theta3)],
                  [sp.diff(T02c[1, 3], theta1), sp.diff(T02c[1, 3], theta2), sp.diff(T02c[1, 3], theta3)],
                  [sp.diff(T02c[2, 3], theta1), sp.diff(T02c[2, 3], theta2), sp.diff(T02c[2, 3], theta3)]])

J_v3 = sp.Matrix([[sp.diff(T03c[0, 3], theta1), sp.diff(T03c[0, 3], theta2), sp.diff(T03c[0, 3], theta3)],
                  [sp.diff(T03c[1, 3], theta1), sp.diff(T03c[1, 3], theta2), sp.diff(T03c[1, 3], theta3)],
                  [sp.diff(T03c[2, 3], theta1), sp.diff(T03c[2, 3], theta2), sp.diff(T03c[2, 3], theta3)]])
J_v3 = sp.simplify(J_v3)
J_v = [J_v1, J_v2, J_v3]
print("J_v1", J_v1)
print("J_v2", J_v2)
print("J_v3", J_v3)

J_w1 = sp.Matrix([[0, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0]])
J_w2 = sp.Matrix([[0, 0, 0],
                  [0, 0, 0],
                  [1, 1, 0]])
J_w3 = sp.Matrix([[0, 0, 0],
                  [0, 0, 0],
                  [1, 1, 1]])
J_w = [J_w1, J_w2, J_w3]


def inertia_tensor(x, y, z, m):
    """
    惯性张量
    """
    return sp.Matrix([[m / 12 * (y ** 2 + z ** 2), 0, 0],
                      [0, m / 12 * (x ** 2 + z ** 2), 0],
                      [0, 0, m / 12 * (x ** 2 + y ** 2)]])


I11 = inertia_tensor(w1, l1, h1, m1)
I22 = inertia_tensor(l2, h2, w2, m2)
I33 = inertia_tensor(l3, h3, w3, m3)
I1 = R01 @ I11
I2 = R02 @ I22
I3 = R03 @ I33
I = [I1, I2, I3]

# ----------------------------------------------------------------
# 动力学分项

# 第一项：质量矩阵
M = sp.zeros(3, 3)
for i in range(3):
    # print(J_v[i].T)
    # print(J_v[i].T * m[i] @ J_v[i])
    M += J_v[i].T * m[i] @ J_v[i] + J_w[i].T @ I[i] @ J_w[i]

# 第二项：速度耦合向量
V = sp.zeros(3, 1)
for i in range(3):
    for j in range(3):
        for k in range(3):
            V[i] += (sp.diff(M[i, j], q[k]) - 1 / 2 * sp.diff(M[j, k], q[i])) * q_dot1[j] * q_dot1[k]
# for i in range(3):
#     print("V[{}] = {}".format(i, V[i]))

# 第三项：重力向量
G = sp.zeros(3, 1)
for i in range(3):
    for j in range(3):
        G[i] += (-(m[j] * g_T @ J_v[j][:, i]))[0, 0]
for i in range(3):
    print("G[{}] = {}".format(i, G[i]))

# ----------------------------------------------------------------------
# 动力学方程
Q = sp.zeros(3, 1)
for i in range(3):
    for j in range(3):
        Q[i] += M[i, j] * q_dot2[j] + 1 / 3 * V[i] + 1 / 3 * G[i]

# for i in range(3):
#     print("Q[{}] = {}".format(i, Q[i]))

T = sp.zeros(3, 1)
T = Q - J_v3.T @ F
# for i in range(3):
#     print("T[{}] = {}".format(i, T[i]))
