import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
# 定义直线的起点和终点
x = [0, 1]
y = [0, 1]
z = [0, 1]

# 画直线
ax.plot(x, y, z, label='直线')

# 添加标签和标题
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
ax.set_title('三维直线示例')

# 显示图例
ax.legend()

# 显示图形
plt.show()
