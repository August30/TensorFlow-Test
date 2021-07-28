# %%
import tensorflow as tf

# %%
x = tf.linspace(-8.,8,100)
y = tf.linspace(-8.,8,100)
x,y = tf.meshgrid(x,y)
x.shape,y.shape

# %%
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z

# %%
import matplotlib
from matplotlib import pyplot as plt
# 导入三维坐标轴支持
from mpl_toolkits.mplot3d import Axes3D

# %%
fig = plt.figure()
ax = Axes3D(fig)
# 根据网格点绘制sin函数的三维曲面
ax.contour3D(x.numpy(),y.numpy(),z.numpy(),50)
plt.show()
# %%
