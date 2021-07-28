# %%
import tensorflow as tf

# %%
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis = 0)

# %%
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis = -1)


