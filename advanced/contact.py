# %%
import tensorflow as tf


# %%
a = tf.random.normal([4,35,8])
b = tf.random.normal([6,35,8])
tf.concat([a,b], axis=0)

