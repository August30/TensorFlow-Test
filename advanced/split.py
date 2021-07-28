# %%
import tensorflow as tf

# %%
x = tf.random.normal([10,35,8])
result =tf.split(x, num_or_size_splits = 10, axis = 0)
len(result)

# %%
result[0]

# %%
x = tf.random.normal([10,35,8])
result =tf.split(x, num_or_size_splits = [4,2,2,2], axis = 0)
len(result)

# %%
result[0]

# %%
x = tf.random.normal([10,35,8])
result =tf.unstack(x, axis = 0)
len(result)

# %%
result[0]