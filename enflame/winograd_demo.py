# -*- coding: utf-8 -*
"""
@Description: winograd convolution demo
@Author: jian.wu

Data: 2021/07/28 13:33
"""

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

def winograd_demo(input_shape, kernel_shape):
    ## TODO(jian.wu): check input param
    N, Hi, Wi, Ci = input_shape
    Hk, Wk, _, Co = kernel_shape
    assert Hi == 4, "Demo for Hi = 4"
    assert Wi == 4, "Demo for Wi = 4"

    # generate input and weight data
    input_data = np.random.uniform(size=input_shape, low=-1.0, high=1.0).astype("float32")
    kernel_data = np.random.uniform(size=kernel_shape, low=-1.0, high=1.0).astype("float32")

    # preprocess input
    preprocess_input_data = np.zeros([N, 4, 4, Ci], dtype="float32")
    Bt = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype="float32")
    for n in range(N):
        for ci in range(Ci):
            input = input_data[n, :, :, ci]  # ：表示切片，这里：，：表示4*4的矩阵切片
            preprocess_input_data[n, :, :, ci] = np.dot(np.dot(Bt, input), Bt.transpose())

    # preprocess kernel
    preprocess_kernel_data = np.zeros([4, 4, Ci, Co], dtype="float32")
    G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype="float32")
    for ci in range(Ci):
        for co in range(Co):
            input = kernel_data[:, :, ci, co]
            preprocess_kernel_data[:, :, ci, co] = np.dot(np.dot(G, input), G.transpose())

    # hadamard product
    hadamard_product_data = np.zeros([N, 4, 4, Co], dtype="float32")
    # for n in range(N):
    #     for co in range(Co):
    #         for ci in range(Ci):
    #             for h in range(4):
    #                 for w in range(4):
    #                     lhs = preprocess_input_data[n, h, w, ci]
    #                     rhs = preprocess_kernel_data[h, w, ci, co]
    #                     hadamard_product_data[n, h, w, co] += lhs * rhs
    for h in range(4):
        for w in range(4):
            lhs = preprocess_input_data[:, h, w, :]
            rhs = preprocess_kernel_data[h, w, :, :]
            hadamard_product_data[:, h, w, :] = np.dot(lhs, rhs)
    
    # postprocess output
    output_data = np.zeros([N, 2, 2, Co], dtype="float32")
    At = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype="float32")
    for n in range(N):
        for co in range(Co):
            input = hadamard_product_data[n, :, :, co]
            output_data[n, :, :, co] = np.dot(np.dot(At, input), At.transpose())

    ## expcect value is TF.covn2d
    with tf.device("cpu"):
        with tf.Session() as sess:
            input_tensor = tf.placeholder("float32", shape=input_shape)
            kernel_tensor = tf.placeholder("float32", shape=kernel_shape)
            net = tf.nn.conv2d(input_tensor, kernel_tensor, padding="VALID")
            feed_dict = {input_tensor: input_data, kernel_tensor: kernel_data}
            sess.run(tf.global_variables_initializer())
            tf_out = sess.run(net, feed_dict=feed_dict)

            np.testing.assert_allclose(output_data, tf_out, atol=1e-4, rtol=1e-4)
            print("check is OK")

if __name__ == "__main__":
    winograd_demo([128, 4, 4, 256], [3, 3, 256, 256])
