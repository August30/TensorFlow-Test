import numpy as np
# from numpy.core.fromnumeric import shape
# import tensorflow.compat.v1 as tf
import tensorflow as tf

def conv3d():
    N = 2
    Di = 9
    Hi = 9
    Wi = 9
    Ci = 17
    Co = 33
    T = 3
    R = 3
    S = 3

    stride_d = 1
    stride_h = 1
    stride_w = 1

    dilation_d = 1 #目前做的 dilation_d只能等于1
    dilation_h = 1
    dilation_w = 1

    # SAME 
    # pad_head = 1
    # pad_tail = 1
    # pad_top = 1
    # pad_bottom = 1
    # pad_left = 1
    # pad_right = 1

    # VALID
    pad_head = 0
    pad_tail = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    actual_t = (T - 1) * dilation_d + 1
    actual_r = (R - 1) * dilation_h + 1
    actual_s = (S - 1) * dilation_w + 1

    Do = int((Di + pad_head + pad_tail - actual_t)/ stride_d + 1)
    Ho = int((Hi + pad_top + pad_bottom - actual_r)/ stride_h + 1)
    Wo = int((Wi + pad_left + pad_right - actual_s)/ stride_w + 1)

    print("Do:", Do)
    print("Ho:", Ho)
    print("Wo:", Wo)

    print("input_shape should be: (%d, %d, %d, %d, %d)", N, Di, Hi, Wi, Ci)
    print("kernel_shape should be: (%d, %d, %d, %d, %d)", T, R, S, Ci, Co)
    print("output_shape should be: (%d, %d, %d, %d, %d)", N, Do, Ho, Wo, Co)

    inputs = np.random.randint(low = 1, high = 2, size = (N, Di, Hi, Wi, Ci)).astype("float32")
    kernels = np.random.randint(low = 1, high = 2, size = (T, R, S, Ci, Co)).astype("float32")

    # print("inputs:", inputs)
    # print("kernels:", kernels)

    print("inputs_shape:", inputs.shape)
    print("kernels_shape:", kernels.shape)


    # [n,d,ho,wo,co] = [1, 1, 1, 1, 32]
    # [n,d,hi,wi,ci] = [1, 1, actual_r + (ho - 1) * stride_h, actual_s + (wo - 1) * stride_w, 16] = [1, 1, actual_r, actual_s, 16]

    n_loop_num = N
    do_loop_num = Do
    ho_loop_num = Ho
    wo_loop_num = Wo
    co_loop_num = int((Co + 32 -1) / 32)
    t_loop_num = T
    ci_loop_num = int((Ci + 16 - 1) / 16)
    print("n_loop_num:", n_loop_num)
    print("do_loop_num:", do_loop_num)
    print("ho_loop_num:", ho_loop_num)
    print("wo_loop_num:", wo_loop_num)
    print("co_loop_num:", co_loop_num)
    print("t_loop_num:", t_loop_num)
    print("ci_loop_num:", ci_loop_num)

    outputs = np.zeros((N, Do, Ho, Wo, Co), dtype="float32")

    for n in range(n_loop_num):
        for d in range(do_loop_num):
            for h in range(ho_loop_num):
                for w in range(wo_loop_num):
                    for c in range(co_loop_num):

                        co_len = Co - c*32 if (c+1)*32 > Co else 32
                        
                        output_data = np.zeros((1, 1, 1, 1, 32), dtype="float32")

                        # t_beign = 0
                        # t_end = t_loop_num

                        # skip the head block
                        t_beign = (pad_head - d * stride_d) if d * stride_d < pad_head else 0
                        # drop the tail block
                        t_end = (t_loop_num - pad_tail + (do_loop_num - d - 1) * stride_d) if (do_loop_num - d - 1) * stride_d < pad_tail else t_loop_num

                        for t in range(t_beign, t_end):
                            for ci in range(ci_loop_num):    
                                cur_di_offset = d * stride_d - pad_head
                                cur_hi_offset = h * stride_h - pad_top
                                cur_wi_offset = w * stride_w - pad_left
                                # cur_hi_offset_end = (1 + h*1 - 1) * stride_h + R - pad_top - 1 # 第一个1指的是一次计算输出的ho长度，这里是1;h*1,这个1指的也是一次计算输出ho的长度, 应该是actual_r 和 actual_s，感觉这里R和S应该是错了
                                # cur_wi_offset_end = (1 + w*1 - 1) * stride_w + S - pad_left - 1
                                cur_hi_offset_end = (1 + h*1 - 1) * stride_h + actual_r - pad_top - 1 # 第一个1指的是一次计算输出的ho长度，这里是1;h*1,这个1指的也是一次计算输出ho的长度
                                cur_wi_offset_end = (1 + w*1 - 1) * stride_w + actual_s - pad_left - 1
                                cur_hi_len = Hi - cur_hi_offset if cur_hi_offset_end > Hi - 1 else cur_hi_offset_end - cur_hi_offset + 1
                                cur_wi_len = Wi - cur_wi_offset if cur_wi_offset_end > Wi - 1 else cur_wi_offset_end - cur_wi_offset + 1

                                #hi_len = actual_r + (ho - 1) * stride_h;
                                #wi_len = actual_s + (wo - 1) * stride_w;
                                input = np.zeros((1, 1, actual_r, actual_s, 16), dtype="float32")
                                kernel = np.zeros((1, R, S, 16, 32), dtype="float32")

                                # print("cur_hi_len:", cur_hi_len)
                                # print("cur_wi_len:", cur_wi_len)
                                # print("cur_di_offset:", cur_di_offset)
                                # print("cur_hi_offset:", cur_hi_offset)
                                # print("cur_hi_offset_end:", cur_hi_offset_end)
                                # print("t:", t)

                                # 如果t_beign = 0，t_end = t_loop_num时可用下面的if判断，这样效率低一些，skip the head block 和  drop the tail block 可以少一些循环
                                # if (cur_hi_len > 0 and cur_wi_len > 0 and cur_di_offset + t >= 0 and cur_di_offset + t < Di) :

                                if (cur_hi_len > 0 and cur_wi_len > 0 and cur_di_offset + t >= 0 and cur_di_offset + t < Di) :

                                    hi_cut_offset =  0 if cur_hi_offset < 0 else  cur_hi_offset
                                    hi_cut_offset_end = Hi - 1 if cur_hi_offset_end > Hi - 1 else cur_hi_offset_end

                                    wi_cut_offset = 0 if cur_wi_offset < 0 else  cur_wi_offset
                                    wi_cut_offset_end = Wi - 1 if cur_wi_offset_end > Wi - 1 else cur_wi_offset_end

                                    ci_len = 16 if (ci+1)*16 <= Ci else Ci - ci*16

                                    
                                    # print("cur_di_offset:", cur_di_offset)
                                    # print("t:", t)
                                    # print("cur_di_offset+t:", cur_di_offset+t)
                                    
                                    # 注意：python中切片右边是一个开区间，factor中的endoffset是一个闭区间
                                    cut_input_data = inputs[n, cur_di_offset+t, hi_cut_offset:hi_cut_offset_end + 1, wi_cut_offset:wi_cut_offset_end + 1, ci*16:ci*16+ci_len]
                                    cut_input_data = cut_input_data[np.newaxis, np.newaxis, :, :, :]

                                    split_h_begin = -cur_hi_offset if cur_hi_offset < 0 else 0
                                    split_h_end = split_h_begin + cur_hi_len 
                                    split_w_begin = -cur_wi_offset if cur_wi_offset < 0 else 0
                                    split_w_end = split_w_begin + cur_wi_len 
                                   
                                    input[:, :, split_h_begin:split_h_end, split_w_begin:split_w_end, :ci_len] = cut_input_data
                                    # np.pad(cur_input_data) https://blog.csdn.net/Tan_HandSome/article/details/80296827

                                    kernel[:, :, :, :ci_len, :co_len] = kernels[t, :, :, ci*16:ci*16+ci_len, c*32:c*32+co_len]

                                
                                # 计算
                                # 1.根据dilation将kernel pad 0, 也可以根据dilation间隔取input数据
                                input = input[:, :, ::dilation_h, ::dilation_w, :]
                                # 2.点成相加，co有32个，做32次，输出一个1*32的向量
                                output = np.zeros((1, 1, 1, 1, 32), dtype="float32")
                                for i in range(32):
                                    output[:,:,:,:,i] = np.sum(input.reshape(R, S, 16) * kernel[:,:,:,:,i].reshape(R, S, 16))
                                output_data += output
                                # 3.将1*32的向量reshape为(1,1,1,1,32),放到outputs对应位置
                                # print("output_data:", output_data)
                        
                        # print("Co:", Co)
                        # print("c:", c)
                        # print("co_len:", co_len)
                        # print(outputs[n, d, h, w, c*32:c*32+co_len].shape, output_data[:, :, :, :, :co_len].shape)
                        outputs[n, d, h, w, c*32:c*32+co_len] = output_data[:, :, :, :, :co_len]

    # print("outputs: ", outputs)
    print("outputs_shape: ", outputs.shape)
    outputs = outputs.transpose(0, 4, 1, 2, 3)
    print("outputs_transpose: ", outputs)
    print("outputs_transpose_shape: ", outputs.shape)

    # tensorflow 1.x
    # with tf.device("cpu"):
    #     with tf.Session() as sess:
    #         tf_input = tf.placeholder("float32", shape = inputs.shape)
    #         tf_kernel = tf.placeholder("float32", shape = kernels.shape)
    #         conv = tf.nn.conv3d(tf_input, tf_kernel, strides = [1, stride_d, stride_h, stride_w, 1], padding = "SAME")
    #         # conv = tf.nn.conv3d(tf_input, tf_kernel, strides = [1, stride_d, stride_h, stride_w, 1], padding = "VALID", dilations = [1, dilation_d, dilation_h, dilation_w, 1])
    #         feed_dict = {tf_input:inputs, tf_kernel:kernels}
    #         sess.run(tf.global_variables_initializer())
    #         tf_out = sess.run(conv, feed_dict = feed_dict)
    #         cpu_in_man = tf_out.transpose(0, 4, 1, 2, 3)
    #         print("conv3d_shape:cpu_output", cpu_in_man.shape)
    #         np.testing.assert_allclose(outputs, cpu_in_man, atol=1e-3, rtol=1e-3)
    #         print("check is OK")

    # tensorflow 2.x
    tf_input = inputs
    tf_kernel = kernels
    cpu_tf_out = tf.nn.conv3d(tf_input, tf_kernel, strides = [1, stride_d, stride_h, stride_w, 1], padding = "VALID")
    # cpu_tf_out = tf.nn.conv3d(tf_input, tf_kernel, strides = [1, stride_d, stride_h, stride_w, 1], padding = "VALID", dilations = [1, dilation_d, dilation_h, dilation_w, 1])
    cpu_tf_out = tf.transpose(cpu_tf_out, perm=[0, 4, 1, 2, 3])
    print("conv3d_shape:cpu_output", cpu_tf_out.shape)
    np.testing.assert_allclose(outputs, cpu_tf_out, atol=1e-3, rtol=1e-3)
    print("check is OK")

                



if __name__ == "__main__":
    conv3d()

# %%
# import numpy as  np
# a = np.zeros((5, 5))
# b = np.ones((3, 3))
# a[2:, 2:] = b
# print(a)
# %%
