import numpy as np

def conv3d():
    N = 2
    Di = 224
    Hi = 131
    Wi = 113
    Ci = 61
    Co = 78
    T = 5
    R = 7
    S = 4

    stride_d = 2
    stride_h = 2
    stride_w = 7

    dilation_d = 1
    dilation_h = 1
    dilation_w = 1

    pad_head = 0
    pad_tail = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    actual_t = (T - 1) * dilation_d + 1
    actual_r = (R - 1) * dilation_h + 1
    actual_s = (S - 1) * dilation_w + 1

    Do = (Di + pad_head + pad_tail - actual_t)/ stride_d + 1
    Ho = (Hi + pad_top + pad_bottom - actual_r)/ stride_h + 1
    Wo = (Wi + pad_left + pad_right - actual_s)/ stride_w + 1

    print("Do:", Do)
    print("Ho:", Ho)
    print("Wo:", Wo)

    print("input_shape should be: (%d, %d, %d, %d, %d)", N, Di, Hi, Wi, Ci)
    print("kernel_shape should be: (%d, %d, %d, %d, %d)", T, R, S, Ci, Co)
    print("output_shape should be: (%d, %d, %d, %d, %d)", N, Do, Ho, Wo, Co)

    inputs = np.random.randint(low = 1, high = 2, size = (N, Di, Hi, Wi, Ci)).astype('float32')
    kernel = np.random.randint(low = 1, high = 2, size = (T, R, S, Ci, Co)).astype('float32')

    print("inputs:", inputs)
    print("kernel:", kernel)

    print("inputs_shape:", inputs.shape)
    print("kernel_shape:", kernel.shape)


    # [n,d,ho,wo,co] = [1, 1, 1, 1, 32]

    n_loop_num = N
    do_loop_num = Do
    ho_loop_num = Ho
    wo_loop_num = Wo
    co_loop_num = (Co + 32 -1) / 32
    t_loop_num = T
    ci_loop_num = (Ci + 16 - 1) / 16

    outputs = np.random.randint(low = 0, high = 1, size = (N, Do, Ho, Wo, Co)).astype('float32')

    for n in range(n_loop_num):
        for d in range(do_loop_num):
            for h in range(ho_loop_num):
                for w in range(wo_loop_num):
                    for c in range(co_loop_num):
                        
                        t_beign = 0
                        t_end = t_loop_num
                        for t in range(t_beign, t_end):
                            for ci in range(ci_loop_num):    
                                cur_hi_offset = h * stride_h - pad_top
                                cur_wi_offset = w * stride_w - pad_left
                                cur_hi_offset_end = (1 + cur_hi_offset - 1) * stride_h + R - pad_top # 第一个1指的是一次计算输出的ho长度，这里是1
                                cur_wi_offset_end = (1 + cur_wi_offset - 1) * stride_w + S - pad_left
                                cur_hi_len = Hi - cur_hi_offset if cur_hi_offset_end > Hi - 1 else cur_hi_offset_end - cur_hi_offset + 1
                                cur_wi_len = Wi - cur_wi_offset if cur_wi_offset_end > Wi - 1 else cur_wi_offset_end - cur_wi_offset + 1

                                #hi_len = actual_r + (ho - 1) * stride_h;
                                #wi_len = actual_s + (wo - 1) * stride_w;
                                input = np.random.randint(low = 0, high = 1, size = (1, 1, actual_r, actual_s, 32)).astype('float32')
                                if (cur_hi_len > 0 & cur_wi_len > 0) :
                                    cur_input_data = inputs[n, d, cur_hi_offset : cur_hi_offset + cur_hi_len - 1, cur_wi_offset : cur_wi_offset + cur_wi_len - 1, ci] 
                                




if __name__ == "__main__":
    conv3d()

