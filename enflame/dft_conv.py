# %%
import math
import numpy as np

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def display(msg=None, x=None):
    """
        display numpy array
    """
    print(msg)
    from IPython.display import display as dsp
    x = pd.DataFrame(x)
    dsp(x)


def to_complex(arr, n):
    """
        to numpy complex array
    """
    return arr[:, :n] + arr[:, n:] * 1j


def dft_coef():
    """
    output: (32, 32), (32, 16) in logical
            0~15  | 16~31
            [real | imag ]
    """
    coef_ = np.zeros((32, 32))

    # row [0-15]
    for n in range(16):
        for k in range(16):
            # real
            coef_[n][k] = np.cos(2 * np.pi * n * k / 31)
            # imag
            coef_[n][k + 16] = np.sin(-2 * np.pi * k * n / 31)

    # row 16 all zores

    # row [17-31]
    for n in range(15):
        for k in range(16):
            coef_[n + 17][k] = coef_[15 - n][k]
            coef_[n + 17][k + 16] = -coef_[15 - n][k + 16]
    return coef_


def dft_31x31_row(input_x, coef_):
    """
    input_x: (31, 31)
    coef_:   (31, 32), (32, 16) in logical
    output:  (31, 32), (31, 16) in logical
            0 ~ 15 | 16~31
            [real  | imag ]
    """
    row_dft_res = np.zeros((31, 32))

    for r in range(31):
        for k in range(16):  # only compute half
            real, imag = 0, 0

            for n in range(16):
                real += input_x[r, n] * coef_[n, k]
            for n in range(16):
                imag += input_x[r, n] * coef_[n, k + 16]

            for n in range(15):
                real += input_x[r, n + 16] * coef_[n + 17, k]
            for n in range(15):
                imag += input_x[r, n + 16] * coef_[n + 17, k + 16]

            row_dft_res[r, k] = real
            row_dft_res[r, k + 16] = imag

    return row_dft_res


def dft_31x31_col(input_x, coef_):
    """
    input_x: (31, 32), (31, 16) in logical
    coef_:   (31, 32), (32, 16) in logical
    output:  (16, 62), (16, 31) in logical
            0 ~ 15 | 16~31
            [real  | imag ]
    """
    col_dft_res = np.zeros((16, 62))
    for c in range(16):
        for k in range(16):
            real_ac, imag_ad = 0, 0
            real_bd, imag_bc = 0, 0
            for n in range(16):  # real_ac part sum
                real_ac += input_x[n, c] * coef_[n, k]
            for n in range(16):  # imag_ad part sum
                imag_ad += input_x[n, c] * coef_[n, k + 16]

            for n in range(15):  # finish real_ac sum
                real_ac += input_x[n + 16, c] * coef_[n + 17, k]
            for n in range(15):  # finish imag_ad sum
                imag_ad += input_x[n + 16, c] * coef_[n + 17, k + 16]

            for n in range(16):  # real_bd part sum
                real_bd += input_x[n, c + 16] * coef_[n, k + 16]
            for n in range(16):  # imag_bc part sum
                imag_bc += input_x[n, c + 16] * coef_[n, k]

            for n in range(15):  # finish real_bd sum
                real_bd += input_x[n + 16, c + 16] * coef_[n + 17, k + 16]
            for n in range(15):  # finish imag_bc sum
                imag_bc += input_x[n + 16, c + 16] * coef_[n + 17, k]

            col_dft_res[k, c] = real_ac - real_bd
            col_dft_res[k, c + 31] = imag_ad + imag_bc

            if c > 0:
                col_dft_res[k, 31 - c] = real_ac + real_bd
                col_dft_res[k, 31 - c + 31] = imag_ad - imag_bc
    return col_dft_res


def idft_31x31_row(input_x, coef_):
    """
    input_x: (16, 62), (16, 31) in logical
    coef_:   (31, 32), (32, 16) in logical
    output:  (16, 62), (16, 31) in logical
    """
    row_idft_res = np.zeros((16, 62))
    for r in range(16):
        for k in range(16):
            real_ac, imag_ad = 0, 0
            real_bd, imag_bc = 0, 0

            for n in range(16):  # real_ac part sum
                real_ac += input_x[r, n] * coef_[n, k]
            for n in range(16):  # imag_ad part sum
                imag_ad += input_x[r, n] * coef_[n, k + 16]

            for n in range(15):  # finish real_ac sum
                real_ac += input_x[r, n + 16] * coef_[n + 17, k]
            for n in range(15):  # finish imag_ad sum
                imag_ad += input_x[r, n + 16] * coef_[n + 17, k + 16]

            for n in range(16):  # real_bd part sum
                real_bd += input_x[r, n + 31] * coef_[n, k + 16]
            for n in range(16):  # imag_bc part sum
                imag_bc += input_x[r, n + 31] * coef_[n, k]

            for n in range(15):  # finish real_bd sum
                real_bd += input_x[r, n + 16 + 31] * coef_[n + 17, k + 16]
            for n in range(15):  # finish imag_bc sum
                imag_bc += input_x[r, n + 16 + 31] * coef_[n + 17, k]

            row_idft_res[r, k] = real_ac + real_bd
            row_idft_res[r, k + 31] = -imag_ad + imag_bc

            if k > 0:
                row_idft_res[r, 31 - k] = real_ac - real_bd
                row_idft_res[r, 31 - k + 31] = -(-imag_ad - imag_bc)
                # row_idft_res[r, 31 - k + 31] = (-imag_ad - imag_bc)
    return row_idft_res


def idft_31x31_col(input_x, coef_):
    """
    input_x:  (16, 62), (16, 31) in logical
    coef_:    (32, 32), (32, 16) in logical
    output:   (31, 31)
    """
    col_idft_res = np.zeros((31, 31))
    for c in range(31):
        for k in range(16):
            real_ac, real_bd = 0, 0
            for n in range(16):  # real_ac part sum
                real_ac += input_x[n, c] * coef_[n, k]
            for n in range(15):  # finish real_ac part sum
                real_ac += input_x[15 - n, c] * coef_[n + 17, k]
            for n in range(16):  # real_bd
                real_bd += input_x[n, c + 31] * coef_[n, k + 16]
            for n in range(15):
                real_bd += -input_x[15 - n, c + 31] * coef_[n + 17, k + 16]

            col_idft_res[k, c] = real_ac + real_bd
            if(k > 0):
                col_idft_res[31 - k, c] = real_ac - real_bd

    return col_idft_res / 961


def dft_3x3_row_reverse(input_x, coef_):
    """
    input_x: (31, 31)
            __________
            |        |
            |0-pad___|
            |    |3x3| 
            |____|___|

    output: (31, 32), (31, 16) in logical
            0 ~ 15 | 16~31
            [real  | imag ]
    """
    row_dft_res = np.zeros((31, 32))

    for r in range(3):
        for k in range(16):
            real, imag = 0, 0
            for n in range(16):
                real += input_x[30 - r, 30 - n] * coef_[n, k]
                imag += input_x[30 - r, 30 - n] * coef_[n, k + 16]

            row_dft_res[r, k] = real
            row_dft_res[r, k + 16] = imag

    return row_dft_res

def dft_3x3_col(input_x, coef_):
    """
    input_x: (31, 32), (31, 16) in logical
            0~15  16~31
            [real | imag ]

            Note: there're so many zeros in logical input
            _______
            |_3x16_|
            |      |
            |  0   |
            |______|

    output: (16, 62), (16, 31) in logical
            0 ~ 30 | 31 ~ 61
            [real  | imag ]
    """
    col_dft_res = np.zeros((16, 62))
    for c in range(16): # only compute half
        for k in range(16):
            real_ac, imag_ad, real_bd, imag_bc = 0, 0, 0, 0
            # only need to caluate 0-3

            for n in range(16): # finish real_ac sum
                real_ac += input_x[n, c] * coef_[n, k]
            for n in range(16): # finish imag_ad sum
                imag_ad += input_x[n, c] * coef_[n, k + 16]
            for n in range(16): # finish real_bd sum
                real_bd += input_x[n, c + 16] * coef_[n, k + 16]
            for n in range(16): # finish imag_bc sum
                imag_bc += input_x[n, c + 16] * coef_[n, k]
            
            col_dft_res[k, c] = real_ac - real_bd
            col_dft_res[k, c + 31] = imag_ad + imag_bc
            if c > 0:
                col_dft_res[k, 31 - c] = real_ac + real_bd
                col_dft_res[k, 31 - c + 31] = imag_ad - imag_bc
    return col_dft_res


def dft_31x31(input_x, coef_):
    """
    input_x: (31, 31)
    """
    dft_row_x = dft_31x31_row(input_x, coef_)
    dft_col_x = dft_31x31_col(dft_row_x, coef_)

    return dft_col_x


def dft_3x3_reverse(input_x, coef_):
    """
    input_x: (31, 31)
    """
    dft_row_x = dft_3x3_row_reverse(input_x, coef_)
    dft_col_x = dft_3x3_col(dft_row_x, coef_)

    return dft_col_x


def idft_31x31(input_x, coef_):
    """
    input_x: (16, 62), (16, 31) in logical
            0 ~ 30 | 31 ~ 61
            [real  | imag ]
    """
    idft_row_x = idft_31x31_row(input_x, coef_)
    idft_col_x = idft_31x31_col(idft_row_x, coef_)

    return idft_col_x

def hadamard_product(dft_input_x, dft_kernel):
    """
    Hadamard product
    input: (16, 62)
            logically (16, 31)
            0 ~ 30 | 31 ~ 61
            [real  | imag ]

    output: (16, 62)
            logically (16, 31)
            0 ~ 30 | 31 ~ 61
            [real  | imag ]
    """
    prod_res = np.zeros((16, 62))
    for r in range(16):
        for c in range(31):
            real = dft_input_x[r][c] * dft_kernel[r][c] - \
                dft_input_x[r][c + 31] * dft_kernel[r][c + 31]
            imag = dft_input_x[r][c] * dft_kernel[r][c + 31] + \
                dft_input_x[r][c + 31] * dft_kernel[r][c]
            prod_res[r][c] = real
            prod_res[r][c + 31] = imag
    return prod_res

def dft_product_29x29_3x3_2d(pad_x, kernel, coef_):

    # padding
    # pad_x = np.zeros((31, 31))
    # pad_x[0:29, 0:29] = input_x
    pad_kernel = np.zeros((31, 31))
    pad_kernel[28:, 28:] = kernel
    # dft
    input_x_dft = dft_31x31(pad_x, coef_)
    kernel_dft = dft_3x3_reverse(pad_kernel, coef_)
    # hadamard product
    res = hadamard_product(input_x_dft, kernel_dft)
  
    return res


def dft_conv(input_x, kernel, coef_):

    N, H, W, Ci = input_x.shape
    Hk, Wk, Ci, Co = kernel.shape

    conv_res = np.zeros((N, H, W, Co))

    block_h = (H - 29) // 27 + (H % 27 > 0) + 1
    block_w = (W - 29) // 27 + (W % 27 > 0) + 1
    print("block_h:", block_h)
    print("block_w:", block_w)
   
    # for test
    for n in range(N):
        for co in range(Co):

            input_h_offset, input_h_offset_end = 0, 0
            input_w_offset, input_w_offset_end = 0, 0

            output_h_offset, output_w_offset = 0, 0
            output_h_offset_end , output_w_offset_end = 0, 0

            for bh in range(block_h):
                for bw in range(block_w):
                    
                    # input offset
                    input_h_offset = bh * 27
                    input_h_offset_end = input_h_offset + 29
                    input_w_offset = bw * 27
                    input_w_offset_end = input_w_offset + 29

                    # output offset
                    if bh == 0:
                        output_h_offset = 0
                    else:
                        output_h_offset = (bh - 1) * 27 + 28

                    if bw == 0:
                        output_w_offset = 0  
                    else:
                        output_w_offset = (bw - 1) * 27 + 28

                    if bh > 0 and bh < block_h - 1:
                        output_h_offset_end = output_h_offset + 27
                    else:
                        output_h_offset_end = output_h_offset + 28

                    if bw > 0 and bw < block_w - 1:
                        output_w_offset_end = output_w_offset + 27
                    else:
                        output_w_offset_end = output_w_offset + 28

                    ci_res = np.zeros((16, 62))
                    
                    for ci in range(Ci):
                        x = input_x[n, input_h_offset: input_h_offset_end, input_w_offset: input_w_offset_end, ci]
                        h_x, w_x = x.shape
                        pad_x = np.zeros((31, 31))
                        pad_x[0:h_x, 0:w_x] = x
                        kernel_ci = kernel[:, :, ci, co]
                        res = dft_product_29x29_3x3_2d(pad_x, kernel_ci, coef_)
                        ci_res += res

                    # idft
                    ci_idft_res = idft_31x31(ci_res, coef_)
                    ci_idft_res = ci_idft_res[1:30, 1:30]

                    # valid output part
                    h_ci, w_ci = ci_idft_res.shape
                    if bh > 0:
                        ci_idft_res = ci_idft_res[1:, :]
                    if bh < block_h - 1:
                        h_ci, _ = ci_idft_res.shape
                        ci_idft_res = ci_idft_res[:h_ci - 1, :]
 
                    if bw > 0:
                        ci_idft_res = ci_idft_res[:, 1:]
                    if bw < block_w - 1:
                        _, w_ci = ci_idft_res.shape
                        ci_idft_res = ci_idft_res[:, :w_ci - 1]

                    if bh == block_h - 1:
                        ci_idft_res = ci_idft_res[:H - input_h_offset - 1, :]
                    
                    if bw == block_w - 1:
                        ci_idft_res = ci_idft_res[:, :W - input_w_offset - 1]

                    conv_res[n, output_h_offset: output_h_offset_end, output_w_offset: output_w_offset_end, co] = ci_idft_res
 
    return conv_res       

if __name__ == '__main__':

    import tensorflow as tf
    # [N, H, W, Ci]
    input_x = np.random.randn(1, 32, 29, 3)
    # [H, W, Ci, Co]
    kernel = np.random.randn(3, 3, 3, 4)

    coef_ = dft_coef()
    # display("coef_", to_complex(coef_, 16))
    
    conv_res = dft_conv(input_x, kernel, coef_)  
    tf_conv_res = tf.nn.conv2d(input_x, kernel, strides=1, padding='SAME').numpy()    
    tf_compare = ((tf_conv_res[:, :, :, :] - conv_res[:, :, :, :]) > 1e-5).astype(int)
    print('fail count:', tf_compare.sum())
    np.testing.assert_allclose(conv_res, tf_conv_res)
    print('ok')
# %%

