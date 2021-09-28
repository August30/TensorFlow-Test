# %%
import numpy as np
from numpy.lib.function_base import rot90
import pandas as pd
from IPython.display import display as dsp


def show(msg=None, x=None):
    print(msg)
    x = pd.DataFrame(x)
    dsp(x)

def toComplex(input,index):
    return input[:,:index] + input[:,index:]*1j

def param():
    res = np.zeros((16,32))
    for k in range(16):
        for n in range(16):
            res[k,n] = np.cos(2 * np.pi * n * k / 31)
            res[k,n + 16] = np.sin(-2 * np.pi * k * n / 31)
    return res  

def row_dft(input, param):
    row_res = np.zeros((31, 32))
    for r in range(31):
        for k in range(16):
            ## n == 0
            real, imag = 0, 0
            real += param[0, k] * input[r, 0]
            # n ~ [1, 31]
            for n in range(1, 16):
                # 1 ~ 15
                real += input[r, n] * param[n, k]
                imag += input[r, n] * param[n, k + 16]
                # 16 ~ 31
                real += input[r, 31 - n] * param[n, k]
                imag += input[r, 31 - n] * (-param[n, k + 16])

            row_res[r, k] = real
            row_res[r, k + 16] = imag
    return row_res


def col_dft(input, param):

    col_res = np.zeros((16, 62))
    for c in range(16):

        for k in range(16):
            real_ac, imag_ad = 0, 0
            real_bd, imag_bc = 0, 0
            # n == 0
            real_ac += input[0, c] * param[0, k]
            real_bd += input[0, c + 16] * param[0, k + 16]

            imag_ad += input[0, c] * param[0, k + 16]
            imag_bc += input[0, c + 16] * param[0, k]
            # n ~ [1, 31]
            for n in range(1, 16):
                # 1 ~ 15
                real_ac += input[n, c] * param[n, k]
                real_bd += input[n, c + 16] * param[n, k + 16]
                imag_ad += input[n, c] * param[n, k + 16]
                imag_bc += input[n, c + 16] * param[n, k]
                # 16 ~ 31
                real_ac += input[31 - n, c] * param[n, k]
                real_bd += input[31 - n, c + 16] * -param[n, k + 16]
                imag_ad += input[31 - n, c] * -param[n, k + 16]
                imag_bc += input[31 - n, c + 16] * param[n, k]

            col_res[k, c] = real_ac - real_bd
            col_res[k, c + 31] = imag_ad + imag_bc

            if c > 0:
                col_res[k, 31 - c] = real_ac + real_bd
                col_res[k, 31 - c + 31] = imag_ad - imag_bc
    return col_res


def dft_31_31(input,param):
    row_dft_res = row_dft(input,param)
    return col_dft(row_dft_res,param)


def row_idft(input,param):
    row_res = np.zeros((16, 62))

    for r in range(16):
        for k in range(16):
            # n ~ [0, 31]
            real_ac, imag_ad = 0, 0
            real_bd, imag_bc = 0, 0
            # n == 0
            real_ac += input[r, 0] * param[0, k]
            real_bd += input[r, 31] * param[0, k + 16]
            imag_ad += input[r, 0] * param[0, k + 16]
            imag_bc += input[r, 31] * param[0, k]
            for n in range(1, 16):

                # n ~ [1, 15]
                real_ac += input[r, n] * param[n, k]
                real_bd += input[r, n + 31] * param[n, k + 16]
                imag_ad += input[r, n] * param[n, k + 16]
                imag_bc += input[r, n + 31] * param[n, k]

                # n ~ [16, 31]
                real_ac += input[r, 31 - n] * param[n, k]
                real_bd += input[r, 31 - n + 31] * -param[n, k + 16]
                imag_ad += input[r, 31 - n] * -param[n, k + 16]
                imag_bc += input[r, 31 - n + 31] * param[n, k]

            row_res[r, k] = real_ac + real_bd
            row_res[r, k + 31] = -imag_ad + imag_bc
            if k > 0:
                row_res[r, 31 - k] = real_ac - real_bd
                row_res[r, 31 - k + 31] = -(-imag_ad - imag_bc)
    return row_res

def col_idft(input,param):
    col_res = np.zeros((31, 31))
    for c in range(31):
        for k in range(16):
            real_ac, real_bd = 0, 0
            real_ac += input[0, c] * param[0, k]
            for n in range(1, 16):
                # n ~ [1, 15] n ~ [16 31]
                real_ac += 2 * input[n, c] * param[n, k]
                real_bd += 2 * input[n, c + 31] * param[n, k + 16]

                # # n ~ [16, 31]
                # real_ac += input[n, c] * param[n, k]
                # real_bd += -input[n, c + 31] * -param[n, k + 16]

            col_res[k][c] = real_ac + real_bd

            if(k > 0):
                col_res[31 - k][c] = real_ac - real_bd
    return col_res /961

def idft_31_31(input,param):
    row_idft_res = row_idft(input,param)
    return col_idft(row_idft_res,param)

def myrot(input):
    res = np.zeros(input.shape)
    w,h = input.shape
    for i in range(w):
        for j in range(h):
            res[i][j] = input[w - i -1][h - j -1]
    return res




def hadamard_product(dft_input_x, dft_kernel):

    prod_res = np.zeros((16, 62))
    for r in range(16):
        for c in range(31):
            real = dft_input_x[r][c] * dft_kernel[r][c] -dft_input_x[r][c + 31] * dft_kernel[r][c + 31]
            imag = dft_input_x[r][c] * dft_kernel[r][c + 31] + dft_input_x[r][c + 31] * dft_kernel[r][c]
            prod_res[r][c] = real
            prod_res[r][c + 31] = imag
    return prod_res


def DFT_conv(input,kernel,param):
    pad_x = np.zeros((31, 31))
    pad_x[0:29, 0:29] = input

    pad_kernel = np.zeros((31, 31))
    pad_kernel[:3, :3] = myrot(kernel)

    # dft
    input_x_dft = dft_31_31(pad_x, param)
    kernel_dft = dft_31_31(pad_kernel, param)

    # hadamard product
    res = hadamard_product(input_x_dft,kernel_dft)
    # idft
    idft_res = idft_31_31(res, param)

    return idft_res


def Conv(kernel,input):
    pad_x = np.pad(input, 2)
    conv_res = np.zeros((31, 31))
    for r in range(31):
        for c in range(31):
            tmp_value = 0
            for kr in range(3):
                for kc in range(3):
                    tmp_value += pad_x[r + kr, c + kc] * kernel[kr, kc]
            conv_res[r][c] = tmp_value
    return conv_res




# %%

kernel = np.random.randint(low=1,high = 3,size=(3,3))
param = param()
input = np.random.randn(29,29)

conv_res = Conv(kernel,input)
show("conv_res",conv_res)

dft_conv_res = DFT_conv(input,kernel,param)
show("dft_conv_res",dft_conv_res)
show("compare",dft_conv_res-conv_res<1e-5)

# %%
