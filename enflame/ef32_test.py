import numpy as np
import tensorflow as tf

# Python program to convert float
# decimal to binary number

import struct

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def test():
    a = np.random.uniform(low = -1.0,high = 1.0, size = (1)).astype('float32')
    # print(a)
    a = 0.09688106
    a_bin_str = float_to_bin(a)
    print('before', a_bin_str)
    a_bin_str = a_bin_str[:-12] + '0'*12
    print('after:', a_bin_str)
    fa = bin_to_float(a_bin_str)
    print(a)
    print(fa)
    print(a - fa)
 

if __name__ == "__main__":
    test()
