import numpy as np

a = np.random.normal(-5, 5, size=(1, 3, 3, 16, 2))

np.set_printoptions(precision=6,threshold=np.inf)
print("random input: ", a)

print("slice input: ", a[:, : , :, :, 0])
print("slice input: ", a[:, : , :, :, 1])

a_t = np.transpose(a, [4, 0, 1, 2, 3])
print("tranpose input: ", a_t)
print("slice tranpose input: ", a_t[0, : , :, :, :])
print("slice tranpose input: ", a_t[1, : , :, :, :])
