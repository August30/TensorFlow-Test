#%%

import numpy as np

#%%

input_array =np.array([[
                [0,1,1,2,2],
                [0,1,1,0,0],
                [1,1,0,1,0],
                [1,0,1,1,1],
                [0,2,0,1,0]
            ],
            [
                [1,1,1,2,0],
                [0,2,1,1,2],
                [1,2,0,0,2],
                [0,2,1,2,1],
                [2,0,1,2,0],
            ],
            [
                [2,0,2,0,2],
                [0,0,1,2,1],
                [1,0,2,2,1],
                [2,0,2,0,0],
                [0,0,1,1,2],
            ]])

#%%

input_array.shape

#%%

img_array = []
for i in range(5):
    for j in range(5):
        img_array.append([input_array[0,i,j],input_array[1,i,j],input_array[2,i,j]])
img_array = np.reshape(img_array,(5,5,3))                    

#%%

img_array[0],img_array.shape

#%%

# 确定padding的范围。
padding_array = np.zeros((7,7,3))
for i in range(3):
    for j in range(5):
        for k in range(5):
            padding_array[j+1,k+1,i] = img_array[j,k,i]
print(padding_array[:,:,0])

#%%

w_0=np.array([
    [[1,-1,0],[1,0,1],[-1,-1,0]],
    [[-1,0,1],[0,0,0],[1,-1,1]],
    [[-1,1,0],[-1,-1,-1],[0,0,1]]
    ])
w_1 = np.array([
    [[-1,1,-1],[-1,-1,0],[0,0,1]],
    [[-1,-1,1],[1,0,0],[0,-1,1]],
    [[-1,-1,0],[1,0,-1],[0,0,0]]
])

#%%

b_0 = 1
pic = []
## 确定 i，j, channel的范围
for i in range(3):
    for j in range(3):
        temp = 0
        for channel in range(3):
            temp = temp+np.sum(padding_array[i*2:i*2+3,j*2:j*2+3,channel]*w_0[:,:,channel])
        pic.append(temp)

#%%

pic_array = np.array(pic)
pic_reshape = np.reshape(pic_array,(3,3))

#%%

pic_reshape = pic_reshape+b_0

#%%

pic_reshape
