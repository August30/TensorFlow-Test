import numpy as np

def conv2d():
    input_array = np.array([[
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

    input_image = []
    for i in range(5):
        for j in range(5):
            input_image.append([input_array[0, i, j], input_array[1, i, j], input_array[2, i, j]])
    input_image = np.reshape(input_image, (5, 5, 3))
    print('input_image:', input_image)

    padding_array = np.zeros((7, 7, 3))
    for i in range(3):
        for j in range(5):
            for k in range(5):
                padding_array[j+1, k+1, i] = input_image[j, k, i]
    print('padding_array[:, :, 0]=', padding_array[:, :, 0])

    w_0 = np.array([[
                [1,-1,0],
                [1,0,1],
                [-1,-1,0]
            ],
            [
                [-1,0,1],
                [0,0,0],
                [1,-1,1]
            ],
            [
                [-1,1,0],
                [-1,-1,-1],
                [0,0,1]
            ]])
    w_1 = np.array([
        [[-1,1,-1],[-1,-1,0],[0,0,1]],
        [[-1,-1,1],[1,0,0],[0,-1,1]],
        [[-1,-1,0],[1,0,-1],[0,0,0]]
    ])

    bias_0 = 1
    bias_1 = 1

    w_array = [w_0,w_1]
    b_array = [bias_0, bias_1]

    stride = 2 
     
    res = []
    for k in range(2): 
        out = [] # (5+2-3)/2 + 1 = 3,  so w=3,h=3
        for w in range(3):
            for h in range(3):
                temp = 0
                for chanel in range(3):
                    temp += np.sum(padding_array[w*stride:w*stride+3, h*stride:h*stride+3, chanel] * w_array[k][:, :, chanel])
                # out.append(temp)
                out = np.append(out, temp)
        
        out = np.reshape(out, (3,3))
        out += b_array[k]
        print("out:", out)
        res.append(out)
    res = np.reshape(res, (2,3,3))
    print("res:", res)
    res = np.reshape(res, (3,3,2))
    print("res:", res)



if __name__ == "__main__":
    conv2d()






