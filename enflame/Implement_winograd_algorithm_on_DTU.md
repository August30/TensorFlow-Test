# 在DTU上实现winograd 卷积加速算法
## 前言
在卷积网络中，大部分的时间耗费在计算卷积的过程中，卷积算子性能的快慢在很大程度上决定了卷积网络的性能。在很多场景下，对网络的性能(throughput和latency)要求是非常严格的。就卷积而言，除了提高DMA效率、CDMA和SIP之间的pipeline、优化指令流水、提高数据复用率等常规手段外，还有没有其他优化方法？[Winograd](https://arxiv.org/abs/1509.09308)是一种快速卷积算法，使得卷积算子性能能够一定程度上超过加速卡的算力瓶颈。本文讲述的就是如何在[DTU(LEO T10)](https://www.enflame-tech.com/product-technologies/T10)上实现winograd卷积加速算法。
## 1D Convolution
输入信号为$d = [d_0, d_1, d_2, d_3]^T$，卷积核为$g = [g_0, g_1, g_2]^T$，则该卷积可写成如下矩阵乘法形式：
$$
F(2,3)=\begin{bmatrix} 
		d_0 & d_1 & d_2 \\ 
		d_1 & d_2 & d_3 \\
\end{bmatrix}
\begin{bmatrix} 
		g_0 \\
		g_1 \\
		g_2 \\
\end{bmatrix}
=
\begin{bmatrix} 
		r_0 \\
		r_1 \\
\end{bmatrix}
$$
如果是一般的矩阵乘法，则需要6次乘法和4次加法，如下：
$$
\begin{matrix}
r_0 = (d_0 \cdot g_0) + (d_1 \cdot g_1) + (d_2 \cdot g_2) \\
r_1 = (d_1 \cdot g_0) + (d_2 \cdot g_1) + (d_3 \cdot g_2) \\
\end{matrix}
$$
但是，卷积运算中输入信号转换成的矩阵不是任意矩阵，其中有规律地分布着大量的重复元素，比如第1行和第2行的$d_1$和$d_2$，卷积转换成的矩阵乘法比一般矩阵乘法的问题域更小，这就让优化存在了可能：
$$
F(2,3)=\begin{bmatrix} 
		d_0 & d_1 & d_2 \\ 
		d_1 & d_2 & d_3 \\
\end{bmatrix}
\begin{bmatrix} 
		g_0 \\
		g_1 \\
		g_2 \\
\end{bmatrix}
=
\begin{bmatrix} 
		m_1 + m_2 + m_3 \\
		m_2 - m_3 - m_4 \\
\end{bmatrix}
$$
其中，
$$
\begin{matrix}
m_1 = (d_0 - d_2)g_0 &  m_2 = (d_1 + d_2) \frac {g_0+g_1+g_2}{2} \\
m_4 = (d_1 - d_3)g_2 &  m_3 = (d_2 - d_1) \frac {g_0-g_1+g_2}{2} \\
\end{matrix}
$$
需要的运算次数分别为：
> + 输入信号$d$上：4次加法
> + 卷积核$g$上：3次加法，2次乘法（3次是因为g1+g2已经算过一次）
> + 输出$m$上：4次乘法，4次加法

在神经网络的推理阶段，卷积核上的元素是固定的，因此$g$上的运算可以提前算好，预测阶段只需计算一次，可以忽略，所以一共所需的运算次数为$d$与$m$上的运算次数之和，即4次乘法和8次加法。
将上面的计算过程写成矩阵形式如下：
$$
Y = A^T[(Gg) \circ (B^Td)]
$$
其中，$\circ$ 为element-wise multiplication（Hadamard product）对应位置相乘，
$$
B^T = \begin{bmatrix} 
		1 & 0 & -1 & 0 \\ 
		0 & 1 & 1 & 0 \\
		0 & -1 & 1 & 0 \\
		0 & 1 & 0 & -1 \\
\end{bmatrix}
$$
$$
G = \begin{bmatrix} 
		1 & 0 & 0 \\
		\frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\
		\frac{1}{2} & -\frac{1}{2} & \frac{1}{2} \\
		0 & 0 & 1 \\
\end{bmatrix}
$$
$$
A^T = \begin{bmatrix} 
		1 & 1 & 1 & 0 \\
		0 & 1 & -1 & -1 \\
\end{bmatrix}
$$
$$
g = \begin{bmatrix} 
		g_0 & g_1 & g_2 \\
\end{bmatrix}^T
$$
$$
d = \begin{bmatrix} 
		d_0 & d_1 & d_2 & d_3 \\
\end{bmatrix}^T
$$
> + $g$：卷积核
> + $d$：输入信号
> + $G$：Filter transform矩阵，尺寸$(m + r - 1) \times r$
> + $B^T$：Input transform矩阵，尺寸$(m + r - 1) \times (m + r - 1)$
> + $A^T$：Output transform矩阵，尺寸$m \times (m + r - 1)$

整个计算过程在逻辑上可以分为4步：
> + Input transform
> + Filter transform
> + Hadamard product
> + Output transform

## 2D Convolution
将1D convolution F(2,3)扩展到2D F(2x2,3x3)运算，公式如下所示：
$$
Y = A^T[[GgG^T] \circ [B^TdB]]A
$$
其中，$g$为$r \times r$ Filter, $d$为$(m + r - 1) \times (m + r - 1)$的image tile。
## Winograd加速比分析
> 以F(2x2, 3x3)为例，进行运算量评估
> 1. Input transform
> > + Input feature map: [N, 4, 4, Ci]
> > + 加法：32 * N * Ci
> 2. Filter transform
> > + Filter: [3, 3, Ci, Co]
> > + 加法：24 * Ci * Co
> > + 乘法: 12 * Ci * Co
> > + 如果是推理，可以在编译期间进行预处理，这部分计算量可以忽略掉
> 3. Hadamard product
> > + 加法：4 * 4 * N * Co * (Ci - 1)
> > + 乘法：4 * 4 * N * Co * Ci
> 4. Output transform
> > + 加法：24 * N * Co
> 5. 在LEO(1 sip 256 macs with F32)上用tensor去计算此卷积所需要的cycle数是：
> > + cycle: (N * 2 * 2 * Co * 3 * 3 * Ci) / 256
> 6. 采用winograd算法计算此卷积所需要的cycle数是：
> > + cycle: (32 * N * Ci) / 32 + (4 * 4 * N * Co * Ci )/ 256 + (24 * N * Co) / 32

Winograd在DTU(LEO F32)上的理论加速比为： 

|（Ci,Co)|(64,64)|(64,128)|(64,256)|(128,64)|(128,128)|(128,256)|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| - | 1.57 | 1.71 | 1.8 | 1.67 | 1.85 | 1.95 |

如果1D和2D指令能够同时发射的话，那么理论加速比就可以达到2.25。
## 在DTU上实现Winograd F(2x2,3x3)
$$
Y = A^T[[GgG^T] \circ [B^TdB]]
$$
### 计算过程
```
Input transform：
    input format in sip dram format is [4, 4, Ci, N]
    用1D指令去做预处理，4x4 总共16个点一起去做
Filter transform：
    在编译期间做预处理
Hadamard product：
    4x4 总共16个点对应元素相乘，看起来像是elementwise计算，但是要用2D指令去加速这个计算过程
    在LEO上可以采用如下方法去做：
    input format is [4, 4, Ci, N]
    kernel format is [4, 4, Ci, Co]
    for i in range(4):
        for j in range(4):
            output[i, j, :, :] = 0
            for ci in range(Ci):
                output[i, j, :, :] += dot(input[i, j, ci, :], kernel[i, j, ci, :])
    LEO需要依赖 N >=16，如果N比较小的话，需要拿更多的input H和W，这里不展开了

    在sip 2.0上可以用vmm指令去做
    input format is [4, 4, Ci]
    kernel format is [4, 4, Ci, Co]
    for i in range(4):
        for j in range(4):
            output[i, j, :] = dot(input[i, j, :], kernel[i, j, :, :])
    在sip 2.0上更灵活，不需要依赖大batch
Output transform：
    output format is [4, 4, N, Co]
    用1D指令去做后处理
    后处理后的format is [2, 2, N, Co]
```
### Convolution format in HBM
```
Input: [N, Hi, Wi, C]
Kernel: [Hk, Wk, Ci, Co]
Output: [N, Ho, Wo, Co]
```
### Tiling size
```
// CDMA
N_CDMA = 32
Hi_CDMA = 4
Wi_CDMA = 4
Ci_CDMA = 64
Hk_CDMA = 4
Wk_CDMA = 4
Ho_CDMA = 2
Wo_CDMA = 2
Co_CDMA = 32
// SDMA
N_SDMA = 32
Hi_SDMA = 4
Wi_SDMA = 4
Ci_SDMA = 64
Hk_SDMA = 1
Wk_SDMA = 2
Co_SDMA = 32
```
### 在编译期间对权重进行预处理
```
1. Reshape
   SRC shape: [Hk, Wk, Ci, Co]
   DST shape: [Ci, Co, Hk, Wk]
   Ctrl: [2, 3, 0, 1]
2. Loop MatrixMulti
   for ci in range(Ci):
       for co in range(Co):
           weight_preprocess[ci,co,:,:] = Dot(Dot(G, weight[ci,co,:,:]), G.transpose())
3. Reshape
   SRC shape: [Ci, Co/Co_CDMA, Co_CDMA, 4, 4]
   DST shape: [Co/Co_CDMA, 4, 4, Ci, Co_CDMA]
   Ctrl: [1, 3, 4, 0, 2]
```
### CSB layout
```
┌────────────────┬────────────────┬────────────────┬────────────────┐
│Kernel buf      │Kernel buf      │Kernel buf      │Kernel buf      │
│In/Out buf ping │In/Out buf ping │In/Out buf ping │In/Out buf ping │
│In/Out buf pong │In/Out buf pong │In/Out buf pong │In/Out buf pong │
│TMP buf         │TMP buf         │TMP buf         │TMP buf         │
│Param buf ping  │                │                │                │
│Param buf pong  │                │                │                │
├────────────────┼────────────────┼────────────────┼────────────────┤
│Kernel buf      │Kernel buf      │Kernel buf      │Kernel buf      │
│In/Out buf ping │In/Out buf ping │In/Out buf ping │In/Out buf ping │
│In/Out buf pong │In/Out buf pong │In/Out buf pong │In/Out buf pong │
│TMP buf         │TMP buf         │TMP buf         │TMP buf         │
│                │                │                │                │
│                │                │                │                │
└────────────────┴────────────────┴────────────────┴────────────────┘
```
### Host Code
```
// Leading: cache weight in L2
for i in range(8):
    CDMA_D2C_COPY [4, 4, Ci_CDMA, Co_CDMA]

// Sync
Sync

// Loop ping/pong patrial run
for pingpong in range(2):
    // CDMA input x 8
    for i in range(8):
        CDMA_D2C_SLICE [N_CDMA, Hi_CDMA, Wi_CDMA, Ci_CDMA]
    // Launch 
    Launch 8 SIPs
    // CDMA output x 8
    for i in range(8):
        CDMA_C2D_DESLICE [N_CDMA, Ho_CDMA, Wo_CDMA, Co_CDMA * 8]
```
### SIP dram layout
```
┌───────────────┐
│Input buf      │
│Kernel buf ping│
│Kernel buf pong│
│Output buf     │
├───────────────┤
│Stack          │
└───────────────┘
```
### Kernel code
```
1. Load input L2->L1
   SDMA_C2S_RESHAPE
   SRC: [N_SDMA, Hi_SDMA, Wi_SDMA, Ci_SDMA]
   DST: [Hi_SDMA, Wi_SDMA, Ci_SDMA, N_SDMA]
   Ctrl: [1, 2, 3, 0]
2. Preload weight L2->L1
   SDMA_C2S_COPY [1(Hk), 2(Wk), Ci_SDMA, Co_SDMA]
3. Preprocess input
   Input format is [4, 4, Ci, Co]
4. for co in range(0, Co_SDMA * 8, Co_SDMA):
       for ho in range(4):
          for wo in range(0, 4, 2):
              // preload next weight L2->L1
              SDMA_C2S_COPY [1(Hk), 2(Wk), Ci_CDMA, Co_CDMA]
              for ci in range(Ci):
                  Output[ho, wo, :, :] = Dot(N, Co)
                  Output[ho, wo + 1, :, :] = Dot(N, Co)
       // postprocess output
       //

       SDMA_S2C_RESHAPE
       SRC: [2, 2, N_SDMA, Co_SDMA]
       DST: [N_SDMA, 2, 2, Co_SDMA]

       DESLICE_C2C_DESLICE
       SRC: [N_SDMA, 2, 2, Co_SDMA]
       DST: [N_SDMA, 2, 2, Co_SDMA * 8]
```
### Zebu仿真数据
测试平台：ZebuA 1C T10 config(DTU 1.15G, SOC 1.1G, MC 1G)  
Test case:
f32[64,28,28,256] convolution(f32[64,30,30,64], f32[3,3,64,256]), window={size=3x3}, dim_labels=b01f_01io->b01f  
ZebuA波形文件:/data/emu_user/yipinsun/leo/pltd.3.2sip/1c8s/2021_04_29_15_28.ztdb20210429153237/  
1C 算力：5.38T(1C理论算力为4.6T)  

![image](./_static/convolution_with_winograd_total_cycles.png)  
![image](./_static/convolution_winograd_sip_launch_cycles.png)  
![image](./_static/convolution_without_winograd_total_cycles.png)  

|加速比|2D inst|2D + 1D|一次sip launch|整个算子时间|DTU with/without winograd|
| :--: | :--: | :--: | :--: | :--: | :--: |
| - | 2.25 | 1.8 | 1.3 | 1.17| 2.39 |
## 总结
> 1. Winograd算法通过减少乘法次数来实现提速。F(2x2,3x3) 2D指令加速比可以达到2.25，F(4x4,3x3) 2D指令加速比可以达到4。将tile的尺寸增大可以获得更大的加速比。
> 2. 虽然将tile的尺寸增大可以获得更大的加速比，但是预处理后的weight也会增大，可能会导致L2/L1空间不够，需要在tile尺寸和加速比之间寻找一个最佳平衡点。
> 3. LEO架构上1D和2D指令不能同时发射，实际加速比会低于2D指令加速比。如果是推理场景，增大Ci、Co，可以逼近2D指令加速比。
> 4. Winograd算法需要通过1D指令去做input的预处理和output的后处理，如果1D和2D指令不能同时发射的话，会降低该算法加速比，比如LEO架构。
> 5. Winograd算法需要通过1D指令去做input的预处理和output的后处理，前处理和后处理公式相对简单，能不能做成一个硬件单元，在L2->L1/L1->L2的时候自动的去做预处理和后处理。比如huawei的davinci架构就有img2col单元。
> 6. Winograd算法需要通过1D指令去做input的预处理和output的后处理，在用1D指令的时候可能会导致精度下降。比如数据类型是FP16的话，用1D指令去做预处理就有可能会导致精度下降。
> 1. Winograd算法可以减少2D指令的数量，在一定程度上降低芯片的功耗，对功耗比较友好。
