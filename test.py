import random
import math
import numpy as np
import time
from typing import Union

from cergen import gergen, rastgele_dogal, rastgele_gercek, cekirdek

def main():
    # a main function to test the functions

    cekirdek(2)

    # g = rastgele_dogal((3,1,2,3))
    # g2 = rastgele_gercek((3, 3, 2, 4))
    # print(g)

    # g = gergen([1, 2, 3])
    # print(g.duzlestir().boyut())
    # print()
    # print(g.listeye())
    # print()
    # print(g)

    ### test the random number generators

    # g1 = rastgele_dogal((3, 3))
    # g1 = rastgele_gercek((3, 3, 3))
    # print(g1)
    # print(gergen())
    # print(g1)
    # print(g1[0][-1])

    ### scalar gergens and init

    # g3 = gergen(2)
    # print(g3)
    # print(g3.D)

    # gi = gergen(4)
    # print(gi)
    # print()
    # gii = gergen([4])
    # print(gii)

    # gv = gergen([1, 2, 3])
    # print(gv)
    # print()
    # gh = gergen([[1], [2], [3]])
    # print(gh)

    ### test the __getitem__ method
    
    # g = gergen([[1, 2, 3], [4, 5, 6]])
    # print(g)
    # print()
    # print(g[0])
    # print()
    # print(g[0, 1]) # = print(g[0][1])
    # print()
    # print(g[-1][-1])
    # print()
    
    # print(g[4, 1]) # IndexError: Index out of range.
    # print(g[0, 2, 1]) 

    # g = gergen([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(g)
    # print()
    # print(g[0])
    # print()
    # print(g[-1])
    # print()
    
    # print(g[10]) # IndexError: Index out of range.
    # print(g[0, 1]) # err

    ### test the __str__ method

    # print(gergen())
    # print(gergen(2))
    # print(gergen([[1, 2, 3]]))
    # print(gergen([[1, 2, 3], [4, 5, 6]]))

    # print(g)
    # print()
    # print(g[0])
    # print()
    # print(g[0, 1])
    # print()
    # print(g[0, 2, 1])
    # print()
    # print(g[0, 2, -1, 3])

    ### test the __mul__ method

    # g1 = gergen([[1, 2, 3], [4, 5, 6]])
    # gs1 = g1 * 2
    # # gss1 = g1 * gergen(2) # is not working
    # g2 = gergen([[7, 8, 9], [10, 11, 12]])
    # gs2 = g2 * -3 # (-3 * g2) is not working
    # # gss2 = gergen(-3) * g2 # is not working
    # g3 = g1 * g2
    # g4 = g1 * g2 * 2 # 
    # g5 = g1 * g2 * g3 * 2 # 
    # g1d1 = gergen(4)
    # g1d2 = gergen(3)
    # g1dm = g1d1 * g1d2

    # print(g3) # [[7, 16, 27], [40, 55, 72]]
    # print()
    # print(gs1) 
    # print()
    # print(gs2)
    # print()
    # print(g4) # [[14, 32, 54], [80, 110, 144]]
    # print()
    # print(g5) 
    # print()
    # print(g1dm)
    # print()
    # print(g1dm * g3)

    # g = gergen([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(g)
    # print()
    # print(g * 2)
    # print()
    # print(g * g)
    # print()
    # print(g * g * 2)
    # print()
    # g2 = gergen([[1, 2, 3], [4, 5, 6]])
    # print(g * g2)
    # g3 = gergen([2])
    # print(g * g3)

    # g = gergen([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(g)
    # print()
    # print(g * 2)
    # print()
    # print(2 * g)
    

    ### test the __truediv__ method

    # g1 = gergen([[1, 2, 3], [4, 5, 6]])
    # z1 = 0
    # gz1 = g1 / z1
    # print(gz1)

    # g2 = g1 * 2
    # z2 = 2
    # gz2 = g2 / z2
    # print(g2)
    # print()
    # print(gz2)

    # g3 = g1 / g1
    # print(g3)

    # g4 = gergen([[7, 8, 9], [10, 11, 12]])
    # print(g4 * 4 / g4)

    # g5 = gergen([6])
    # print(g5 / 3)
    # g6 = gergen([6, 7])
    # # print(g5 / g6) # ValueError: Cannot divide gergens with different dimensions.
    # g7 = gergen([3])
    # print(g5 / g7)

    # g = gergen([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(g)
    # print()
    # print(g / 2)
    # print()
    # print(g / g)
    # print()
    # print(g / g / 2)
    # print()
    # g3 = gergen([2])
    # print(g / g3)
    # g2 = gergen([[1, 2, 3], [4, 5, 6]])
    # print(g / g2)

    # g = gergen([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(g)
    # print()
    # print(g / 2)
    # print()
    # print(2 / g)
    # print()
    
    # g = rastgele_dogal((5, 3, 1, 2))
    # print(g)
    # print()
    # print(g / 2)
    # print()
    # print(4 * g)
    # print()
    # print(g / 4)
    # print()
    # print(1 / g)

    ### test the __add__ method

    ## add two scalars
    # gs1 = gergen(2)
    # gs2 = gs1 + 3
    # print(gs2)
    # gs3 = gs1 + gs2
    # print(gs3)

    ## add a scalar to a gergen
    # g = gergen([[1, 2, 3], [4, 5, 6]])
    # gs = g + 2
    # print(gs)

    ## add two gergens
    # ga = gergen([[1, 2, 3], [4, 5, 6]])
    # gb = gergen([[7, 8, 9], [10, 11, 12]])
    # gc = ga + gb
    # print(gc) # [[8, 10, 12], [14, 16, 18]]

    ## add 2 1x1 gergens
    # g11 = gergen([1])
    # g12 = gergen([4])
    # g13 = g11 + g12
    # print(g13)

    ## add 2 1x3 gergens
    # g21 = gergen([1, 2, 3])
    # g22 = gergen([4, 5, 6])
    # g23 = g21 + g22
    # print(g23) # [[5, 7, 9]]

    ## add 2 2x1 gergens
    # g211 = gergen([[1], [2]])
    # g212 = gergen([[3], [4]])
    # g213 = g211 + g212
    # print(g213) # [[4], [6]]

    ## add 1x1 and 1x3 gergens - ValueError: Cannot add gergens with different dimensions.
    # g11 = gergen([1])
    # g13 = gergen([4, 5, 6])
    # g113 = g11 + g13
    # print(g113)

    ## add 1x2 and 2x1 gergens - ValueError: Cannot add gergens with different dimensions.
    # g21 = gergen([1, 2])
    # g212 = gergen([[3], [4]])
    # g213 = g21 + g212
    # print(g213)

    ## add scalar to 5x3x1x2 gergen
    # g53 = rastgele_dogal((5, 3, 1, 2))
    # print(g53)
    # print()
    # gs53 = g53 + 2
    # print(gs53)

    ## test commutativity with the prev one
    # print(g53)
    # print()
    # print(g53 + g53)
    # print()
    # print(g53 + 5)
    # print()
    # print(5 + g53)
    # print()
    # print(-5 + g53)

    ## add 3x3x2x1 and 3x3x2x1 gergens
    # gg1 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg2 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg3 = gg1 + gg2
    # print(gg3)      

    ### test the __sub__ method

    ## subtract two scalars
    # gs1 = gergen(2)
    # gs2 = gs1 - 3
    # print(gs2)
    # gs3 = gs1 - gs2
    # print(gs3)

    ## subtract a scalar from a gergen
    # g = gergen([[1, 2, 3], [4, 5, 6]])
    # gs = g - 2
    # print(gs)

    ## subtract two gergens
    # ga = gergen([[1, 2, 3], [4, 5, 6]])
    # gb = gergen([[7, 8, 9], [10, 11, 12]])
    # gc = ga - gb
    # print(gc) # [[-6, -6, -6], [-6, -6, -6]]

    ## subtract 2 1x1 gergens
    # g11 = gergen([1])
    # g12 = gergen([4])
    # g13 = g11 - g12
    # print(g13)

    ## subtract 2 1x3 gergens
    # g21 = gergen([1, 2, 3])
    # g22 = gergen([4, 5, 6])
    # g23 = g21 - g22
    # print(g23) # [[-3, -3, -3]]

    ## subtract 2 2x1 gergens
    # g211 = gergen([[1], [2]])
    # g212 = gergen([[3], [4]])
    # g213 = g211 - g212
    # print(g213) # [[-2] \n [-2]]

    ## subtract 1x1 and 1x3 gergens - ValueError: Cannot subtract gergens with different dimensions.
    # g11 = gergen([1])
    # g13 = gergen([4, 5, 6])
    # g113 = g11 - g13
    # print(g113)

    ## subtract 1x2 and 2x1 gergens - ValueError: Cannot subtract gergens with different dimensions.
    # g21 = gergen([1, 2])
    # g212 = gergen([[3], [4]])
    # g213 = g21 - g212
    # print(g213)

    ## subtract scalar from 5x3x1x2 gergen
    # g53 = rastgele_dogal((5, 3, 1, 2))
    # print(g53)
    # print()
    # gs53 = g53 - 2
    # print(gs53)

    ## test commutativity with the prev one
    # print(g53)
    # print()
    # print(g53 - g53)
    # print()
    # print(g53 - 5)
    # print()
    # print(5 - g53)
    # print()
    # print(-5 - g53)

    ## subtract 3x3x2x1 and 3x3x2x1 gergens
    # gg1 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg2 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg3 = gg1 - gg2
    # print(gg3)

    ### test the uzunluk method

    ## empty gergen
    # g = gergen()
    # print(g.uzunluk()) # ValueError: Cannot get the length of an empty gergen.

    ## 1x1 gergen
    # g = gergen([1])
    # print(g.uzunluk()) # 1

    # gl = gergen([[1, 2, 3], [4, 5, 6]])
    # print(gl.uzunluk()) # 6

    # gll = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # print(gll.uzunluk()) # 12

    # glong = rastgele_dogal((5, 3, 1, 2, 4, 9))
    # print(glong.uzunluk()) # 1080

    # garr = gergen([1,2,3,4,5])
    # print(garr.uzunluk()) # 5

    # g = gergen([[1], [2], [3]])
    # print(g.uzunluk()) # 3

    ### test the boyut method

    ## empty gergen
    # g = gergen()
    # print(g.boyut()) # ValueError: Cannot get the shape of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.boyut()) # (1,)

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.boyut()) # (3,)
    # print(g.D)

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.boyut()) # (3, 1)

    ### test the devrik method

    ## empty gergen
    # g = gergen()
    # print(g.devrik()) # ValueError: Cannot get the transpose of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.devrik()) # 10

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g)
    # print()
    # print(g.devrik()) # [[1] \n [2] \n [3]]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g)
    # print()
    # print(g.devrik()) # [[1, 2, 3]]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.devrik()) # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g)
    # print()
    # print(g.boyutlandir((2,3,3)))
    # print()
    # print(g.devrik()) 
    # print("--------------------")
    # arr = np.array([[[1, 2], [3, 4], [5, 6]],
    #                 [[7, 8], [9, 10], [11, 12]],
    #                 [[13, 14], [15, 16], [17, 18]]])
    # print(arr.transpose().shape)
    # print(arr.transpose())

    ## 3x4x2x5 gergen
    # g = rastgele_dogal((3, 4, 2, 5))
    # print(g)
    # print()
    # print(g.devrik())
    # print("--------------------")
    # arr = np.array(g.listeye())
    # print(arr.transpose().shape)
    # print(arr.transpose())

    ### test the sin, cos, tan methods

    ## empty gergen
    # g = gergen()
    # print(g.sin()) # ValueError: Cannot apply the function to an empty gergen.
    # print(g.cos()) # ValueError: Cannot apply the function to an empty gergen.
    # print(g.tan()) # ValueError: Cannot apply the function to an empty gergen.

    ## 1x1 gergen
    # g = gergen([math.pi / 2])
    # print(g.sin()) # 1.0
    # print(g.cos()) # 0.0
    # print(g.tan()) # err

    ## 1x3 gergen
    # g = gergen([math.pi / 6, math.pi / 3, math.pi / 4])
    # print(g.sin()) 
    # print(g.cos()) 
    # print(g.tan()) 

    # 3x1 gergen
    # g = gergen([[math.pi / 6], [math.pi / 3], [math.pi / 4]])
    # print(g.sin()) # [[0.5], [0.8660254037844386], [0.7071067811865475]]
    # print(g.cos()) # [[0.8660254037844386], [0.5], [0.7071067811865475]]
    # print(g.tan()) # [[0.5773502691896257], [1.7320508075688774], [0.9999999999999999]]

    ## 3x3 gergen
    # g = gergen([[math.pi / 6, math.pi / 3, math.pi / 4], [math.pi / 6, math.pi / 3, math.pi / 4], [math.pi / 6, math.pi / 3, math.pi / 4]])
    # print(g.sin()) 
    # print(g.cos())
    # print(g.tan())

    ## 3x3x2 gergen
    # g = gergen([[[math.pi / 6, math.pi / 3], [math.pi / 4, math.pi / 6], [math.pi / 3, math.pi / 4]], [[math.pi / 6, math.pi / 3], [math.pi / 4, math.pi / 6], [math.pi / 3, math.pi / 4]], [[math.pi / 6, math.pi / 3], [math.pi / 4, math.pi / 6], [math.pi / 3, math.pi / 4]]])
    # print(g.sin())
    # print(g.cos())
    # print(g.tan())

    ### test the us method

    ## empty gergen
    # g = gergen()
    # print(g.us(2)) # ValueError: Cannot raise an empty gergen to a power.

    ## scalar gergen
    # g = gergen(3)
    # print(g.us(2)) # 9

    # g = gergen(0)
    # print(g.us(0)) # err

    ## 1x1 gergen
    # g = gergen([3])
    # print(g.us(2)) # 9

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.us(2)) # [1, 4, 9]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.us(2)) # [[1], [4], [9]]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.us(2)) # [[1, 4, 9], [16, 25, 36], [49, 64, 81]]

    ## 3x3 gergen with -
    # g = gergen([[4, 4, 4], [4, 4, 4], [4, 4, 4]])
    # print(g.us(-1)) # err

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.us(2))

    ## 3x5x3x2 gergen with 0
    # g = gergen([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]], [[25, 26], [27, 28], [29, 30]]], [[[31, 32], [33, 34], [35, 36]], [[37, 38], [39, 40], [41, 42]], [[43, 44], [45, 46], [47, 48]], [[49, 50], [51, 52], [53, 54]], [[55, 56], [57, 58], [59, 60]]], [[[61, 62], [63, 64], [65, 66]], [[67, 68], [69, 70], [71, 72]], [[73, 74], [75, 76], [77, 78]], [[79, 80], [81, 82], [83, 84]], [[85, 86], [87, 88], [89, 90]]]])
    # print(g.us(0))

    ### test the log methods

    ## empty gergen
    # g = gergen()
    # print(g.log()) # ValueError: Cannot calculate the logarithm of an empty gergen.
    # print(g.ln()) # ValueError: Cannot calculate the natural logarithm of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.log()) # 1.0
    # print(g.ln()) # 2.302585092994046

    ## scalar gergen
    # g = gergen(10)
    # print(g.log()) # 1.0
    # print(g.ln()) # 2.302585092994046

    # g = gergen(0)
    # print(g.log()) # err
    # print(g.ln()) # err

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.log()) # [0.0, 0.3010299956639812, 0.47712125471966244]
    # print(g.ln()) # [0.0, 0.6931471805599453, 1.0986122886681098]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.log()) # [[0.0], [0.3010299956639812], [0.47712125471966244]]
    # print(g.ln()) # [[0.0], [0.6931471805599453], [1.0986122886681098]]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.log()) # [[0.0, 0.3010299956639812, 0.47712125471966244], [0.6020599913279624, 0.6989700043360189, 0.7781512503836436], [0.8450980400142568, 0.9030899869919435, 0.9542425094393249]]
    # print(g.ln()) # [[0.0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055], [1.9459101490553132, 2.0794415416798357, 2.1972245773362196]]

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.log())
    # print(g.ln())

    ### test the L1, L2, Lp methods

    ## empty gergen
    # g = gergen()
    # print(g.L1()) # ValueError: Cannot calculate the L1 norm of an empty gergen.
    # print(g.L2()) # ValueError: Cannot calculate the L2 norm of an empty gergen.
    # print(g.Lp(3)) # ValueError: Cannot calculate the Lp norm of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.L1()) # 10
    # print(g.L2()) # 10.0
    # print()
    # print(g.Lp(1)) # 10
    # print(g.Lp(2)) # 10.0
    # print(g.Lp(3)) # 10.0

    ## scalar gergen
    # g = gergen(10)
    # print(g.L1()) # 10
    # print(g.L2()) # 10.0
    # print()
    # print(g.Lp(1)) # 10
    # print(g.Lp(2)) # 10.0
    # print(g.Lp(3)) # 10.0

    # print(g.Lp(0)) # ValueError: p must be a positive number.
    # print(g.Lp(-1)) # ValueError: p must be a positive number.

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.L1()) # 6
    # print(g.L2()) # 3.7416573867739413
    # print()
    # print(g.Lp(1)) # 6
    # print(g.Lp(2)) # 3.7416573867739413
    # print(g.Lp(3)) # 3.3019272488946263

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.L1()) # 6
    # print(g.L2()) # 3.7416573867739413
    # print()
    # print(g.Lp(1)) # 6
    # print(g.Lp(2)) # 3.7416573867739413
    # print(g.Lp(3)) # 3.3019272488946263

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.L1()) # 45
    # print(g.L2()) # 16.881943016134134
    # print()
    # print(g.Lp(1)) # 45
    # print(g.Lp(2)) # 16.881943016134134
    # print(g.Lp(3)) # 

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.L1())
    # print(g.L2())
    # print()
    # print(g.Lp(1))
    # print(g.Lp(2))
    # print(g.Lp(3))

    ### test the duzlestir method

    ## empty gergen
    # g = gergen()    
    # print(g.duzlestir()) # ValueError: Cannot flatten an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.duzlestir()) # [10]

    ## scalar gergen
    # g = gergen(10)
    # print(g.duzlestir()) # [10]

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.duzlestir()) # [1, 2, 3]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.duzlestir()) # [1, 2, 3]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.duzlestir()) # [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.duzlestir()) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    ## 3x5x3x2 gergen
    # g = rastgele_dogal((3, 5, 3, 2))
    # print(g.duzlestir())

    ## some 3d gergen
    # g = gergen([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    # print(g)
    # print()
    # print(g.duzlestir()) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    ### test the boyutlandir method

    ## empty gergen
    # g = gergen()
    # print(g.boyutlandir((1, 1))) # ValueError: Cannot reshape an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.boyutlandir((1,))) # [[10]]
    # print(g.boyutlandir((1, 2))) # ValueError: Cannot reshape a gergen to a different size.

    ## scalar gergen
    # g = gergen(10)
    # print(g.boyutlandir((1,))) # [[10]]
    # print(g.boyutlandir((1, 1, 5))) # ValueError: Cannot reshape a gergen to a different size.

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.boyutlandir((3, 1))) # [[1] \n [2] \n [3]]
    # print(g.boyutlandir((1, 3))) # [[1, 2, 3]]
    # print(g.boyutlandir((3, 1, 1, 1, 6))) # ValueError: Cannot reshape a gergen to a different size.

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.boyutlandir((1, 3))) # [[1, 2, 3]]
    # print(g.boyutlandir((3,))) # [1, 2, 3]
    # print(g.boyutlandir((3, 1))) # [[1] \n [2] \n [3]]
    # print(g.boyutlandir((1, 1, 3, 1, 1))) 

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.boyutlandir((1, 9))) # [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    # print(g.boyutlandir((9, 1))) # [[1] \n [2] \n [3] \n [4] \n [5] \n [6] \n [7] \n [8] \n [9]]
    # print(g.boyutlandir((3, 3))) # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # print(g.boyutlandir((3, 1, 1, 3, 1)))
    # print(g.boyutlandir((1, 3, 1, 1, 3, 1, 1, 2))) # ValueError: Cannot reshape a gergen to a different size.

    ## 2x3x3 gergen to 3x2x3
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((3, 2, 3)))

    ## 2x3x3 gergen to 9x2 gergen
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((9, 2)))

    ## 2x3x3 gergen to 1x18 gergen
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((1, 18)))

    ## 2x3x3 gergen to 18x1 gergen
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((18, 1)))

    ### test the ic_carpim method

    ## empty gergen
    # g = gergen()
    # print(g.ic_carpim(g)) # ValueError: Cannot calculate the inner product of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.ic_carpim(g)) # 100

    ## scalar gergen
    # g = gergen(10)
    # print(g.ic_carpim(g)) # 100

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.ic_carpim(g)) # 14
    # print()
    # arr = np.array(g.listeye())
    # print(np.dot(arr, np.array(g.listeye()))) # 14

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.ic_carpim(g.boyutlandir((1, 3)))) # 14
    # # print(g.ic_carpim(g)) # err
    # print()
    # arr = np.array(g.listeye())
    # print(np.dot(arr, np.array(g.boyutlandir((1,3)).listeye())))

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.ic_carpim(g)) 
    # print()
    # arr = np.array(g.listeye())
    # print(np.dot(arr, arr))

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.ic_carpim(g)) # err

    ### test the dis_carpim method

    ## empty gergen
    # g = gergen()
    # print(g.dis_carpim(g)) # ValueError: Cannot calculate the outer product of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.dis_carpim(g)) # [[100]]
    # print()
    # arr = np.array(g.listeye())
    # print(np.outer(arr, arr))

    ## scalar gergen
    # g = gergen(10)
    # print(g.dis_carpim(g)) # err

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.dis_carpim(g)) # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    # print()
    # arr = np.array(g.listeye())
    # print(np.outer(arr, arr))

    # g = gergen([1, 2, 3, 4, 5, 6])
    # print(g.dis_carpim(g)) # [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], [3, 6, 9, 12, 15, 18], [4, 8, 12, 16, 20, 24], [5, 10, 15, 20, 25, 30], [6, 12, 18, 24, 30, 36]]
    # print()
    # arr = np.array(g.listeye())
    # print(np.outer(arr, arr))

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.dis_carpim(g)) # ValueError: Cannot calculate if the gergen is not 1D.

    ## 3x3 gergen 
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.dis_carpim(g)) # ValueError: Cannot calculate if the gergen is not 1D.

    ### test the topla method

    ## empty gergen
    # g = gergen()
    # print(g.topla()) # ValueError: Cannot sum up elements of an empty gergen.
    # print(g.topla(0)) # ValueError: Cannot sum up elements of an empty gergen. 

    ## 1x1 gergen
    # g = gergen([3])
    # print(g.topla()) # 3
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))
    # print()
    # print(g.topla(0)) # 3
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=0))
    # print()
    # print(g.topla(1)) # err
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=1))

    ## scalar gergen
    # g = gergen(10)
    # print(g.topla()) # 10
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))

    # print(g.topla(0)) # err

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.topla()) # 6
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))
    # print()
    # print(g.topla(0)) # 6
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=0))

    # print(g.topla(1)) # err

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])

    # print(g.topla()) # 6
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))

    # print()
    # print(g.topla(0)) # [6]
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=0))

    # print(g.topla(1)) # [1, 2, 3]
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=1))

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # print(g.topla()) # 45
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))

    # print(g.topla(0)) # [12, 15, 18]
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=0))

    # print(g.topla(1)) # [6, 15, 24]
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=1))

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g)

    # print(g.topla()) # 171
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))

    # print(g.topla(0))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=0))

    # print(g.topla(1))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=1))

    # print(g.topla(2))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=2))

    ## 3x5x3x2 gergen
    # g = rastgele_dogal((3, 5, 3, 2))

    # print(g.topla())
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr))

    # print(g.topla(0))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=0))

    # print(g.topla(1))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=1))

    # print(g.topla(2))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=2))

    # print(g.topla(3))
    # print()
    # arr = np.array(g.listeye())
    # print(np.sum(arr, axis=3))

    ### test the ortalama method

    ## empty gergen
    # g = gergen()
    # print(g.ortalama()) # ValueError: Cannot calculate the average of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.ortalama()) # 10.0
    # print("----------")
    # print(g.ortalama(0)) # 10.0
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=0))

    # print(g.ortalama(1)) # err

    ## scalar gergen
    # g = gergen(10)
    # print(g.ortalama()) # 10.0
    # print("----------")
    # print(g.ortalama(0)) # err no axis in scalars
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=0))

    # print(g.ortalama(1)) # err

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.ortalama()) # 2.0
    # print("----------")
    # print(g.ortalama(0)) # 2.0
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=0))

    # print(g.ortalama(1)) # err

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.ortalama()) # 2.0
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr))
    # print("----------")
    # print(g.ortalama(0)) # 2.0
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=0))
    # print("----------")
    # print(g.ortalama(1)) # [2.0]
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=1))

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.ortalama()) # 5.0
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr))
    # print("----------")
    # print(g.ortalama(0)) # [4.0, 5.0, 6.0]
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=0))
    # print("----------")
    # print(g.ortalama(1)) # [2.0, 5.0, 8.0]
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=1))

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.ortalama()) # 9.5
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr))
    # print("----------")
    # print(g.ortalama(0))
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=0))
    # print("----------")
    # print(g.ortalama(1))
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=1))
    # print("----------")
    # print(g.ortalama(2))
    # print("----------")
    # arr = np.array(g.listeye())
    # print(np.mean(arr, axis=2))
    
if __name__ == "__main__":
    main()
