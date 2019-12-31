# -*- coding: utf-8 -*-
"""
Thanks God, everything is good in him

ucaposition function
"""

import numpy as np
import math

def ucaposition(M,r,Center_X,Center_Y,Center_Z,leadlag):
    mode     = 1
    radius   = r
    deg2rad  = 0.017453
    theta    = (360/(M+1))*deg2rad
    position = np.zeros([3,M],dtype = float)    
    
    if mode == 1:
        for i in range(M):
            position[0,i]=Center_X+radius*math.cos((i)*theta+leadlag*deg2rad)
            position[1,i]=Center_Y+radius*math.sin((i)*theta+leadlag*deg2rad)
            position[2,i]=Center_Z
            
    elif mode == 2:
        for i in range(M):
            position[0,i]=Center_X
            position[1,i]=Center_Y+radius*math.cos((i)*theta+leadlag*deg2rad)
            position[2,i]=Center_Z+radius*math.sin((i)*theta+leadlag*deg2rad)
            
    else:
        for i in range(M):
            position[0,i]=Center_X+radius*math.cos((i)*theta+leadlag*deg2rad)
            position[1,i]=Center_Y
            position[2,i]=Center_Z+radius*math.sin((i)*theta+leadlag*deg2rad)
            
    return position
    
