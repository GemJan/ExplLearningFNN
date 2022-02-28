# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:26:30 2021

@author: Jan
"""

#interesting observation, argument loss seems to do better at dicision boundary
from random import randint
import numpy as np


data = []
datafiltered = []

for i in range(2000):
    a = randint(-100,100)
    b = randint(-100,100)
    c = randint(-100,100)
    y = (a>b)
    #helper variables
    ag = a>0
    al = a<0
    bg = b>0
    bl = b<0
    # features a,b,c, class y0, class y1     
    data.append([a,b,c,int(y),int(not(y)),
                0,0,0]) #args
    
    #just hardcode it, because lazy:
    # if a>0 then its a positive argument if y=1 (a>b), else for y=0 its a neg arg
    if(ag):
        if(y):
            data[i][5]=1
        else:
            data[i][5]=-1
    # if a<0 then its a negative argument if y=1 (a>b), else for y=0 its a pos arg
    if(al):
        if(y):
            data[i][5]=-1
        else:
            data[i][5]=1
    # if b>0 then its a negative argument if y=1 (a>b), else for y=0 its a pos arg
    if(bg):
        if(y):
            data[i][6]=-1
        else:
            data[i][6]=1
    # if b<0 then its a positive argument if y=1 (a>b), else for y=0 its a neg arg
    if(bl):
        if(y):
            data[i][6]=1
        else:
            data[i][6]=-1


c0=0
c1=0
maxn = 400

for x in data:
    if(x[3]>0):
        if(c1<maxn):
            c1+=1
            datafiltered.append(x)
    else:
        if(c0<maxn):
            c0+=1
            datafiltered.append(x)




with open ("out.txt", "w", encoding='utf-8') as file:
	for x in datafiltered:
		file.write("" + str(x[0]) + "\t" + str(x[1]) + "\t" + str(x[2]) + "\t" + str(x[3]) + "\t" + str(x[4])+ "\t" )
		file.write(str(x[5]) + "\t" + str(x[6]) + "\t" + str(x[7]) + "\n")