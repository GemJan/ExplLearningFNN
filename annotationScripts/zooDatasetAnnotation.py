# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:06:42 2021

@author: Jan
"""
import numpy as np
import pandas as pd
import tensorflow as tf


file = "zooAnimals"
pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
featurecolumns = ["hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone",
                  "breathes","venomous","fins","legs","tail","domestic","catsize"]
#amount of features is 16
argcolumns = ["arg" + str(i) for i in range(16)]
#amount of labels is 7
labelcolumns = ["y" + str(i) for i in range(7)]
#convert type to onehot encoding for classification
onehot = np.eye(7)[np.array(pddata["type"],dtype="int64")-1]
pddata[labelcolumns] = onehot
pdfeatures = pddata[featurecolumns]
pdlabels = pddata[labelcolumns]

tfdata = tf.cast(pddata, tf.float32)

N = len(pdfeatures.columns)

Arg = np.zeros((len(pdfeatures), N))

#mammal, bird, reptile, fish, amphibian, insect, and invertebrate
# y0      y1     y2      y3      y4       y5          y6

# hair	feathers	eggs	milk	airborne	aquatic	  predator	 toothed	backbone	
#  0       1         2       3         4          5         6          7           8
# breathes	venomous	fins	legs	tail	domestic	catsize
#   9          10        11      12      13        14          15


for i in range(len(pdfeatures)):
    x = pdfeatures[i:i+1].values[0]
    y = pdlabels[i:i+1].values[0]
    hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize = x
    
            
    #arguments based on rules in ABML paper
    #IF milk=yes THEN type=Mammal
	#IF fins=yes AND breathes=no THEN type=Fish
	#IF feathers=yes THEN type=Bird
	#IF legs=6 AND breathes=yes THEN type=Insect
	#IF legs=4 AND hair=no AND predator=yes THEN type=Amphibian
	#IF backbone=yes AND eggs=yes AND aquatic=no AND feathers=no THEN type=Reptile
	#IF eggs=no AND breathes=no THEN type=Reptile

    if(y[0] == 1):
        if(milk==1):
            Arg[i, 3] = 1
    if(y[1] == 1):
        if(feathers==1):
            Arg[i, 1] = 1
    if(y[2] == 1):
        if(backbone == 1 and eggs == 1 and aquatic == 0 and feathers == 0):
            Arg[i, 8] = 1
            Arg[i, 2] = 1
            Arg[i, 5] = 1
            Arg[i, 1] = 1
        if(eggs == 0 and breathes == 0):
            Arg[i, 2] = 1
            Arg[i, 9] = 1
    if(y[3] == 1):
        if(fins==1 and breathes == 0):
            Arg[i, 11] = 1
            Arg[i, 9] = 1
    if(y[4] == 1):
        if(legs == 4 and hair == 0 and predator == 1):
            Arg[i,12] = 1
            Arg[i,0] = 1
            Arg[i,6] = 1
        if(backbone==1 and aquatic == 1 and eggs == 1 and legs==4):
            Arg[i,8] = 1
            Arg[i,5] = 1
            Arg[i,2] = 1
            Arg[i,12] = 1
        
    if(y[5] == 1):
        if(legs == 6 and breathes == 1):
            Arg[i,12] = 1
            Arg[i, 9] = 1
            
    #invertebrate argument not included in ABML paper
    if(y[6]==1):
        if(backbone==0):
            Arg[i,8] = 1
        
    

            

            
with open("out.txt", "w") as f:
    for x in Arg:
        for value in x:
            f.write(str(value)+"\t")
        f.write("\n")
        
        
        
    
     