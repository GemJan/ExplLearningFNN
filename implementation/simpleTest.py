# -*- coding: utf-8 -*-
"""
@author: Jan Gemander

compares multiple loss methods over a single dataset
evaluates accuracy of predictions
"""

import ExtendedLosses
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import layers
import numpy as np
import random as rn
from sklearn.metrics import accuracy_score
import HelperFunctions as HF


#inputset:         0               1                    2                        3 
#     set:   IntegerData      JapCreditData         germanCreditData        zooDataSet  
inputset =  1

#select methods that we want to compare
#default loss (0), shapley loss(1), gradient_loss(2) , integrated_gradient(3), sampling shapley(4),  
#RRC-reasoning-module-loss (5), SS_pos(6), SS_pen(7), SS_random_args (8), SS_inverted_args (9)
methods = [4,5,7]

#batch size during training
batch_size=4
#amount of training epochs
epochs=4

#lambda definition for weighing arguments: l_arg*argLoss
l_arg = 3
#lambda defintion for weighing cross entropy error: l_ce*CeLoss
l_ce = 1
#lambda term for weighing regularization error: l_reg**regLoss
l_reg = 0.02


use_normalisation = False

rn_state = rn.randint(0,2**32-1) #42 
tf.random.set_seed(rn_state)
np.random.seed(rn_state)
rn.seed(rn_state)

datadir = "../data/"


#read in data
#pddata contains data X
#pdlabels contains y* of x in X, corresponding indieces
#pdArgs contains A^* for every x in X
#pdInformation contains extended labelvector y_tilde = (y*, x, A^*) for every x in X
if(inputset == 0):
    file = "simpleIntegerData"
    pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
    featurecolumns = ["a","b","c"]
    argcolumns = ["arg1","arg2","arg3"]
    labelcolumns = ["y","y2"]
    #apply normalisation
    if(use_normalisation):
        pddata[featurecolumns] = pddata[featurecolumns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    pdfeatures = pddata[featurecolumns]
    pdlabels = pddata[labelcolumns]
    pdArgs = pddata[argcolumns]
    pdInformation = pddata[labelcolumns + featurecolumns + argcolumns]
    tfdata = tf.cast(pddata, tf.float32)
elif(inputset == 1):
    file = "creditDataJapan"
    pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
    featurecolumns = ["jobless","item","male","unmarried","probRegion","age","wealth","monthpay","numbermonths","yearscompany"]
    argcolumns = ["arg0","arg1","arg2","arg3","arg4","arg5","arg6","arg7","arg8","arg9",]
    labelcolumns = ["y0","y1"]
    #apply normalisation
    if(use_normalisation):
        pddata[featurecolumns] = pddata[featurecolumns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    pdfeatures = pddata[featurecolumns]
    pdlabels = pddata[labelcolumns]
    pdArgs = pddata[argcolumns]
    pdInformation = pddata[labelcolumns + featurecolumns + argcolumns]
    tfdata = tf.cast(pddata, tf.float32)
elif(inputset == 2):
    file = "creditDataGerman"
    pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
    pddata = pddata.rename(columns={"kredit": "y0"})
    #create two binary labels from one binary label
    pddata["y1"] = -pddata["y0"]+1
    featurecolumns = ["laufkont","laufzeit","moral","verw","hoehe","sparkont","beszeit","rate","famges","buerge",
                      "wohnzeit","verm","alter","weitkred","wohn","bishkred","beruf","pers","telef","gastarb"]
    argcolumns = ["arg" + str(i) for i in range(20)]
    labelcolumns = ["y0","y1"]
    #apply normalisation
    if(use_normalisation):
        pddata[featurecolumns] = pddata[featurecolumns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    pdfeatures = pddata[featurecolumns]
    pdlabels = pddata[labelcolumns]
    pdArgs = pddata[argcolumns]
    pdInformation = pddata[labelcolumns + featurecolumns + argcolumns]
    tfdata = tf.cast(pddata, tf.float32)
elif(inputset==3):
    file = "zooAnimals"
    pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
    featurecolumns = ["hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone",
                      "breathes","venomous","fins","legs","tail","domestic","catsize"]
    #amount of features and correconding arguments is 16
    argcolumns = ["arg" + str(i) for i in range(16)]
    #amount of labels is 7
    labelcolumns = ["y" + str(i) for i in range(7)]
    #convert type to onehot encoding for classification
    onehot = np.eye(7)[np.array(pddata["type"],dtype="int64")-1]
    pddata[labelcolumns] = onehot
    #apply normalisation
    if(use_normalisation):
        pddata[featurecolumns] = pddata[featurecolumns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    pdfeatures = pddata[featurecolumns]
    pdlabels = pddata[labelcolumns]
    pdArgs = pddata[argcolumns]
    pdInformation = pddata[labelcolumns + featurecolumns + argcolumns]
    tfdata = tf.cast(pddata, tf.float32)
    

#apply normalisation
if(use_normalisation):
    pddata[featurecolumns] = pddata[featurecolumns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    
m = len(pdlabels.columns)
n = len(pdfeatures.columns)

for method in methods:
    
    f = HF.initModel(output_shape=m)
    EL = ExtendedLosses.ExtendedLoss(f, n, m, lambda_ce = l_ce, lambda_arg=l_arg)
    
    #methods for all losses defined in Extended Losses
    if(method == 0):
        loss = EL.CE
    elif(method == 1):
        #shapley function takes very long for larger feature spaces
        if(n>6):
            continue    
        loss = EL.shapley_loss
    elif(method == 2):
        loss = EL.gradient_loss
    elif(method == 3):
        loss = EL.integrated_gradient_loss
    elif(method == 4):
        loss = EL.sampling_shapley_loss
    elif(method == 5):
        loss = EL.rrc_reasoning_loss
    elif(method == 6):
        loss = EL.sampling_shapley_loss_pos
    elif(method == 7):
        loss = EL.sampling_shapley_loss_pen
    elif(method == 8):
        loss = EL.sampling_shap_random_arguments
    elif(method == 9):
        loss = EL.sampling_shap_inverted_arguments
    else:
        continue
    
    #split data into training and testing data, yL = y* , yI = y_tilde = [x,y*,A*]
    X_train, X_test, yL_train, yL_test, yI_train, yI_test = train_test_split(
      pdfeatures,         pdlabels,        pdInformation, test_size=0.2, random_state=rn_state)
    
    f.compile(loss = loss, optimizer = tf.optimizers.Adam(), run_eagerly=False)
    
    print(loss)
    f.fit(X_train, yI_train, epochs=epochs, batch_size=batch_size, shuffle = False)
    print("accuarcy score train: " + str(accuracy_score(np.argmax(yL_train.values, axis=-1),np.argmax(f(X_train.values), axis=-1))))
    print("accuarcy score test: " + str(accuracy_score(np.argmax(yL_test.values, axis=-1),np.argmax(f(X_test.values), axis=-1))))
