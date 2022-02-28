# -*- coding: utf-8 -*-
"""
@author: Jan

compares multiple loss methods, over mulitple datasets and epochs
averages over a defined amount of random sets
evaluates accuracy of predictions, cosine similarities of explanations and arguments, and f1 scores of categorised explanations
"""
import numpy as np
import pandas as pd
import random as rn

import tensorflow as tf
import ExtendedLosses
import HelperFunctions as HF

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as sc



#number of randomstates, used 40 in thesis
I = 5

#random states for tensorflow, sampling and other random number generators, for reproducible results (sort of, still not 100%) 
tf_states = []
sampling_states = []
rn_states = []
for i in range(I):
    tf_states.append(rn.randint(0,2**32-1))
    sampling_states.append(rn.randint(0,2**32-1))
    rn_states.append(rn.randint(0,2**32-1))

#batch size used during training
batch_size = 4
#amount of training epochs
epochs = 4

#inps contains inputs we want to look at
#inputset:         0               1                    2                        3 
#     set:   IntegerData      JapCreditData         germanCreditData        zooDataSet  
inps = [0,1,2,3]

#select methods that we want to compare
#default loss (0), shapley loss(1), gradient_loss(2) , integrated_gradient(3), sampling shapley(4),  
#RRC-reasoning-module-loss (5), SS_pos(6), SS_pen(7), SS_random_args (8), SS_inverted_args (9)
#SS-Baseline (10), SS-IG-explanations (10), SS ommitting CE and L1-Regularisation (11)
methods = [0,3,4,5,11]


#lambda definition for weighing arguments: l_arg*argLoss
l_arg = 3
#lambda defintion for weighing cross entropy error: l_ce*CeLoss
l_ce = 1
#lambda term for weighing regularization error: l_reg**regLoss
l_reg = 0.02

#we have omitted normalisation of training data in all our tests
use_normalisation = False

saveResults = False
#output Directory and filename
output = "results"
#directory of dataset files
datadir = "../data/" 


#we store our results over datasets, epochs, methods and random iterations in the following tensors
#initialised with zeros, computed values can be added later
accuraciesTrain = np.zeros((len(inps),epochs,len(methods), I))
accuraciesTest = np.zeros((len(inps),epochs,len(methods), I))

cosSimTrain = np.zeros((len(inps),epochs,len(methods), I))
cosSimTest = np.zeros((len(inps),epochs,len(methods), I))

Ef1Train = np.zeros((len(inps),epochs, len(methods), I,3))
Ef1Test = np.zeros((len(inps),epochs, len(methods), I,3))

#iterate over selected inputs
for inp_i, inputset in enumerate(inps):
    
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
    else:
        continue
        
    
    
    #amount of labels
    m = len(pdlabels.columns)
    #amount of features
    n = len(pdfeatures.columns)
    
    ############################## compute a Baseline #########################
    
    sums = np.zeros((m,n))
    counts = np.zeros(m)
    
    #iterate over all datapoints
    for i in range(len(pdlabels)):
        for j in range(m):
            if(pdlabels.iloc[i,j]==1):
                counts[j] += 1
                sums[j] += tf.cast(pdfeatures[i:i+1].values, tf.float32)
    #we compute average of every class seperately and average them to account for bias 
    b = tf.cast(tf.reduce_sum((sums/counts[:,None]),axis=-2)/m,tf.float32)
    ###########################################################################
    
    #iterate over selected methods
    for j, method in enumerate(methods):
        
        #iterate over random states
        for i in range(I):
            
            #set all seeds
            tf_state = tf_states[i]
            tf.random.set_seed(tf_state)
            rn_state = rn_states[i]
            np.random.seed(rn_state)
            rn.seed(rn_state)
            sampling_state = sampling_states[i]
            
            #initialise model f
            f = HF.initModel(output_shape=m, lambda_regularisation=l_reg)
            #define EL weights
            EL = ExtendedLosses.ExtendedLoss(f, n, m, lambda_ce = l_ce, lambda_arg=l_arg)
            
            #explanation function for evaluation of results
            expf = HF.get_shapley_explan
            
            #per default we don't use a baseline
            baseline=None
        
            #specify all loss variants as a method, so that we can iterate over it
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
            elif(method == 10):
                #example of how to incorporate computed baseline in losses:
                EL = ExtendedLosses.ExtendedLoss(f, n, m, baseline=b)
                #incorporate baseline in explanation function for evaluation used below:
                baseline = b
                loss = EL.sampling_shapley_loss
            elif(method == 11):
                #example of how to use different explanation function for evaluation
                expf = HF.get_IG_explan
                #does not effect loss in any way
                loss = EL.sampling_shapley_loss
            elif(method == 12):
                #example of how to omit regularisation
                f = HF.initModel(output_shape=m, lambda_regularisation=0)
                #example of how to omit cross entropy:
                EL = ExtendedLosses.ExtendedLoss(f, n, m, lambda_ce=0)
                #final loss only contains sampled shapley argument loss
                loss = EL.sampling_shapley_loss
            else:
                continue
            
            
            print(str(i) + "th random state of data " + file + " |loss  " +  str(loss))  
            
            #split data into training and testing data, yL = y* , yI = y_tilde = [x,y*,A*]
            X_train, X_test, yL_train, yL_test, yI_train, yI_test = train_test_split(
                pdfeatures,       pdlabels,        pdInformation, test_size=0.2, random_state=sampling_state)
            
            f.compile(loss = loss, optimizer = tf.optimizers.Adam())

            #iterate over epochs
            for ep in range(epochs):
                #train one epoch
                f.fit(X_train, yI_train, epochs=1, batch_size=batch_size)
                              
                
                #accuracies of predictions
                accTrain = accuracy_score(np.argmax(yL_train.values, axis=-1),np.argmax(f(X_train.values), axis=-1))
                accTest = accuracy_score(np.argmax(yL_test.values, axis=-1),np.argmax(f(X_test.values), axis=-1))
                
                accuraciesTest[inp_i, ep,j,i] += accTest
                accuraciesTrain[inp_i, ep,j,i] += accTrain
                
                
                
                #generate explanations
                explan = expf(X_train.values,yL_train.values,f,baseline)
                #corresponding arguments
                args = yI_train.values[:,m+n:m+2*n]
                
                #cosinus similarity between explanations and arguments
                cosSimTrain[inp_i, ep,j,i] = HF.get_cosine_sim(explan, args) 
                
                #scores of categorised explanations and arguments
                precision, recall, fscore, support = sc(np.matrix.flatten(HF.categorise_explan(explan).numpy()),
                                                    np.matrix.flatten(args),labels=[-1,0,1],zero_division=0)
                #save f1 score
                Ef1Train[inp_i,ep,j,i] = fscore

                #redo above for testing set
                HF.get_cosine_sim(explan, args) 
                explan = expf(X_test.values,yL_test.values,f,baseline)
                args = yI_test.values[:,m+n:m+2*n]
                
                cosSimTest[inp_i, ep,j,i] = HF.get_cosine_sim(explan, args)  
                
                precision, recall, fscore, support = sc(np.matrix.flatten(HF.categorise_explan(explan).numpy()),
                                                    np.matrix.flatten(args),labels=[-1,0,1],zero_division=0)
                       
                Ef1Test[inp_i,ep, j,i] = fscore

    print(str((inp_i+1)/len(inps)*100) + "% of datasets done")


#first dimension is input set
#second dimension is different epoch lengths
#third dimension is method used for training (eg cce, shapley loss etc)
#fourth dimension is repeated iterations, in the following we reduce to the mean
#print("accuraciesTrain:")
#print(tf.reduce_mean(accuraciesTrain,axis=-1).numpy())
#print("accuraciesTest:")
#print(tf.reduce_mean(accuraciesTest,axis=-1).numpy())
#print("cosine similarities train:")
#print(tf.reduce_mean(cosSimTrain,axis=-1).numpy())
#print("cosine similarities test:")
#print(tf.reduce_mean(cosSimTest,axis=-1).numpy())
#print("f1 Scores categorised eplanations test:")
#print("cosine similarities train:")
#print(tf.reduce_mean(Ef1Train,axis=-2).numpy())
#print("cosine similarities test:")
#print(tf.reduce_mean(Ef1Test,axis=-2).numpy())


#write results to file
if(saveResults):
    with open(output + ".npy", 'wb') as out:
        np.save(out, accuraciesTrain)
        np.save(out, accuraciesTest)
        np.save(out, cosSimTrain)
        np.save(out, cosSimTest)
        np.save(out, Ef1Train)
        np.save(out, Ef1Test)