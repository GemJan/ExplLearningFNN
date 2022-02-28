# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:06:42 2021

@author: Jan
"""
import numpy as np
import pandas as pd
import tensorflow as tf



datadir = "../data/"
file = "creditDataGerman"
pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
pddata = pddata.rename(columns={"kredit": "y0"})
pddata["y1"] = -pddata["y0"]+1
featurecolumns = ["laufkont","laufzeit","moral","verw","hoehe","sparkont","beszeit","rate","famges","buerge",
                  "wohnzeit","verm","alter","weitkred","wohn","bishkred","beruf","pers","telef","gastarb"]
pdfeatures = pddata[featurecolumns]
pdlabels = pddata[["y0","y1"]]
tfdata = tf.cast(pddata, tf.float32)

N = len(pdfeatures.columns)

Arg = np.zeros((len(pdfeatures), N))



# laufkont,   laufzeit,   moral,   verw,   hoehe,   sparkont,   beszeit,   rate,   famges,   buerge,
#     0          1          2       3        4         5          6         7        8         9
# wohnzeit,   verm,   alter,   weitkred,   wohn,   bishkred,   beruf,   pers,   telef,   gastarb
#    10        11       12       13         14         15      16       17       18        19        

for i in range(len(pdfeatures)):
    x = pdfeatures[i:i+1].values[0]
    y = pdlabels[i:i+1].values[0]
    laufkont,laufzeit,moral,verw,hoehe,sparkont,beszeit,rate,famges,buerge,wohnzeit,verm,alter,weitkred,wohn,bishkred,beruf,pers,telef,gastarb = x
    
            
    #arguments mainly based on point system described at https://data.ub.uni-muenchen.de/23/1/DETAILS.html
    
    if(laufkont >2):
        if(y[0]==1):
            Arg[i,0]=1
        else:
            Arg[i,0]=-1
    else:
        if(y[0]==1):
            Arg[i,0]=-1
        else:
            Arg[i,0]=1
    
    if(laufzeit <= 24):
        if(y[0]==1):
            Arg[i,1]=1
        else:
            Arg[i,1]=-1
    elif(laufzeit >= 30):
        if(y[0]==1):
            Arg[i,1]=-1
        else:
            Arg[i,1]=1
            
    if(moral > 2):
        if(y[0]==1):
            Arg[i,2]=1
        else:
            Arg[i,2]=-1
    elif(moral < 2):
        if(y[0]==1):
            Arg[i,2]=-1
        else:
            Arg[i,2]=1
            
    if(verw > 6):
        if(y[0]==1):
            Arg[i,3]=1
        else:
            Arg[i,3]=-1
    elif(verw < 4):
        if(y[0]==1):
            Arg[i,3]=-1
        else:
            Arg[i,3]=1
    
    if(hoehe <= 5000):
        if(y[0]==1):
            Arg[i,4]=1
        else:
            Arg[i,4]=-1
    elif(hoehe >= 5000):
        if(y[0]==1):
            Arg[i,4]=-1
        else:
            Arg[i,4]=1
    
    if(sparkont > 2):
        if(y[0]==1):
            Arg[i,5]=1
        else:
            Arg[i,5]=-1
    elif(sparkont <= 2):
        if(y[0]==1):
            Arg[i,5]=-1
        else:
            Arg[i,5]=1
    
    if(beszeit > 3):
        if(y[0]==1):
            Arg[i,6]=1
        else:
            Arg[i,6]=-1
    elif(beszeit <= 2):
        if(y[0]==1):
            Arg[i,6]=-1
        else:
            Arg[i,6]=1
            
#   rate seems to be a bad predictor in this data   
#    if(rate > 2):
#        if(y[0]==1):
#            Arg[i,7]=1
#        else:
#            Arg[i,7]=-1
#    elif(rate <= 2):
#        if(y[0]==1):
#            Arg[i,7]=-1
#        else:
#            Arg[i,7]=1
            
    if(famges > 2):
        if(y[0]==1):
            Arg[i,8]=1
        else:
            Arg[i,8]=-1
    elif(famges <= 2):
        if(y[0]==1):
            Arg[i,8]=-1
        else:
            Arg[i,8]=1
    
    #no observable impact of no additional brokers (which make up a big part of total)
    if(buerge > 2):
        if(y[0]==1):
            Arg[i,9]=1
        else:
            Arg[i,9]=-1
    
    #doesnt rly make that much of a difference 
#    if(wohnzeit > 2):
#        if(y[0]==1):
#            Arg[i,10]=1
#        else:
#            Arg[i,10]=-1
#    elif(wohnzeit <= 2):
#        if(y[0]==1):
#            Arg[i,10]=-1
#        else:
#            Arg[i,10]=1
    
    #results actually counter intuitive, where people with property
    #are less likely to pay off credit than people with no prior properties
#    if(verm	 > 2):
#        if(y[0]==1):
#            Arg[i,11]=1
#        else:
#            Arg[i,11]=-1
#    elif(verm	 <= 2):
#        if(y[0]==1):
#            Arg[i,11]=-1
#        else:
#            Arg[i,11]=1
    
    if(alter	 > 2):
        if(y[0]==1):
            Arg[i,12]=1
        else:
            Arg[i,12]=-1
    elif(alter	 < 2):
        if(y[0]==1):
            Arg[i,12]=-1
        else:
            Arg[i,12]=1
            
# laufkont,   laufzeit,   moral,   verw,   hoehe,   sparkont,   beszeit,   rate,   famges,   buerge,
#     0          1          2       3        4         5          6         7        8         9
# wohnzeit,   verm,   alter,   weitkred,   wohn,   bishkred,   beruf,   pers,   telef,   gastarb
#    10        11       12       13         14         15      16       17       18        19        


    if(weitkred	 > 2):
        if(y[0]==1):
            Arg[i,13]=1
        else:
            Arg[i,13]=-1
    elif(weitkred	 < 2):
        if(y[0]==1):
            Arg[i,13]=-1
        else:
            Arg[i,13]=1
            
            
    if(wohn	 >= 2):
        if(y[0]==1):
            Arg[i,14]=1
        else:
            Arg[i,14]=-1
    elif(wohn	 < 2):
        if(y[0]==1):
            Arg[i,14]=-1
        else:
            Arg[i,14]=1
            
    
    if(bishkred	 > 2):
        if(y[0]==1):
            Arg[i,15]=1
        else:
            Arg[i,15]=-1
    elif(bishkred	< 2):
        if(y[0]==1):
            Arg[i,15]=-1
        else:
            Arg[i,15]=1
 
    #again no observable difference           
#    if(beruf	 > 2):
#        if(y[0]==1):
#            Arg[i,16]=1
#        else:
#            Arg[i,16]=-1
#    elif(beruf	<= 2):
#        if(y[0]==1):
#            Arg[i,16]=-1
#        else:
#            Arg[i,16]=1
            
    #new feature same story
#    if(pers	 > 1):
#        if(y[0]==1):
#            Arg[i,17]=1
#        else:
#            Arg[i,17]=-1
#    elif(pers	< 2):
#        if(y[0]==1):
#            Arg[i,17]=-1
#        else:
#            Arg[i,17]=1            


    if(telef	 > 1):
        if(y[0]==1):
            Arg[i,18]=1
        else:
            Arg[i,18]=-1
    elif(telef	< 2):
        if(y[0]==1):
            Arg[i,18]=-1
        else:
            Arg[i,18]=1 
    
    #data shows foreign workers to be more likely to repay       
#    if(gastarb	 > 1):
#        if(y[0]==1):
#            Arg[i,19]=1
#        else:
#            Arg[i,19]=-1
#    elif(gastarb < 2):
#        if(y[0]==1):
#            Arg[i,19]=-1
#        else:
#            Arg[i,19]=1
            

            
with open("out.txt", "w") as f:
    for x in Arg:
        for value in x:
            f.write(str(value)+"\t")
        f.write("\n")
        
        
        
    
     