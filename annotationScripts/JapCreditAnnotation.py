# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:06:42 2021

@author: Jan
"""
import numpy as np
import pandas as pd
import tensorflow as tf



datadir = "../data/"
file = "creditDataJapan"
pddata = pd.read_excel(datadir + file + ".xls", dtype="float64")
pdfeatures = pddata[["jobless","item","male","unmarried","probRegion","age","wealth","monthpay","numbermonths","yearscompany"]]
pdlabels = pddata[["y0","y1"]]

tfdata= tf.cast(pddata, tf.float32)

N = len(pdfeatures.columns)

Arg = np.zeros((len(pdfeatures), N))

# Rules from ABML paper
#IF problem_region=yes AND monthly_payment<= 9 THEN credit=no
#IF jobless=yes AND money_bank<= 50 AND monthly_payment> 2.0 THEN credit=no
#IF item=bike AND money_bank<= 10 THEN credit=no
#
#IF problem_region=yes AND years_work<= 10 THEN credit=no
#IF jobless=yes AND sex=male THEN credit=no
#IF sex=female AND jobless=yes AND enough_money=no THEN credit=no
#IF age>59 AND years_working<3 THEN credit=no
#IF jobless=yes AND sex=female AND married=no THEN credit=no
#IF enough_money=no AND age<=19 THEN credit_approved=no
#IF item=bike AND sex=female THEN credit=no

#jobless	item	male	unmarried	probRegion	age	    wealth	monthpay	numbermonths	yearscompany
# 0          1       2       3              4         5       6       7                8        9        

for i in range(len(pdfeatures)):
    x = pdfeatures[i:i+1].values[0]
    y = pdlabels[i:i+1].values[0]
    jobless,item,male,unmarried,probRegion,age,wealth,monthpay,numbermonths,yearscompany = x
    
            
    #OWN Positive ADDITIONS ############################
    
    if(wealth >(monthpay*numbermonths-20)):
        if(y[0]==1):
            Arg[i,6]=1
            Arg[i,7]=1
            Arg[i,8]=1
        else:
            Arg[i,6]=-1
            Arg[i,7]=-1
            Arg[i,8]=-1
            
    if(jobless == 0 and age in range(25,52)):
        if(y[0]==1):
            Arg[i,0]=1
            Arg[i,5]=1
        else:
            Arg[i,0]=-1
            Arg[i,5]=-1
    
    if((monthpay*numbermonths)>wealth*4):
        if(y[0]==1):
            Arg[i,6]=-1
            Arg[i,7]=-1
            Arg[i,8]=-1
        else:
            Arg[i,6]=1
            Arg[i,7]=1
            Arg[i,8]=1
    if(age > 58):
        if(y[0]==1):
            Arg[i,5]=-1
        else:
            Arg[i,5] =1
            
    ########################## Excluding conditions in ABML paper
    #IF problem_region=yes AND monthly_payment<= 9 THEN credit=no
    if(probRegion==1 and monthpay<=9):
        if(y[0]==1):
            Arg[i,4]=-1
            Arg[i,7]=-1
        else:
            Arg[i,4]=1
            Arg[i,7]=1
    #IF jobless=yes AND money_bank<= 50 AND monthly_payment> 2.0 THEN credit=no
    if(jobless == 1 and wealth<=50 and monthpay>2):
        if(y[0]==1):
            Arg[i,0]=-1
            Arg[i,6]=-1
            Arg[i,7]=-1
        else:
            Arg[i,0]=1
            Arg[i,6]=1
            Arg[i,7]=1
    #IF item=bike AND money_bank<= 10 THEN credit=no
    if(item==5 and wealth <= 10):
        if(y[0]==1):
            Arg[i,1]=-1
            Arg[i,6]=-1
        else:
            Arg[i,1]=1
            Arg[i,5]=1
    #IF problem_region=yes AND years_work<= 10 THEN credit=no
    if(probRegion==1 and yearscompany<=10):
        if(y[0]==1):
            Arg[i,4]=-1
            Arg[i,9]=-1
        else:
            Arg[i,4]=1
            Arg[i,9]=1
    #IF jobless=yes AND sex=male THEN credit=no
    if(jobless==1 and male==1):
        if(y[0]==1):
            Arg[i,0]=-1
            Arg[i,2]=-1
        else:
            Arg[i,0]=1
            Arg[i,2]=1
            
    #jobless	item	male	unmarried	probRegion	age	    wealth	monthpay	numbermonths	yearscompany
    # 0          1       2       3              4         5       6       7                8        9   
    #IF sex=female AND jobless=yes AND enough_money=no THEN credit=no
    if(jobless == 1 and male == 0 and wealth < numbermonths*monthpay):
        if(y[0]==1):
            Arg[i,0]=-1
            Arg[i,2]=-1
            Arg[i,6]=-1
            Arg[i,7]=-1
            Arg[i,8]=-1
        else:
            Arg[i,0]=1
            Arg[i,2]=1
            Arg[i,6]=1
            Arg[i,7]=1
            Arg[i,8]=1
    #IF age>59 AND years_working<3 THEN credit=no
    if(age > 59 and yearscompany<3):
        if(y[0]==1):
            Arg[i,5]=-1
            Arg[i,9]=-1
        else:
            Arg[i,5]=1
            Arg[i,9]=1
    
    #IF jobless=yes AND sex=female AND married=no THEN credit=no
    if(jobless==1 and male==0 and unmarried==1):
        if(y[0]==1):
            Arg[i,0]=-1
            Arg[i,2]=-1
            Arg[i,3]=-1
        else:
            Arg[i,0]=1
            Arg[i,2]=1
            Arg[i,3]=1
    #IF enough_money=no AND age<=19 THEN credit_approved=no
    if(wealth < numbermonths*monthpay and age<=19):
        if(y[0]==1):
            Arg[i,5]=-1
            Arg[i,6]=-1
            Arg[i,7]=-1
            Arg[i,8]=-1
        else:
            Arg[i,5]=1
            Arg[i,6]=1
            Arg[i,7]=1
            Arg[i,8]=1
        
    #IF item=bike AND sex=female THEN credit=no
    if(item == 5 and male==0):
        if(y[0]==1):
            Arg[i,1]=-1
            Arg[i,2]=-1
        else:
            Arg[i,1]=1
            Arg[i,2]=1
            
with open("out.txt", "w") as f:
    for x in Arg:
        for value in x:
            f.write(str(value)+"\t")
        f.write("\n")
        
        
        
    
     