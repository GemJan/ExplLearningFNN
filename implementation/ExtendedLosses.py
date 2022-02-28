# -*- coding: utf-8 -*-
"""
@author: Jan Gemander

defines our extended loss, includes all loss variants as a loss method
"""
import tensorflow as tf
import numpy as np
import itertools
import math
import ContributionFunctions as CF

cce = tf.keras.losses.CategoricalCrossentropy()

class ExtendedLoss():
    
    
    def __init__(self, tensorflow_model, featureLength, outputLength, lambda_ce=1, lambda_arg=3, baseline=None):
        "Initialises loss for a model with weighing factors"
        self.f = tensorflow_model
        self.baseline = baseline
        self.l1 = tf.cast(lambda_arg, tf.float32)
        self.l0 = tf.cast(lambda_ce, tf.float32)
        self.n = featureLength
        self.m = outputLength
        
    def get_network(self):
        "Returns Network used in extended Loss"
        return self.f
    
    def CE(self, y_true, y_pred):
        """ Computes Cross entropy Loss
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy (+ regularisation loss [is added outside of custom Loss by Tensorflow])
        """
        m = self.m
        #get y*
        y = y_true[:,0:m] 
        
        
        return self.l0*cce(y,y_pred)
        
    def shapley_loss_legacy(self,y_true, y_pred):
        """Legacy implementation to compute Shapley Loss, iterates over features,
        computes shapley values for arguments and adds these to the loss.
        Only works for batch_size = 1, Does not implement baseline. 
        Not recommended, a more flexible implementation can be found below.
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #set off all possible subsets
        possibleSubsets = np.array(list(itertools.product("01",repeat=n)), dtype="float32")
        
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        #iterate over arguments
        for i in range(self.n):
            #if a feature is a pos arg continue
            if args[:,i] != 0:
                shapley = np.array([0.,0.], dtype="float32")
                #iterate over possible subsets
                for S in possibleSubsets:
                    if(S[i] == 0):
                        #Sux_i is subset S \union x_i
                        Sux_i = S.copy()
                        Sux_i[i] = 1
                        #subsetsize = amount of 1s in subset S
                        subsetsize = sum(S)
                        #fac = (subsetsize!*(n-subsetsize-1)!)/n!
                        #n is given by length of feature vector
                        fac = (math.factorial(subsetsize)*math.factorial(n-subsetsize-1))/math.factorial(n)
                        #sh_i = \sum_subsets fac*(f(subset \union x_i)-f(subset)
                        #we mask values outside of subset by multiplying them with 0
                        shapley += fac*(self.f(x*Sux_i)-self.f(x*S))
                #this ensures that we add the shapley value for the correct class
                #y is a 1 hot vector where only y* == 1
                shapley=shapley*y
                        
                #subtract weighted  shapley values from loss, if we have a negative argument then args[:,i] = -1, and shapley value gets added instead
                #sum is used to reduce loss to a single value (all contributions to y which are not y* are 0 from the above call)
                loss -= tf.math.reduce_sum((lArg*shapley*args[:,i]))
    
        #weight regularizaion, is apperently added outside of custom loss, src https://stackoverflow.com/questions/67912239/custom-loss-function-with-regularization-cost-added-in-tensorflow
        #loss += sum(f.losses)
    
        return loss

    def shapley_loss(self, y_true, y_pred):
        """Computes Shapley Loss, uses get_shapley from contribution functions.
        
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]]  
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
    
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        #compute shapley values
        shap = CF.get_shapley(x,self.f,self.baseline)
        #zero out contributions towards classes that we're not interested in
        argLoss = tf.reduce_sum(shap*y[:,None],axis=-1)
        #select arguments as described in thesis, inverting contributions of negative arguments,
        #keep those in positive arguments, remove others by setting them to 0
        #take negative and reduce to sum to recieve argument loss defined in our thesis
        argLoss = -tf.math.reduce_sum(argLoss*args)
        
        #subtract weighted contributions, thus rewarding correct ones and penalising false ones
        loss += lArg*argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss

    def sampling_shapley_loss(self,y_true, y_pred):
        """Computes sampled Shapley Loss, uses get_sampling_shapley from contribution functions.
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
    
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        #get shapley values and only use contributions for the correct output
        shap = CF.get_sampling_shapley(x,self.f,self.baseline)
        
        #select correct class
        argLoss = tf.reduce_sum(shap*y[:,None],axis=-1)        
        
        #select arguments as described in thesis, inverting contributions of negative arguments,
        #keep those in positive arguments, remove others by setting them to 0
        #take negative and reduce to sum to recieve argument loss defined in our thesis
        argLoss = -tf.math.reduce_sum(argLoss*args)
        
        #add weighted argumentLoss
        loss += lArg*argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss

    def gradient_loss(self, y_true, y_pred):
        """Computes gradient loss, using get_gradients from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #cross entropy error
        loss =  lCce*tf.cast(cce(y,y_pred), dtype="float32")
        
        # gradient values, restricted to correct class 
        #(jacobian method below is more akin to gradient variant described in thesis but slower)
        grads = CF.get_gradient(x, y, self.f)
        
        
        #select arguments as described in thesis, inverting contributions of negative arguments,
        #keep those in positive arguments, remove others by setting them to 0
        #take negative and reduce to sum to recieve argument loss defined in our thesis
        argLoss = -tf.math.reduce_sum(grads*args)
    
        #subtract weighted contributions, thus rewarding correct ones and penalising false ones
        loss += (lArg*argLoss)
    
        #weight regularization, is already added outside of custom loss in tensorflow
        #loss += sum(f.losses)
            
        return loss
    
    def jacobian_loss(self, y_true, y_pred):
        """Computes gradient loss, using get_jacobians from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #cross entropy error
        loss =  lCce*tf.cast(cce(y,y_pred), dtype="float32")
        #get gradients of all predictions with respect to inputs
        grads = CF.get_jacobian(x,self.f)
        
        #select arguments as described in thesis, inverting contributions of negative arguments,
        #keep those in positive arguments, remove others by setting them to 0
        argLoss = tf.reduce_sum(grads*args[:,None],axis=-1)
        #restrict towards correct class, take negative and reduce to sum to recieve argument loss defined in our thesis
        argLoss = - tf.reduce_sum(argLoss*y)
    
        #subtract weighted contributions, thus rewarding correct ones and penalising false ones
        loss += (lArg*argLoss)
    
        #weight regularization, is already added outside of custom loss in tensorflow
        #loss += sum(f.losses)
            
        return loss
    
    def integrated_gradient_loss(self, y_true, y_pred):
        """Computes integrated gradient loss, using get_integrated_gradients from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #cross entropy error
        loss =  lCce*tf.cast(cce(y,y_pred), dtype="float32")
        
        # integrated gradient values, restricted to correct class 
        #(integrated jacobian method below is more akin to integrated gradient variant described in thesis but slower)
        IntGrads = CF.get_integrated_gradients(x, y, self.f, self.baseline)
        #apply arguments, take negative and sum to recieve argument loss described in our thesis
        argLoss = -tf.math.reduce_sum(IntGrads*args)
    
        #subtract weighted contributions, thus rewarding correct ones and penalising false ones
        loss += (lArg*argLoss)
    
        #weight regularization, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
            
        return loss
    
    def integrated_jacobian_loss(self, y_true, y_pred):
        """Computes integrated gradient loss, using get_integrated_jacobians from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #cross entropy error
        loss =  lCce*tf.cast(cce(y,y_pred), dtype="float32")
        
        #integrated gradients for all predictions towards all classes
        IntGrads = CF.get_integrated_jacobians(x, self.f, self.baseline)
        #apply arguments
        argLoss = tf.reduce_sum(IntGrads*args[:,None],axis=-1)
        #select correct class, take negative and sum to recieve argument loss described in our thesis
        argLoss = -tf.reduce_sum(argLoss*y)
        
    
        #subtract weighted contributions, thus rewarding correct ones and penalising false ones
        loss += (lArg*argLoss)
    
        #weight regularization, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
            
        return loss
    
    def rrc_reasoning_loss(self, y_true, y_pred):
        """Computes RRC-Reasoning loss, using get_integrated_gradients from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        #RRR loss has a matrix A, where features that should be of no releveance are marked with a 1, and all others with a 0
        #we achieve this by first removing Arguments using maximum ([-1,0,1] turns to [0,0,1])
        #and then swap positive features with others ([0,0,1] -> [1,1,0])
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")   
        args = tf.maximum(0.,args)
        args = 1-args
        
        #cross entropy error
        loss = lCce*cce(y,y_pred)
    
        #integrated gradients of predictions for correct class with respect to inputs
        IntGrads = CF.get_integrated_gradients(x, y, self.f, self.baseline)
        #apply arguments, recieving integrated gradients of features that should not have contributions
        argLoss = IntGrads*args
        #In equation 3 of the paper  "Right for the Right Concept" ( https://arxiv.org/abs/2011.12854 )
        #the authors state that they apply a zero threshhold to only recieve positive importance,
        #they do this by applying a minimum function, here i assume this was an error and use the maximum function instead
        argLoss = tf.math.reduce_sum(tf.maximum(argLoss, 0))
        
        #Contributions of features not in positive Arguments are added to loss
        loss += lArg * argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss
    
    
    
    
    ######################################### Custom Losses Alterations #############################################
    # eg penalisation variant, pos variant, random arguments, inverted arguments, can be found here
    
    
    
    
    
    def sampling_shapley_loss_pen(self, y_true, y_pred):
        """Computes the shapley loss penalising negative contributions of x_i \notin A^- and positive contributions of x_i \notin A^+
            using get_sampling_shapley from contribution functions
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #positive arguments = 1, rest are zero
        pos = tf.maximum(0., args)
        #negative arguments = -1, rest are zero
        neg = tf.minimum(0., args)
        
    
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        #get shapley values
        shap = CF.get_sampling_shapley(x,self.f)
        
        #select correct class
        shap = tf.reduce_sum(shap*y[:,None],axis=-1)        
        
        #penalise positive contributions of features not in A+ (1-pos args), where pos args are 
        argLoss = tf.math.reduce_sum(tf.maximum(0.,(1-pos)*shap))
        #(-1-neg) turns prev neg arguments to 0 and all others to -1
        #(-1-neg)*shap) thus negates contributions of prev non negative features (x_i \not in A^-)
        argLoss += tf.math.reduce_sum(tf.maximum(0.,(-1-neg)*shap))
        
        #added for penalisation
        loss += lArg*argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss

    def sampling_shapley_loss_pos(self,y_true, y_pred):
        """Computes the shapley loss only looking at positive arguments
            using get_sampling_shapley from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        
        #recieve positive arguments
        pos = tf.maximum(0., args)
        
    
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        #get shapley values and only use contributions for the correct output
        shap = tf.reduce_sum(CF.get_sampling_shapley(x,self.f)*y[:,None],axis=-1)
        
        #reward positive contributions of features in A+ and penalise negative contributions by taking negative
        argLoss = -tf.math.reduce_sum(pos*shap)
        
        #weighted argument loss added
        loss += lArg*argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss
    
    def sampling_shap_random_arguments(self,y_true, y_pred):
        """Computes shapley loss using random arguments,
            using get_sampling_shapley from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #random arguments
        args = tf.cast(1-np.random.choice(3,n), tf.float32)
        
    
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        
        #get shapley values and only use contributions for the correct output
        argLoss = tf.reduce_sum(CF.get_sampling_shapley(x,self.f)*y[:,None],axis=-1)
        #apply random arguments
        argLoss = tf.math.reduce_sum(argLoss*args)
        
        loss -= lArg*argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss
    
    def sampling_shap_inverted_arguments(self,y_true, y_pred):
        """Computes the shapley loss using inverted arguments,
            using get_sampling_shapley from contribution functions
        
        Args:
            y_pred: network predictions for f(x)
            y__true: [[y1, ...,ym,x1, ...,xn,arg1, ..., argn]] 
            where argi==-1 -> is negative argument, argi==0 -> i is not an argument, argi==1 -> i is pos argument
        Returns:
            loss computed by cross entropy + argument loss  (+ regularisation loss)
        """
        #lambda for arguments and cross entropy
        lArg = self.l1
        lCce = self.l0
        #n = length of feature vector, m = length of output vector
        n = self.n
        m = self.m
        
        #additional information is appended to y_true
        #result vector y*
        y = y_true[:,0:m]
        #feature vector x
        x = y_true[:,m:m+n]
        #inverted arguments
        args = tf.cast(y_true[:,m+n:m+2*n], dtype="float32")
        args = 0-args
        
    
        #cross entropy error
        loss = lCce*cce(y,y_pred)
        
        
        #get shapley values and only use contributions for the correct output
        argLoss = tf.reduce_sum(CF.get_sampling_shapley(x,self.f)*y[:,None],axis=-1)
        #apply arguments
        argLoss = tf.math.reduce_sum(argLoss*args)
        
        loss -= lArg*argLoss
    
        #weight regularizaion, is apperently added outside of custom loss in tensorflow
        #loss += sum(f.losses)
    
        return loss
        