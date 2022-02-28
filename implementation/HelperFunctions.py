# -*- coding: utf-8 -*-
"""
@author: Jan Gemander

defines some helper functions for managing and evaluating our models
"""
import tensorflow as tf
import ContributionFunctions as CF
from tensorflow.keras import layers
from scipy import spatial
    

def initModel(output_shape,  lambda_regularisation=0.02, softmax_output = True):
    """ Creates model and returns it
    Args:
        output_shape: amount of possible labels for predictions
        lambda_regularisation: lambda to weigh L1 regularisation loss
        softmax_output: weather to use softmax output, classifciation without is not recommended
    Returns:
        created tensorflow model)
    """
    if(softmax_output):
        f = tf.keras.Sequential([
          layers.Dense(16,kernel_regularizer=tf.keras.regularizers.L1(lambda_regularisation),activation='relu'),
          layers.Dense(16,kernel_regularizer=tf.keras.regularizers.L1(lambda_regularisation),activation='relu'),
          layers.Dense(output_shape,kernel_regularizer=tf.keras.regularizers.L1(lambda_regularisation),activation='softmax')
        ])
    else:
        f = tf.keras.Sequential([
          layers.Dense(16,kernel_regularizer=tf.keras.regularizers.L1(lambda_regularisation),activation='relu'),
          layers.Dense(16,kernel_regularizer=tf.keras.regularizers.L1(lambda_regularisation),activation='relu'),
          layers.Dense(output_shape,kernel_regularizer=tf.keras.regularizers.L1(lambda_regularisation)),
        ])
    return f

    
def get_shapley_explan(x, y, model, baseline=None):
    """ computes sampled Shapley explanations for a label of a model on a input, 
        simply put sampled shapley comutation restricted to a label
    Args:
        x: input vector
        y: output vector
        model: model used for prediction
    Returns:
        sampled Shapley Explanation for a label of a model on a input)
    """
    expl = tf.reduce_sum(CF.get_sampling_shapley(x,model,baseline)*y[:,None],axis=-1)
    return expl

def get_IG_explan(x, y, model, baseline=None):
    """ computes integrated gradient explanations for a label of a model on a input, 
        get_integrated_gradients function already restricted to label, return that
    Args:
        x: input vector
        y: output vector
        model: model used for prediction
    Returns:
        sampled Shapley Explanation for a label of a model on a input)
    """
    expl = CF.get_integrated_gradients(x,y,model,baseline)
    return expl

def get_cosine_sim(explan, args):
    """ computes cosine similarities of two vectors, usually explanations and arguments
    Args:
        explan: vector given by explanations
        arg: vector given by arguments
    Returns:
        Cosine similarities of vectors)
    """
    c_sim = 0.
    counterZeroArgs = 0
    #in case we recieve multiple datapoints iterate over each and average later
    for k,c in enumerate(explan):
        arg = args[k]
        #ignore examples with no arguments
        if(sum(arg) == 0):
            counterZeroArgs += 1
            continue
        #use cosine distance        
        cos_dist = spatial.distance.cosine(arg, c)
        if(tf.math.is_nan(cos_dist)):
            #eg when division by zero, assume 0 similarity
            c_sim+=0
        else:
            c_sim+= (1-cos_dist)
    c_sim = c_sim/(explan.shape[0]-counterZeroArgs)
    return c_sim

def categorise_explan(explan, threshold = .15):
    """ categorises explanations into negative(-1), neutral(0) and positive(1)
    Args:
        explan: vector given by explanations
        threshold: used as theshhold for categories
    Returns:
        categorised explanations
    """
    #normalise explanations to a range [-1,1]
    explan = explan / tf.reduce_max(abs(explan))
    #select positive explanations
    pos = tf.cast(explan>(threshold),tf.float32)
    #select negative explanations
    neg = tf.cast(explan<(-threshold),tf.float32)
    args = pos - neg
    return args


def get_score(explan, args):
    """ legacy scoring function, not used in thesis.
        uses L1 norm to normalise explanation vector to an absolute sum of 1
        thus by summing up correct explanations we get percentage of correctly attributed explanations in, 
        where remaining percent are wrongly attributed exlpanations
    Args:
        explan: vector given by explanations
        args: vector given by arguments
    Returns:
        score of explanation w.r.t. arguments
    """
    explan = explan / tf.norm(explan,axis =-1, ord=1)[:,None]
    #there are cases where the above equation divides by 0, eg when x is equal to the baseline, then contributions don't differ from it
    #this results in nan values, which we replace here with 0s
    if(tf.reduce_any(tf.math.is_nan(explan))):
        contr = tf.where(tf.math.is_nan(explan), tf.zeros_like(explan), explan)
    score = tf.reduce_sum(tf.maximum((args>0)*contr,0)**2,axis=-1) - tf.reduce_sum(tf.minimum((args<0)*explan,0)**2,axis=-1)
    return tf.reduce_mean(score)