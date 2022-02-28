# -*- coding: utf-8 -*-
"""
@author: Jan Gemander

define contribution functions used to compute explanations
"""
import tensorflow as tf
import numpy as np
import itertools
from scipy.special import factorial




def get_gradient(x, yt, model):
    """Computes gradients for a label (usually correct class)

    Args:
        x: input tensor
        yt: class one hot vector for x (=y*)

    Returns:
        Gradients of predictions w.r.t input
    """
    x = tf.cast(x, tf.float32)
    yt = tf.cast(yt, tf.float32)

    #gradient tape for model
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
        #zero out all wrong classes (.gradient function sums up over predictions, 
        #which in case of softmax is always 1 and therefore results in 0 gradients)
        top_class= tf.reduce_sum(y*yt)
        
    #get gradients
    grads = tape.gradient(top_class, x)
    return grads

    
def get_integrated_gradients(x, yt, model, baseline=None, num_steps=100):
    """Computes Integrated Gradients for a label (usually correct class)
        based on https://keras.io/examples/vision/integrated_gradients/
    Args:
        x: Original input
        yt: one hot vector for x (=y*)
        baseline: baseline for interpolation 
        num_steps: Number of interpolation steps between the baseline
            and the input
    Returns:
        Integrated gradients w.r.t input image
    """

    #convert baseline or create a zero baseline, if none was given
    if baseline == None:
        baseline = tf.zeros(x.shape[-1],tf.float32)
    else:
        baseline = tf.cast(baseline, tf.float32)

    x = tf.cast(x, tf.float32)
    
    #Interpolate points over steps
    interpolated_x = [
        baseline + (step / num_steps) * (x - baseline)
        for step in range(num_steps + 1)
    ]
    
    interpolated_x = np.array(interpolated_x)


    #Get the gradients
    grads = []
    for x_s in interpolated_x:
        grad = get_gradient(x_s, yt, model)
        grads.append(grad)
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    #Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    #Calculate integrated gradients
    integrated_grads = (x - baseline) * avg_grads
    return integrated_grads


def get_jacobian(x, model):
    """Computes gradients for all classes

    Args:
        x: input tensor
    Returns:
        Gradients of predictions w.r.t input
    """
    x = tf.cast(x, tf.float32)

    #gradient tape for model
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
        
    #get jacobians to recieve gradients of all classes
    grads = tape.batch_jacobian(y,x)
    return grads

def get_integrated_jacobians(x, model, baseline=None, num_steps=100):
    """Computes Integrated Gradients for all classes
        based on https://keras.io/examples/vision/integrated_gradients/

    Args:
        x: Original input
        baseline: baseline for interpolation 
        num_steps: Number of interpolation steps between the baseline
            and the input

    Returns:
        Integrated gradients w.r.t input image
    """
    #convert baseline or create a zero baseline, if none was given
    if baseline == None:
        baseline = tf.zeros(x.shape[-1],tf.float32)
    else:
        baseline = tf.cast(baseline, tf.float32)

    x = tf.cast(x, tf.float32)
    
    #Interpolate points over steps
    interpolated_x = [
        baseline + (step / num_steps) * (x - baseline)
        for step in range(num_steps + 1)
    ]
    
    interpolated_x = np.array(interpolated_x)


    #Get the gradients
    grads = []
    for x in interpolated_x:
        #jacobian to recieve gradients for all classes
        grad = get_jacobian(x, model)
        grads.append(grad)
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    #Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    #Calculate integrated gradients
    integrated_grads = (x - baseline)[:,None] * avg_grads
    return integrated_grads

def get_shapley(x, model, baseline=None):
    """Computes Shapley Values for all classes

    Args:
        x: Original input
        baseline: baseline for replacement

    Returns:
        Shapley values
    """
    #amount of labels
    m = model.output_shape[-1]
    #amount of features
    n = x.shape[-1]
    #amount of datapoints (batch_size), using tf.shape(x), because batch_size is None during initialization,
    #which breakes tf.reshape below, tf.shape gives a scalar tensor with the variable batch_size (source: https://github.com/tensorflow/tensorflow/issues/7253)
    N = tf.shape(x)[0]
    
    #convert baseline or create a zero baseline, if none was given
    if baseline == None:
        baseline = tf.zeros(n)
    else:
        baseline = tf.cast(baseline, tf.float32)
    
    shapley = 0
    #set off all possible subsets
    possibleSubsets = np.array(list(itertools.product("01",repeat=n)), dtype="float32")
    

    #similar to identity matrix but has 1 and 0 swapped
    #used to remove each feature for the respective shapley computations
    zeroId = np.ones((n,n))-np.eye(n)

    #for shapley usually we iterate over (S \ x_i) \in Subsets, and then add x_i
    #our method instead iterates over all Subsets and omits x_i
    #for subsets where x_i is already omitted we have S == S\x_i and therefore f(S) - f(S\x_i) = f(S) - f(S) = 0
    for S in possibleSubsets:
        #Sox_i is S omitting x_i by multiplying these values with 0
        Sox_i = S*zeroId
        #subsetsize = amount of 1s in subsets without x_i
        subsetsizes = np.sum(Sox_i, axis=-1)
        #Subset applied on x, baseline added where values are omitted
        x_S = S*x+(1-S)*baseline
        #x with features on diag and features not in subset masked, baseline added where they were masked
        x_mask = tf.reshape(Sox_i*x[:,None]+(1-Sox_i)*baseline,(N*n,n))
        
        #fac = (subsetsize!*(n-subsetsize-1)!)/n!
        fac = (factorial(subsetsizes)*factorial(n-subsetsizes-1))/factorial(n)
        fac = tf.cast(tf.tile(fac, [N]),tf.float32)
        
        #addign shapley value for current subset fac*(f(S) - f(S\x_i))
        #reshaping such that we have a (n x m) shape for each datapoint
        shapley += tf.reshape(fac[:,None]*(tf.repeat(model(x_S),n,axis=0)-model(x_mask)),(N,n,m))
    return shapley



def get_sampling_shapley(x, model, baseline=None, runs=2, q_splits=100):
    """Computes sampled Shapley Values for all classes

    Args:
        x: Original input
        baseline: baseline for replacement
        runs: refine expectency parameter E
        q_splits = steps to approximate integral

    Returns:
        sampled Shapley values
    """
    #amount of labels
    m = model.output_shape[-1]
    #amount of features
    n = x.shape[-1]
    #amount of datapoints
    N = tf.shape(x)[0]
    
    #convert baseline or create a zero baseline, if none was given
    if baseline == None:
        baseline = tf.zeros(n, tf.float32)
    else:
        baseline = tf.cast(baseline, tf.float32)
        
        
    shapley = 0.

        
    #set of subsets, randomly generated
    S = []
    for _ in range(runs):
        for q_num in range(q_splits + 1):
            q = q_num / q_splits
            S.append(np.random.binomial(1, q, n))
    S = tf.cast(tf.stack(S), tf.float32)
    
    #we iterate over features
    for i in range(n):
        #onehot vector with feature i as 1
        i_one = tf.one_hot(i,n)
        #omitting feature j from subsets Sox_i = (S\x_i)
        Sox_i = S*(1-i_one)
        #including feature j 
        Sux_i = Sox_i + i_one
        #x with omitted features replaced by baseline
        x_mask = Sox_i*x[:,None] + (1-Sox_i) * baseline
        #x where x_i is included
        x_S = Sux_i*x[:,None] + (1-Sux_i)*baseline
        
        #flatten mask such that we have only 2 dimensions with first being subsets and batches, and second one being the values
        #tensorflow otherwise promts warnings for nested datapoints
        x_mask = tf.reshape(x_mask, ((q_splits+1)*runs*N,n))
        x_S = tf.reshape(x_S, ((q_splits+1)*runs*N,n))
        
        #(f(S) - f(S\x_i))
        diff = tf.reshape(model(x_S) - model(x_mask), (N, (q_splits+1)*runs,m))
        #add only for correct feature
        shapley += i_one[:,None]*tf.math.reduce_mean(diff, axis=-2)[:,None]
        
    return shapley

