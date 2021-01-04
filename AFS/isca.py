#[2017]-"Sine–cosine algorithm for feature selection with elitism strategy and new updating mechanism"
# Note: This algorithm only consider one fitness function

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x


def jfs(xtrain, ytrain, opts):
    # Parameters
    ub        = 1
    lb        = 0
    thres     = 0.5
    alpha     = 2       # constant
    num_elite = 2       # number of elites
    
    N             = opts['N']
    max_iter      = opts['T']
    if 'alpha' in opts:
        alpha     = opts['alpha'] 
    if 'Ne' in opts:
        num_elite = opts['Ne'] 
        
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)

    #--- Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)

    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xdb   = np.zeros([1, dim], dtype='float')
    fitD  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
        
    #--- Fitness
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitD:
            Xdb[0,:] = X[i,:]
            fitD     = fit[i,0]
                
    #--- Store result
    curve[0,t] = fitD.copy()
    print("Iteration:", t + 1)
    print("Best (ISCA):", curve[0,t])
    t += 1

    while t < max_iter:
        #--- implement elitism
        Xe  = np.zeros([num_elite, dim], dtype='float')
        ind = np.argsort(fit, axis=0)
        for i in range(num_elite):
            Xe[i,:] = X[ind[i,0],:].copy()
        
        # Parameter r1, decreases linearly from alpha to 0 
        r1 = alpha - t * (alpha / max_iter)
        
        for i in range(N):
            for d in range(dim):
                # Random parameter r2 & r3 & r4
                r2 = (2 * np.pi) * rand()
                r3 = 2 * rand()
                r4 = rand()
                # Position update (2)
                if r4 < 0.5:
                    # Sine update (2)
                    X[i,d] = X[i,d] + r1 * np.sin(r2) * abs(r3 * Xdb[0,d] - X[i,d]) 
                else:
                    # Cosine update (2)
                    X[i,d] = X[i,d] + r1 * np.cos(r2) * abs(r3 * Xdb[0,d] - X[i,d])
                
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d]) 

        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)
    
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitD:
                Xdb[0,:] = X[i,:]
                fitD     = fit[i,0]
                    
        # Store result
        curve[0,t] = fitD.copy()
        print("Iteration:", t + 1)
        print("Best (ISCA):", curve[0,t])
        t += 1
        
        #--- replace worst agents
        ind = np.argsort(-1 * fit, axis=0)
        for i in range(num_elite):
            X[ind[i,0],:] = Xe[i,:].copy()

    
    # Best feature subset
    Gbin       = binary_conversion(Xdb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    isca_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return isca_data   
        
                    
        
        
        