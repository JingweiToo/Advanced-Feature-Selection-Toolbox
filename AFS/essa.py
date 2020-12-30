#[2019]-"Enhanced salp swarm algorithm: Application to variable speed wind generators"

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
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position & velocity
    X     = init_position(lb, ub, N, dim)
    
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xf    = np.zeros([1, dim], dtype='float')
    fitF  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitF:
                Xf[0,:] = X[i,:]
                fitF    = fit[i,0]
        
        # Store result
        curve[0,t] = fitF.copy()
        print("Iteration:", t + 1)
        print("Best (ESSA):", curve[0,t])
        t += 1
        
        for i in range(N):          
            # First leader update
            if i == 0: 
                for d in range(dim):
                    #--- random number between [0, 50]
                    r1 = 50 * rand()
                    #--- compute c1 (5)
                    c1 = 2 * np.exp(-((4 * r1 * t) / max_iter) ** 2)
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand() 
                    c3 = rand()
              	    # Leader update (1)
                    if c3 >= 0.5: 
                        X[i,d] = Xf[0,d] + c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                    else:
                        X[i,d] = Xf[0,d] - c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                
                    # Boundary
                    X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d]) 
                
            # Salp update
            elif i >= 1:
                for d in range(dim):
                    #--- random number between [0, 1]
                    r2     = rand()
                    #--- Salp update by following front salp (6)
                    X[i,d] = r2 * (X[i,d] + X[i-1, d])
                    # Boundary
                    X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d]) 
        

    # Best feature subset
    Gbin       = binary_conversion(Xf, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    essa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return essa_data  