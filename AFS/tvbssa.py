#[2020]-"Time-varying hierarchical chains of salps with random weight networks for feature selection"

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


#--- transfer function (10)
def transfer_function(x):
    Tx = 1 / (1 + np.exp(-x))
    
    return Tx


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
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)
    
    #--- Binary conversion
    X     = binary_conversion(X, thres, N, dim)    
    
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xf    = np.zeros([1, dim], dtype='int')
    fitF  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    while t < max_iter:
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, X[i,:], opts)
            if fit[i,0] < fitF:
                Xf[0,:] = X[i,:]
                fitF    = fit[i,0]
        
        # Store result
        curve[0,t] = fitF.copy()
        print("Iteration:", t + 1)
        print("Best (TVBSSA):", curve[0,t])
        t += 1
        
 	    # Compute coefficient, c1 (3)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        
        #--- update number of leaders (12)
        L  = np.ceil(N * (t / (max_iter + 1))) 
        
        for i in range(N):          
            #--- leader update
            if i < L:  
                for d in range(dim):
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand() 
                    c3 = rand()
              	    # Leader update (2)
                    if c3 >= 0.5: 
                        Xn = Xf[0,d] + c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                    else:
                        Xn = Xf[0,d] - c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                
                    #--- transfer function
                    Tx = transfer_function(Xn)
                    #--- binary update (11)
                    if rand() < Tx:
                        X[i,d] = 0
                    else:
                        X[i,d] = 1
                
            #--- Salp update
            else:
                for d in range(dim):
                    # Salp update by following front salp (4)
                    Xn     = (X[i,d] + X[i-1, d]) / 2
                    # Boundary
                    Xn     = boundary(Xn, lb[0,d], ub[0,d]) 
                    #--- Binary conversion
                    if Xn > thres:
                        X[i,d] = 1
                    else:
                        X[i,d] = 0


    # Best feature subset
    Gbin       = Xf[0,:] 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    tvbssa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return tvbssa_data  

