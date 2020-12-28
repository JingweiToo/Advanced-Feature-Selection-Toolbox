#[2020]-"Improved Salp Swarm Algorithm based on opposition based learning and novel local search algorithm for feature selection"

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


def opposition_based_learning(X, lb, ub, thres, N, dim):
    Xo = np.zeros([N, dim], dtype='float')
    # opposition based learning (7)
    for i in range(N):
        for d in range(dim):
            Xo[i,d] = lb[0,d] + ub[0,d] - X[i,d]
                
    return Xo


def jfs(xtrain, ytrain, opts):
    # Parameters
    ub             = 1
    lb             = 0
    thres          = 0.5
    max_local_iter = 10
    
    N              = opts['N']
    max_iter       = opts['T']
    if 'maxLt' in opts:
        max_local_iter = opts['maxLt']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='int')
        lb = lb * np.ones([1, dim], dtype='int')
        
    # Initialize position & velocity
    X     = init_position(lb, ub, N, dim)
    
    # Pre
    fit  = np.zeros([N, 1], dtype='float')
    Xf   = np.zeros([1, dim], dtype='float')
    fitF = float('inf')
    
    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)
        
    # Fitness
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitF:
            Xf[0,:] = X[i,:]
            fitF    = fit[i,0]

    #--- Opposition based learning
    Xo    = opposition_based_learning(X, lb, ub, thres, N, dim) 
    #--- Binary conversion
    Xobin = binary_conversion(Xo, thres, N, dim)
    
    #--- Fitness
    fitO  = np.zeros([N, 1], dtype='float')
    for i in range(N):
        fitO[i,0] = Fun(xtrain, ytrain, Xobin[i,:], opts)
        if fitO[i,0] < fitF:
            Xf[0,:] = Xo[i,:]
            fitF    = fitO[i,0]
    
    #--- Merge opposite & current population, and select best N
    XX  = np.concatenate((X, Xo), axis=0)
    FF  = np.concatenate((fit, fitO), axis=0)
    #--- Sort in ascending order
    ind = np.argsort(FF, axis=0)
    for i in range(N):
        X[i,:]   = XX[ind[i,0],:]
        fit[i,0] = FF[ind[i,0]]
           
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    # Store result
    curve[0,t] = fitF
    print("Iteration:", t + 1)
    print("Best (ISSA):", curve[0,t])
    t += 1

    while t < max_iter:
 	    # Compute coefficient, c1 (2)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)

        for i in range(N):
            # First leader update
            if i == 0:  
                for d in range(dim):                
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
                    # Salp update by following front salp (3)
                    X[i,d] = (X[i,d] + X[i-1, d]) / 2              
                    # Boundary
                    X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d]) 
                
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitF:
                Xf[0,:] = X[i,:]
                fitF    = fit[i,0]
        
        #--- Local search algorithm
        Lt        = 0
        temp      = np.zeros([1, dim], dtype='float')
        temp[0,:] = Xf[0,:]       
        
        while Lt < max_local_iter:
            # Random three features
            RD = np.random.permutation(dim)
            for d in range(3):
                index = RD[d]                
                # Flip the selected three features
                if temp[0,index] > thres:
                    temp[0,index] = temp[0,index] - thres
                else:
                    temp[0,index] = temp[0,index] + thres
                           
            # Binary conversion
            temp_bin = binary_conversion(temp, thres, 1, dim)

            # Fitness
            Fnew = Fun(xtrain, ytrain, temp_bin[0,:], opts)
            if Fnew < fitF:
                fitF    = Fnew 
                Xf[0,:] = temp[0,:]
            
            Lt += 1


        # Store result
        curve[0,t] = fitF
        print("Iteration:", t + 1)
        print("Best (ISSA):", curve[0,t])
        t += 1


    # Best feature subset
    Gbin       = binary_conversion(Xf, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    issa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return issa_data  
