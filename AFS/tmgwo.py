#[2020]-"A new fusion of grey wolf optimizer algorithm with a two-phase mutation for feature selection"

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
 

#--- transfer function update binary position (4.3.2)
def transfer_function(x):
    Xs = abs(np.tanh(x)) 

    return Xs


def jfs(xtrain, ytrain, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    Mp    = 0.5    # mutation probability
    
    N        = opts['N']
    max_iter = opts['T']
    if 'Mp' in opts:
        Mp   = opts['Mp']   
        
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X  = init_position(lb, ub, N, dim)
    
    #--- Binary conversion
    X  = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='int')
    Xbeta  = np.zeros([1, dim], dtype='int')
    Xdelta = np.zeros([1, dim], dtype='int')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, X[i,:], opts)
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = Falpha.copy()
    print("Iteration:", t + 1)
    print("Best (TMGWO):", curve[0,t])
    t += 1
    
    while t < max_iter:  
      	# Coefficient decreases linearly from 2 to 0 (3.5)
        a = 2 - t * (2 / max_iter) 
        
        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.7 - 3.9)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.7 -3.9) 
                X1 = Xalpha[0,d] - A1 * Dalpha
                X2 = Xbeta[0,d] - A2 * Dbeta
                X3 = Xdelta[0,d] - A3 * Ddelta
                # Update wolf (3.6)
                Xn = (X1 + X2 + X3) / 3                
                #--- transfer function
                Xs = transfer_function(Xn)
                #--- update position (4.3.2)
                if rand() < Xs:
                    X[i,d] = 0
                else:
                    X[i,d] = 1
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, X[i,:], opts)
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha      = fit[i,0]
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fit[i,0]
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]
        
        curve[0,t] = Falpha.copy()
        print("Iteration:", t + 1)
        print("Best (TMGWO):", curve[0,t])
        t += 1
        
        #--- two phase mutation: first phase
        # find index of 1
        idx        = np.where(Xalpha == 1)
        idx1       = idx[1]
        Xmut1      = np.zeros([1, dim], dtype='int')
        Xmut1[0,:] = Xalpha[0,:]
        for d in range(len(idx1)):
            r = rand()
            if r < Mp:
                Xmut1[0, idx1[d]] = 0
                Fnew1 = Fun(xtrain, ytrain, Xmut1[0,:], opts)
                if Fnew1 < Falpha:
                    Falpha      = Fnew1
                    Xalpha[0,:] = Xmut1[0,:]
                    
        #--- two phase mutation: second phase        
        # find index of 0
        idx        = np.where(Xalpha == 0)
        idx0       = idx[1]
        Xmut2      = np.zeros([1, dim], dtype='int')
        Xmut2[0,:] = Xalpha[0,:]    
        for d in range(len(idx0)):
            r = rand()
            if r < Mp:
                Xmut2[0, idx0[d]] = 1
                Fnew2 = Fun(xtrain, ytrain, Xmut2[0,:], opts)
                if Fnew2 < Falpha:
                    Falpha      = Fnew2
                    Xalpha[0,:] = Xmut2[0,:]
                
        
    # Best feature subset
    Gbin       = Xalpha[0,:]
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    tmgwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return tmgwo_data    
                
                
                
    
