import numpy as np


def ask_cstll(M):
    # returns an M-QAM constellation as an
    # array of dimension Mx1
    
    X = np.arange(-M+1,M,2,dtype=int)
    
    X = X.reshape((1,X.size))
    X = X/np.sqrt(np.mean(X**2))
    
    return X

def qam_cstll(M):
    # returns an M-QAM constellation as an
    # array of dimension Mx1
    
    M_ask = int(np.sqrt(M))
    tmp = ask_cstll(M_ask).reshape((M_ask,1))
   
    tmp = tmp + 1j*tmp.T
    X = tmp.flatten()
    X = X/np.sqrt(np.mean(np.abs(X)**2))
    
    return X

def mimo_qam_cstll(M, Nt):
    # returns a MIMO constellation for Nt antennas
    # and using a M-QAM constellation on all of them
    
    m = int(np.log2(M))
    X = qam_cstll(M)
    Xidx = np.arange(0,M,dtype=int)
    
    labels = qam_labels(M)
    labels_mimo = np.zeros((M**Nt,m*Nt), dtype=int)
    pX_mimo = np.zeros((1,M**Nt), dtype=float) / (M**Nt)
    
    arr = np.empty([M for i in range(Nt)] + [Nt], dtype=complex)
    arr1 = np.empty([M for i in range(Nt)] + [Nt], dtype=int)
    for i, a in enumerate(np.ix_(*[X for ii in np.arange(Nt)])):
        arr[...,i] = a
    for i, a in enumerate(np.ix_(*[Xidx for ii in np.arange(Nt)])):
        arr1[...,i] = a 
    X_mimo = arr.reshape(-1, Nt).T
    X_idx_mimo = arr1.reshape(-1, Nt).T
    for k in np.arange(0,M**Nt):
        labels_mimo[k,:] = np.concatenate(labels[X_idx_mimo[:,k],:],0)
    
    X_mimo = X_mimo / np.sqrt(np.mean(np.linalg.norm(X_mimo,axis=0)**2))
    return (X_mimo, pX_mimo, labels_mimo)

def ask_labels(M):
    # returns the Gray labels of an M-ASK constellation
    # as an array of dimensions M x log2(M)
    m = int(np.log2(M))
    labels = np.zeros((M,m), dtype=int)
    for k in range(1,m+1):
        tmp = np.floor(np.mod(np.arange(0,M)/(2**k)+0.5,2))
        labels[:,-k] = tmp

    return labels

def qam_labels(M):
    # returns te Gray labels of an M-QAM constellation
    # as an array of dimensions M x log2(M)
    
    # get the number of bits for a single 1D (ASK) constellation
    M_ask = int(np.sqrt(M))
    m_ask = int(np.log2(M_ask))
    m = int(m_ask*2)

    
    labels_ask = ask_labels(M_ask)
    labels_qam = np.empty((M,m),dtype=int)
    
    k = 0
    for i in range(0,M_ask):
        for j in range(0,M_ask):
            labels_qam[k,:] = np.hstack([labels_ask[i,:], labels_ask[j,:]])
            k = k + 1
            
    return labels_qam

def bicm_cap_mc(B, L):
    """
    Calculates the BICM capacity of an AWGN channel with input B and loglikelihood
    ratios L via Monte Carlo estimation. Both B and L have dimension #samples x m,
    where m denotes the number of bits of each constellation point.
    """
    m = L.shape[1]
    
    R = m - np.sum(np.mean(np.log2(1+np.exp(-L * (1-2*B))),axis=0))
    
    return R
