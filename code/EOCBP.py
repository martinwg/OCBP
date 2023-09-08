def EOCBP(dataframe, *, y = None, peeled_to_obs = 2, n_estimators = 50, standardize = False, kernel = "rbf"):
    """
    Implementation of EOCBP method
    """
    from sklearn import svm 
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import iqr
    
    ## Standardize Data (if standardize == True)
    if standardize == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        dataframe = scaler.fit_transform(dataframe) 
    
    ## Extract info from dataframe 
    N = len(dataframe)
    p = dataframe.shape[1]
    invsigma = 1/p
    ens_peel = np.zeros((N,n_estimators))
    ens_dist = np.zeros((N,n_estimators))
    ens_flags = np.zeros((N,n_estimators))
    
    for estimator in range(n_estimators):
        bagdat = dataframe.copy()
        max_features=int(np.sqrt(p))
        bagdat = bagdat[:,np.random.randint(0,p,max_features)]
        df = bagdat.copy()
        ocsvm = svm.OneClassSVM(nu=0.01, kernel = kernel, gamma = invsigma)
        KD_OCP = np.zeros(N).reshape(N,1)
        peel = 0
        peel_del = np.zeros(N)
        m = N
        ## Main while loop
        while m > peeled_to_obs:
                mu_OCP = np.mean(bagdat, axis = 0)
                w = ocsvm.fit(bagdat)
                bagdat = np.delete(bagdat, tuple(w.support_), axis = 0)
                m = bagdat.shape[0]
                peel = peel + 1
                peel_del[[i for i, x in enumerate(df) if x in w.support_vectors_]] = peel
                dec_func_dist = ocsvm.decision_function(df)*-1
                KD_OCP = np.concatenate([KD_OCP, dec_func_dist.reshape(-1,1)], axis=1)  
        peel_del[[i for i, x in enumerate(df) if x in bagdat]] = peel+1
        peel_del = np.max(peel_del) / peel_del
        ens_peel[:,estimator] = peel_del
        ens_dist[:,estimator] = np.mean(KD_OCP[:,1:], axis = 1)
        
    mean_dist = np.mean(ens_dist, axis = 1)*np.mean(ens_peel, axis = 1)
    h  = np.quantile(mean_dist, 0.75) + 1.5*iqr(mean_dist)
    OCP_Flag =  np.array([int(mean_dist[i] < h)*2 - 1 for i in range(N)])
    
    return mean_dist, OCP_Flag, h, np.mean(ens_peel, axis = 1)