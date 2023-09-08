def OCBP(dataframe, *, y = None, peeled_to_obs = 4, h = None, standardize = False, kernel = "rbf"):
    """
    Implementation of OCBP method
    """
    from sklearn import svm 
    import numpy as np
    import matplotlib.pyplot as plt
    
    ## Standardize Data (if standardize == True)
    from sklearn.preprocessing import StandardScaler
    if standardize == True:
        scaler = StandardScaler()
        dataframe = scaler.fit_transform(dataframe) 
    
    ## Extract info from dataframe 
    N = dataframe.shape[0]
    m = dataframe.shape[0]
    p = dataframe.shape[1]
    invsigma = 1/(p+1)

    bagdat = dataframe.copy()
    ocsvm = svm.OneClassSVM(kernel = kernel, gamma = invsigma, nu = 0.01)
    ocsvm.fit(bagdat)
    KD_OCP = ocsvm.decision_function(dataframe)*-1
    KD_OCP = KD_OCP.reshape(N,1)
    
    ## Main while loop
    while m > peeled_to_obs:
            mu_OCP = np.mean(bagdat, axis = 0)
            w = ocsvm.fit(bagdat)
            bagdat = np.delete(bagdat, tuple(w.support_), axis = 0)
            m = bagdat.shape[0]
            # concatenation of distances
            dec_func_dist = ocsvm.decision_function(dataframe)*-1
            #dec_func_dist[dec_func_dist < 0] = 0
            KD_OCP = np.concatenate([KD_OCP, dec_func_dist.reshape(-1,1)], axis=1)  

    RKD_OCP = np.mean(KD_OCP, axis = 1)

    
    # Threshold. 
    if h == None:
        from scipy.stats import iqr
        M = medcouple(RKD_OCP)
        h  = np.quantile(RKD_OCP, 0.75) + 1.5*iqr(RKD_OCP)       
    
    OCP_Flag =  np.array([int(RKD_OCP[i] < h)*2 - 1 for i in range(N)])
    num_peels = KD_OCP.shape[1]
 
    return RKD_OCP, OCP_Flag, h, num_peels