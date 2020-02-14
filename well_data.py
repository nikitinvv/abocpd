

import numpy as np
import random
from scipy.special import psi, gammaln

def logistic_h(v, theta_h):

    h = theta_h[0]
    a = theta_h[1]
    b = theta_h[2]

    
    lp = 1 / (1 + np.exp(-(a * v + b)))
    lm = 1 / (1 + np.exp(a * v + b))
    H = h * lp

    dH = np.zeros(3)
    dH[0] = lp
    lp_lm_v =  (lp * lm * v)
    dH[1] = h * lp_lm_v
    dH[2] = h * lp * lm
    return  H, dH     


def studentpdf(x, mu, var, nu):

    c = np.exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2)) * pow((nu * np.pi * var), -0.5)
    p = c * pow((1 + (1 / (nu * var)) * (x - mu) ** 2),(-(nu + 1) / 2))

    dp = np.zeros(3)

    error = (x - mu) / np.sqrt(var)
    sq_error = (x - mu) ** 2 / var

    dlogp = (1 / np.sqrt(var)) * ((nu + 1) * error) / (nu + sq_error)
    dp[0] = p * dlogp

    dlogpdprec = np.sqrt(var) - ((nu + 1) * (x - mu) * error) / (nu + sq_error)
    dp[1] = - .5 * (p / pow(var, 1.5)) * dlogpdprec

    dlogp = psi(nu / 2 + .5) - psi(nu / 2) - (1 / nu) - np.log(1 + (1 / nu) * sq_error) + ((nu + 1) * sq_error) / (nu ** 2 + nu * sq_error)
    dp[2] = .5 * p * dlogp
    return p,dp

def gaussian1d_init(T, D, theta_m):
    post_params = theta_m
    dmessage = np.eye(4) 
    return post_params, dmessage


def gaussian1d_predict(post_params, xnew, dpost_params):
    N = len(post_params)    
    mus = post_params[0]
    kappas = post_params[1]
    alphas = post_params[2]
    betas = post_params[3]

    predictive_variance = betas * (kappas + 1) / (alphas * kappas)
    df = 2 * alphas
    
    pred, dtpdf = studentpdf(xnew, mus, predictive_variance, df)
    dmu_dtheta = dpost_params[0]#[[2,1,0,3],0]
    dkappa_dtheta = dpost_params[1]#[[2,1,0,3],1]
    dalpha_dtheta = dpost_params[2]#[[2,1,0,3],2]
    dbeta_dtheta = dpost_params[3]#[[2,1,0,3],3]
    
    dnu_dtheta = 2 * dalpha_dtheta

    dpv_dtheta = np.zeros(4)
    for ii in range(4):
        QRpart = (dbeta_dtheta[ii] * alphas - betas * dalpha_dtheta[ii]) / alphas ** 2
        dpv_dtheta[ii] = -(betas / (alphas * kappas ** 2)) * dkappa_dtheta[ii] + (1 + 1 / kappas) * QRpart    

    dp_dtheta = np.zeros(4)
    for ii in range(4):
        dp_dtheta[ii] = dtpdf[0] * dmu_dtheta[ii] + dtpdf[1] * dpv_dtheta[ii] + dtpdf[2] * dnu_dtheta[ii]
    dpred = dp_dtheta
    
    return pred, dpred


def bocpd_deriv(theta_h, theta_m, X):

    num_hazard_params = len(theta_h)
    num_model_params = len(theta_m)
    
    T = len(X)    
    D = 1
    R = np.zeros([T + 1, T + 1])
    dR_h = np.zeros([T + 1, T + 1, num_hazard_params])
    dR_m = np.zeros([T + 1, T + 1, num_model_params])
    R[0,0] = 1

    Z = np.zeros(T)
    dZ_h = np.zeros([T, num_hazard_params])
    dZ_m = np.zeros([T, num_model_params])
    
    post_params, dmessage = gaussian1d_init(T + 1, D, theta_m)
    
    for t in range(0,T):
        predprobs, dpredprobs = gaussian1d_predict(post_params, X[t], dmessage)
        H, dH = logistic_h(np.arange(t+1)+1, theta_h)        
        R[1:t + 1, t + 1] = R[:t, t] * predprobs * (1 - H)
        print( dH)
        for ii in range(num_hazard_params):
            dR_h[1:t + 2, t + 1, ii] = predprobs * (dR_h[:t+1, t, ii]  * (1 - H) - R[:t+1, t] * dH[ii])
            
        print(np.linalg.norm(dR_h))




#######################################continue here











def rt_minimize(X, f, length, *varargin):
    

    INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0                  # extrapolate maximum 3 times the current step-size
    MAX = 20                         # max 20 function evaluations per line search
    RATIO = 10                                       # maximum allowed slope ratio
    SIG = 0.1 
    RHO = SIG/2 # SIG and RHO are the constants controlling the Wolfe-

    if  isinstance(length, (list, tuple, np.ndarray)):
        red = length[1]
        length = length[0]
    else:
        red = 1

    if length>0:
        S='Linesearch'
    else:
        S='Function evaluation'
    i = 0;                             
    ls_failed = 0;        
    f0,df0 = f(X, *varargin) 

    fX = f0
    i = i + (length<0)                  
    s = -df0; d0 = -np.sum(s*s)#s'*s;           
    x3 = red/(1-d0)
    


def bocpd_dwrap1D(theta, X, conversion, num_hazard_params):

    # Warning: this code assumes: theta_h are in logit scale, theta_m(1) is in
    # linear, and theta_m(2:end) are in log scale!
    theta[conversion == 2] = 1 / (1 + np.exp(-theta[conversion == 2]))
    theta[conversion == 1] = np.exp(theta[conversion == 1])

    # Seperate theta into hazard and model hypers
    theta_h = theta[:num_hazard_params]
    theta_m = theta[num_hazard_params:]

    [nlml, dnlml_h, dnlml_m] = bocpd_deriv(theta_h, theta_m, X)

    # Put back into one vector for minimize
    dnlml = np.concatenate((dnlml_h, dnlml_m))

    # Correct for the distortion
    dnlml[conversion == 2] = dnlml[conversion == 2] * theta[conversion == 2] * (1 - theta[conversion == 2])
    dnlml[conversion == 1] = dnlml[conversion == 1] * theta[conversion == 1]
    return nlml, dnlml

def learn_bocpd(X, useLogistic):

    max_minimize_iter = 30

    num_hazard_params = 3    
    hazard_init = np.array([np.log(.01/(1-.01)),0,0])
    model_init = np.array([0,np.log(.1),np.log(.1),np.log(.1)])
    conversion = np.array([2,0,0,0,1,1,1])

    theta = np.concatenate((hazard_init,model_init))
    rt_minimize(theta, bocpd_dwrap1D, -max_minimize_iter, X, conversion, num_hazard_params)

    # hazard_params = theta(1:num_hazard_params)
    # model_params = theta((num_hazard_params + 1):end)

    # hazard_params(1) = logistic(hazard_params(1))
    # model_params(2:end) = exp(model_params(2:end))

    # function [nlml, dnlml] = ...
    # bocpd_dwrap1D(theta, X, gaussian1D, logistic_h, conversion, num_hazard_params)

    # % Warning: this code assumes: theta_h are in logit scale, theta_m(1) is in
    # % linear, and theta_m(2:end) are in log scale!

    # theta[conversion == 2] = logistic(theta[conversion == 2])
    # theta[conversion == 1] = exp(theta[conversion == 1])

    # % Seperate theta into hazard and model hypers
    # theta_h = theta(1:num_hazard_params)
    # theta_m = theta(num_hazard_params+1:end)

    # [nlml, dnlml_h, dnlml_m] = ...
    # bocpd_deriv(theta_h', theta_m', X, logistic_h, gaussian1D)

    # % Put back into one vector for minimize
    # dnlml = [dnlml_h dnlml_m]

    # % Correct for the distortion
    # dnlml[conversion == 2] = dnlml[conversion == 2] .* theta[conversion == 2] .* (1 - theta[conversion == 2])
    # dnlml[conversion == 1] = dnlml[conversion == 1] .* theta[conversion == 1]

#return [hazard_params, model_params, nlml]

if __name__ == '__main__':
    print('Trying well log data')

    # I think all this code is deterministic, but let's fix the seed to be safe.
    random.seed(4)

    well = np.genfromtxt('data/well.dat')
    
    # We don't know the physical interpretation, so lets just standardize the
    # readings to make them cleaner.
    X = (well - np.mean(well,axis=0))/(1e-16 + np.std(well,axis=0))        
    Tlearn = 2000
    Ttest = X.shape[0] - Tlearn
    useLogistic = True

    # TODO compare logistic h and constant h
    # Can try learn_IFM usinf IFMfast to speed this up
    print('Starting learning')
    learn_bocpd(X[:Tlearn], useLogistic)    

    print('Learning Done')
    