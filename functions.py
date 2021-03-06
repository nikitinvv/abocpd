

# functions module

import numpy as np
import random
from scipy.special import psi, gammaln
from scipy.stats import norm, t
import matplotlib.pyplot as plt


def logistic_h(v, theta_h, nargout=1):

    h = theta_h[0]
    a = theta_h[1]
    b = theta_h[2]

    lp = 1 / (1 + np.exp(-(a * v + b)))
    lm = 1 / (1 + np.exp(a * v + b))
    H = h * lp
    if(nargout == 1):
        return H

    dH = np.zeros([len(lp), 3])
    dH[:, 0] = lp
    lp_lm_v = (lp * lm * v)
    dH[:, 1] = h * lp_lm_v
    dH[:, 2] = h * lp * lm
    return H, dH


def constant_h(v, theta_h, nargout=1):

    H = np.ones(v.shape)*theta_h
    if(nargout == 1):
        return H
    dH = np.ones(v.shape)
    return H, dH


def studentpdf(x, mu, var, nu, nargout=1):

    c = np.exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2)) * \
        pow((nu * np.pi * var), -0.5)
    p = c * pow((1 + (1 / (nu * var)) * (x - mu) ** 2), (-(nu + 1) / 2))
    if (nargout == 1):
        return p

    N = len(mu)
    dp = np.zeros([N, 3])

    error = (x - mu) / np.sqrt(var)
    sq_error = (x - mu) ** 2 / var

    dlogp = (1 / np.sqrt(var)) * ((nu + 1) * error) / (nu + sq_error)
    dp[:, 0] = p * dlogp

    dlogpdprec = np.sqrt(var) - ((nu + 1) * (x - mu) * error) / (nu + sq_error)
    dp[:, 1] = - .5 * (p / pow(var, 1.5)) * dlogpdprec

    dlogp = psi(nu / 2 + .5) - psi(nu / 2) - (1 / nu) - np.log(1 + (1 / nu)
                                                               * sq_error) + ((nu + 1) * sq_error) / (nu ** 2 + nu * sq_error)
    dp[:, 2] = .5 * p * dlogp
    return p, dp


def studentpdffast(x, mu, var, nu):
    global Gcache
    global Pcache

    D = len(x)

    c = Gcache[:nu.shape[0], :] * pow((nu * np.pi * var), -0.5)
    sq_error = (mu - x) ** 2
    p = c*pow((1 + (1 / (nu * var)) * sq_error), (-(nu + 1) / 2))

    N = nu.shape[0]
    dp = np.zeros([N, D, 3])

    error = -(mu-x) / np.sqrt(var)
    sq_error = (mu-x) ** 2 / var

    dlogp = (1 / np.sqrt(var)) * ((nu + 1) * error) / (nu + sq_error)
    dp[:, :, 0] = p * dlogp

    dlogpdprec = np.sqrt(var) - ((nu + 1) * -(mu-x) * error) / (nu + sq_error)
    dp[:, :, 1] = - .5 * (p / pow(var, 1.5)) * dlogpdprec

    dlogp = Pcache[:nu.shape[0], :] - np.log(1 + (1 / nu) * sq_error) + (
        (nu + 1) * sq_error) / (nu ** 2 + nu * sq_error)
    dp[:, :, 2] = .5 * p * dlogp
    return p, dp


def gaussian1D_init(T, D, theta_m, nargout=1):
    post_params = np.reshape(theta_m, [1, len(theta_m)])
    if nargout == 1:
        return post_params
    dmessage = np.reshape(np.eye(4), [4, 4, 1])
    return post_params, dmessage


def IFMfast_init(T, D, theta_m, nargout=1):
    post_params = np.reshape(theta_m, [1, len(theta_m)])
    dmessage = np.tile(np.reshape(np.eye(4), [4, 4, 1, 1]), [1, 1, T, D])
    dmessage = np.transpose(dmessage, (2, 0, 1, 3))
    setupGcache(theta_m, 5000, D)
    return post_params, dmessage


def IFM_init(T, D, theta_m, nargout=1):
    post_params = np.reshape(theta_m, [1, len(theta_m)])
    dmessage = np.tile(np.reshape(np.eye(4), [4, 4, 1, 1]), [1, 1, T, D])
    dmessage = np.transpose(dmessage, (2, 0, 1, 3))
    return post_params


def gaussian1D_predict(post_params, xnew, dpost_params=None, nargout=1):
    N = post_params.shape[0]
    mus = post_params[:, 0]
    kappas = post_params[:, 1]
    alphas = post_params[:, 2]
    betas = post_params[:, 3]

    predictive_variance = betas * (kappas + 1) / (alphas * kappas)
    df = 2 * alphas

    if (nargout == 1):
        pred = studentpdf(xnew, mus, predictive_variance, df)
        return pred
    pred, dtpdf = studentpdf(xnew, mus, predictive_variance, df, 2)
    dmu_dtheta = np.transpose(dpost_params[0], [1, 0])
    dkappa_dtheta = np.transpose(dpost_params[1], [1, 0])
    dalpha_dtheta = np.transpose(dpost_params[2], [1, 0])
    dbeta_dtheta = np.transpose(dpost_params[3], [1, 0])
    dnu_dtheta = 2 * dalpha_dtheta

    dpv_dtheta = np.zeros([N, 4])
    for ii in range(4):
        QRpart = (dbeta_dtheta[:, ii] * alphas -
                  betas * dalpha_dtheta[:, ii]) / alphas ** 2
        dpv_dtheta[:, ii] = -(betas / (alphas * kappas ** 2)) * \
            dkappa_dtheta[:, ii] + (1 + 1 / kappas) * QRpart

    dp_dtheta = np.zeros([N, 4])
    for ii in range(4):
        dp_dtheta[:, ii] = dtpdf[:, 0] * dmu_dtheta[:, ii] + dtpdf[:,
                                                                   1] * dpv_dtheta[:, ii] + dtpdf[:, 2] * dnu_dtheta[:, ii]
    dpred = dp_dtheta

    return pred, dpred


def IFMfast_predict(post_params, xnew, dpost_params, nargout=1):
    N = post_params.shape[0]
    D = len(xnew)
    mus = post_params[:, :D]
    kappas = post_params[:, D:2 * D]
    alphas = post_params[:, 2 * D:3 * D]
    betas = post_params[:, 3 * D:4 * D]
    predictive_variance = betas * (kappas + 1) / (alphas * kappas)
    df = 2 * alphas

    predIF, dtpdf = studentpdffast(xnew, mus, predictive_variance, df)
    pred = np.prod(predIF, 1)
    if (nargout == 1):
        return pred

    dp_dtheta = np.zeros([N, D, 4])

    for ii in range(D):
        dmu_dtheta = dpost_params[:N, :, 0, ii]
        dkappa_dtheta = dpost_params[:N, :, 1, ii]
        dalpha_dtheta = dpost_params[:N, :, 2, ii]
        dbeta_dtheta = dpost_params[:N, :, 3, ii]

        dnu_dtheta = 2 * dalpha_dtheta

        dpv_dtheta = np.zeros([N, 4])
        for jj in range(4):
            QRpart = (dbeta_dtheta[:, jj] * alphas[:, ii] -
                      betas[:, ii] * dalpha_dtheta[:, jj]) / alphas[:, ii]**2
            dpv_dtheta[:, jj] = -(betas[:, ii] / (alphas[:, ii] * kappas[:, ii] ** 2)) * \
                dkappa_dtheta[:, jj] + (1 + 1 / kappas[:, ii]) * QRpart
        for jj in range(4):
            dp_dtheta[:, ii, jj] = dtpdf[:, ii, 0] * dmu_dtheta[:, jj] + dtpdf[:,
                                                                               ii, 1] * dpv_dtheta[:, jj] + dtpdf[:, ii, 2] * dnu_dtheta[:, jj]
            dp_dtheta[:, ii, jj] = dp_dtheta[:,
                                             ii, jj] * (pred / predIF[:, ii])
    dpred = np.zeros([N, 4 * D])
    dpred[:, :D] = dp_dtheta[:, :, 0]
    dpred[:, D:2 * D] = dp_dtheta[:, :, 1]
    dpred[:, 2 * D:3 * D] = dp_dtheta[:, :, 2]
    dpred[:, 3 * D:4 * D] = dp_dtheta[:, :, 3]

    return pred, dpred


def IFM_predict(post_params, xnew, dpost_params=None, nargout=1):
    N = post_params.shape[0]
    D = len(xnew)
    mus = post_params[:, :D]
    kappas = post_params[:, D:2 * D]
    alphas = post_params[:, 2 * D:3 * D]
    betas = post_params[:, 3 * D:4 * D]
    predictive_variance = betas * (kappas + 1) / (alphas * kappas)
    df = 2 * alphas

    predIF = np.zeros([N, D])

    for ii in range(D):
        predIF[:, ii] = studentpdf(
            xnew[ii], mus[:, ii], predictive_variance[:, ii], df[:, ii], 1)

    pred = np.prod(predIF, 1)
    if (nargout == 1):
        return pred

    dp_dtheta = np.zeros([N, D, 4])

    for ii in range(D):

        tmp, dtpdf = studentpdf(
            xnew[ii], mus[:, ii], predictive_variance[:, ii], df[:, ii], 2)

        dmu_dtheta = dpost_params[:N, :, 0, ii]
        dkappa_dtheta = dpost_params[:N, :, 1, ii]
        dalpha_dtheta = dpost_params[:N, :, 2, ii]
        dbeta_dtheta = dpost_params[:N, :, 3, ii]

        dnu_dtheta = 2 * dalpha_dtheta

        dpv_dtheta = np.zeros([N, 4])
        for jj in range(4):
            QRpart = (dbeta_dtheta[:, jj] * alphas[:, ii] -
                      betas[:, ii] * dalpha_dtheta[:, jj]) / alphas[:, ii]**2
            dpv_dtheta[:, jj] = -(betas[:, ii] / (alphas[:, ii] * kappas[:, ii] ** 2)) * \
                dkappa_dtheta[:, jj] + (1 + 1 / kappas[:, ii]) * QRpart
        for jj in range(4):
            dp_dtheta[:, ii, jj] = dtpdf[:, 0] * dmu_dtheta[:, jj] + \
                dtpdf[:, 1] * dpv_dtheta[:, jj] + \
                dtpdf[:, 2] * dnu_dtheta[:, jj]
            dp_dtheta[:, ii, jj] = dp_dtheta[:,
                                             ii, jj] * (pred / predIF[:, ii])
    dpred = np.zeros([N, 4 * D])
    dpred[:, :D] = dp_dtheta[:, :, 0]
    dpred[:, D:2 * D] = dp_dtheta[:, :, 1]
    dpred[:, 2 * D:3 * D] = dp_dtheta[:, :, 2]
    dpred[:, 3 * D:4 * D] = dp_dtheta[:, :, 3]

    return pred, dpred


def gaussian1D_update(theta_prior, post_params, xt, dpost_params=None, nargout=1):

    mus = post_params[:, 0]
    kappas = post_params[:, 1]
    alphas = post_params[:, 2]
    betas = post_params[:, 3]

    mus_new = np.concatenate(
        ([theta_prior[0]], (kappas * mus + xt) / (kappas + 1)))
    kappas_new = np.concatenate(([theta_prior[1]], kappas + 1))
    alphas_new = np.concatenate(([theta_prior[2]], alphas + 0.5))
    betas_new = np.concatenate(
        ([theta_prior[3]], betas + (kappas * (xt - mus) ** 2) / (2 * (kappas + 1))))
    post_params = np.array(
        [mus_new, kappas_new, alphas_new, betas_new]).swapaxes(0, 1)
    if(nargout == 1):
        return post_params

    dmu_dmu0 = dpost_params[0, 0]
    dmu_dkappa0 = dpost_params[0, 1]
    dkappa_dkappa0 = dpost_params[1, 1]
    dbeta_dmu0 = dpost_params[3, 0]
    dbeta_dkappa0 = dpost_params[3, 1]

    dmu_dmu0_new = (kappas / (kappas + 1)) * dmu_dmu0
    dmu_dkappa0_new = (dkappa_dkappa0 * mus + dmu_dkappa0 * kappas) / \
        (kappas + 1) - ((kappas * mus + xt) * dkappa_dkappa0) / ((kappas + 1) ** 2)
    dbeta_dmu0_new = dbeta_dmu0 - \
        ((kappas * (xt - mus)) / (kappas + 1) * dmu_dmu0)

    den = 2 * (kappas + 1)
    num = kappas * (xt - mus) ** 2
    dden_dkappa0 = 2 * dkappa_dkappa0
    dnum_dkappa0 = dkappa_dkappa0 * \
        (xt - mus) ** 2 + 2 * kappas * (mus - xt) * dmu_dkappa0
    QR = (dnum_dkappa0 * den - dden_dkappa0 * num) / den ** 2
    dbeta_dkappa0_new = dbeta_dkappa0 + QR

    dpost_params[0, 0] = dmu_dmu0_new
    dpost_params[0, 1] = dmu_dkappa0_new
    dpost_params[3, 0] = dbeta_dmu0_new
    dpost_params[3, 1] = dbeta_dkappa0_new

    deye = np.expand_dims(np.eye(dpost_params.shape[0]), axis=2)
    dpost_params = np.concatenate((deye, dpost_params), axis=2)

    return post_params, dpost_params


def IFMfast_update(theta_prior, post_params, xt, dpost_params, maxRunConsidered, nargout=1):

    N = post_params.shape[0]
    D = len(xt)
    num_params = post_params.shape[1]

    mus = post_params[:, :D]
    kappas = post_params[:, D:2 * D]
    alphas = post_params[:, 2 * D:3 * D]
    betas = post_params[:, 3 * D:4 * D]

    mu_prior = theta_prior[:D]
    kappa_prior = theta_prior[D:2 * D]
    alpha_prior = theta_prior[2 * D: 3 * D]
    beta_prior = theta_prior[3 * D: 4 * D]

    mus_new = np.zeros([N + 1, D])
    kappas_new = np.zeros([N + 1, D])
    alphas_new = np.zeros([N + 1, D])
    betas_new = np.zeros([N + 1, D])
    for ii in range(D):
        mus_new[:, ii] = np.concatenate(
            ([mu_prior[ii]], (kappas[:, ii] * mus[:, ii] + xt[ii]) / (kappas[:, ii] + 1)))
        kappas_new[:, ii] = np.concatenate(
            ([kappa_prior[ii]], kappas[:, ii] + 1))
        alphas_new[:, ii] = np.concatenate(
            ([alpha_prior[ii]], alphas[:, ii] + 0.5))
        betas_new[:, ii] = np.concatenate(([beta_prior[ii]], betas[:, ii] + (
            kappas[:, ii] * (xt[ii] - mus[:, ii]) ** 2) / (2 * (kappas[:, ii] + 1))))
    post_params = np.concatenate(
        (mus_new, kappas_new, alphas_new, betas_new), axis=1)

    post_params = post_params[:maxRunConsidered, :]
    if nargout == 1:
        return

    for ii in range(D):
        dmu_dmu0 = dpost_params[:N, 0, 0, ii]
        dmu_dkappa0 = dpost_params[:N, 1, 0, ii]
        dkappa_dkappa0 = dpost_params[:N, 1, 1, ii]
        dbeta_dmu0 = dpost_params[:N, 0, 3, ii]
        dbeta_dkappa0 = dpost_params[:N, 1, 3, ii]

        dmu_dmu0_new = (kappas[:, ii] / (kappas[:, ii] + 1)) * dmu_dmu0
        dmu_dkappa0_new = (dkappa_dkappa0 * mus[:, ii] + dmu_dkappa0 * kappas[:, ii]) / (kappas[:, ii] + 1) - \
            ((kappas[:, ii] * mus[:, ii] + xt[ii]) *
             dkappa_dkappa0) / ((kappas[:, ii] + 1) ** 2)
        dbeta_dmu0_new = dbeta_dmu0 - \
            ((kappas[:, ii] * (xt[ii] - mus[:, ii])) /
             (kappas[:, ii] + 1) * dmu_dmu0)

        den = 2 * (kappas[:, ii] + 1)
        num = kappas[:, ii] * (xt[ii] - mus[:, ii]) ** 2
        dden_dkappa0 = 2 * dkappa_dkappa0
        dnum_dkappa0 = dkappa_dkappa0 * \
            (xt[ii] - mus[:, ii]) ** 2 + 2 * kappas[:, ii] * \
            (mus[:, ii] - xt[ii]) * dmu_dkappa0
        QR = (dnum_dkappa0 * den - dden_dkappa0 * num) / den ** 2
        dbeta_dkappa0_new = dbeta_dkappa0 + QR

        dpost_params0 = dpost_params
        dpost_params = np.zeros([N+1, *dpost_params.shape[1:]])
        dpost_params[:dpost_params0.shape[0]] = dpost_params0
        dpost_params[1:N + 1, 0, 0, ii] = dmu_dmu0_new
        dpost_params[1:N + 1, 1, 0, ii] = dmu_dkappa0_new
        dpost_params[1:N + 1, 0, 3, ii] = dbeta_dmu0_new
        dpost_params[1:N + 1, 1, 3, ii] = dbeta_dkappa0_new

        dpost_params[1:N + 1, 1, 1, ii] = np.ones(N)
        dpost_params[1:N + 1, 2, 2, ii] = np.ones(N)
        dpost_params[1:N + 1, 3, 3, ii] = np.ones(N)

    for ii in range(D):
        dpost_params[0, :, :, ii] = np.eye(dpost_params.shape[1])

    dpost_params = dpost_params[:maxRunConsidered, :, :, :]

    return post_params, dpost_params


def IFM_update(theta_prior, post_params, xt, dpost_params, maxRunConsidered, nargout=1):

    N = post_params.shape[0]
    D = len(xt)
    num_params = post_params.shape[1]

    mus = post_params[:, :D]
    kappas = post_params[:, D:2 * D]
    alphas = post_params[:, 2 * D:3 * D]
    betas = post_params[:, 3 * D:4 * D]

    mu_prior = theta_prior[:D]
    kappa_prior = theta_prior[D:2 * D]
    alpha_prior = theta_prior[2 * D: 3 * D]
    beta_prior = theta_prior[3 * D: 4 * D]

    mus_new = np.zeros([N + 1, D])
    kappas_new = np.zeros([N + 1, D])
    alphas_new = np.zeros([N + 1, D])
    betas_new = np.zeros([N + 1, D])

    for ii in range(D):

        mus_new[:, ii] = np.concatenate(
            ([mu_prior[ii]], (kappas[:, ii] * mus[:, ii] + xt[ii]) / (kappas[:, ii] + 1)))
        kappas_new[:, ii] = np.concatenate(
            ([kappa_prior[ii]], kappas[:, ii] + 1))
        alphas_new[:, ii] = np.concatenate(
            ([alpha_prior[ii]], alphas[:, ii] + 0.5))
        betas_new[:, ii] = np.concatenate(([beta_prior[ii]], betas[:, ii] + (
            kappas[:, ii] * (xt[ii] - mus[:, ii]) ** 2) / (2 * (kappas[:, ii] + 1))))
    post_params = np.concatenate(
        (mus_new, kappas_new, alphas_new, betas_new), axis=1)

    post_params = post_params[:maxRunConsidered, :]
    if nargout == 1:
        return post_params

    for ii in range(D):
        dmu_dmu0 = dpost_params[:N, 0, 0, ii]
        dmu_dkappa0 = dpost_params[:N, 1, 0, ii]
        dkappa_dkappa0 = dpost_params[:N, 1, 1, ii]
        dbeta_dmu0 = dpost_params[:N, 0, 3, ii]
        dbeta_dkappa0 = dpost_params[:N, 1, 3, ii]

        dmu_dmu0_new = (kappas[:, ii] / (kappas[:, ii] + 1)) * dmu_dmu0
        dmu_dkappa0_new = (dkappa_dkappa0 * mus[:, ii] + dmu_dkappa0 * kappas[:, ii]) / (kappas[:, ii] + 1) - \
            ((kappas[:, ii] * mus[:, ii] + xt[ii]) *
             dkappa_dkappa0) / ((kappas[:, ii] + 1) ** 2)
        dbeta_dmu0_new = dbeta_dmu0 - \
            ((kappas[:, ii] * (xt[ii] - mus[:, ii])) /
             (kappas[:, ii] + 1) * dmu_dmu0)

        den = 2 * (kappas[:, ii] + 1)
        num = kappas[:, ii] * (xt[ii] - mus[:, ii]) ** 2
        dden_dkappa0 = 2 * dkappa_dkappa0
        dnum_dkappa0 = dkappa_dkappa0 * \
            (xt[ii] - mus[:, ii]) ** 2 + 2 * kappas[:, ii] * \
            (mus[:, ii] - xt[ii]) * dmu_dkappa0
        QR = (dnum_dkappa0 * den - dden_dkappa0 * num) / den ** 2
        dbeta_dkappa0_new = dbeta_dkappa0 + QR

        dpost_params = np.zeros([N+1, *dpost_params.shape[1:]])
        dpost_params[1:N + 1, 0, 0, ii] = dmu_dmu0_new
        dpost_params[1:N + 1, 1, 0, ii] = dmu_dkappa0_new
        dpost_params[1:N + 1, 0, 3, ii] = dbeta_dmu0_new
        dpost_params[1:N + 1, 1, 3, ii] = dbeta_dkappa0_new

        dpost_params[1:N + 1, 1, 1, ii] = np.ones(N)
        dpost_params[1:N + 1, 2, 2, ii] = np.ones(N)
        dpost_params[1:N + 1, 3, 3, ii] = np.ones(N)

    for ii in range(D):
        dpost_params[0, :, :, ii] = np.eye(dpost_params.shape[1])

    dpost_params = dpost_params[:maxRunConsidered, :, :, :]

    return post_params, dpost_params


def rt_minimize(X, f, length, *varargin):

    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG/2

    red = 1
    S = 'Function evaluation'

    i = 0
    ls_failed = 0
    f0, df0 = f(X, *varargin)    
    fX = [f0]
    i = i + (length < 0)
    s = -df0
    d0 = -np.sum(s*s)
    x3 = red/(1-d0)
    
    
    while i < abs(length):
        i = i + (length > 0)
        X0 = X
        F0 = f0
        dF0 = df0
        M = min(MAX, -length-i)
        while 1:
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0
            while (not success and M > 0):
                try:
                    M = M - 1
                    i = i + (length < 0)
                    f3, df3 = f(X+x3*s, *varargin)
                    if np.isnan(f3) or np.isinf(f3) or any(np.isnan(df3)+np.isinf(df3)):
                        raise Exception('error')
                    success = 1
                except Exception as ex:
                    x3 = (x2+x3)/2
            if f3 < F0:
                X0 = X+x3*s
                F0 = f3
                dF0 = df3
            d3 = np.sum(df3*s)
            if (d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0):
                break
            x1 = x2
            f1 = f2
            d1 = d2
            x2 = x3
            f2 = f3
            d2 = d3
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            tmp = B*B-A*d1*(x2-x1)
            if(tmp >= 0):
                x3 = x1-d1*(x2-x1)**2/(B+np.sqrt(tmp))
            if (tmp < 0 or np.isnan(x3) or np.isinf(x3) or x3 < 0):
                x3 = x2*EXT
            elif (x3 > x2*EXT):
                x3 = x2*EXT
            elif (x3 < x2+INT*(x2-x1)):
                x3 = x2+INT*(x2-x1)

        while ((np.abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0):
            if (d3 > 0 or f3 > f0+x3*RHO*d0):
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                x2 = x3
                f2 = f3
                d2 = d3
            if f4 > f0:
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
            if (np.isnan(x3) or np.isinf(x3)):
                x3 = (x2+x4)/2
            x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2))
            f3, df3 = f(X+x3*s, *varargin)
            if f3 < F0:
                X0 = X+x3*s
                F0 = f3
                dF0 = df3
            M = M - 1
            i = i + (length < 0)
            d3 = np.sum(df3*s)

        if (np.abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0):
            X = X+x3*s
            f0 = f3
            fX = np.concatenate((fX, [f0]))
            print("%s %6i;  Value %4.6e\r" % (S, i, f0))
            s = (np.sum(df3*df3)-np.sum(df0*df3))/np.sum(df0*df0)*s - df3
            df0 = df3
            d3 = d0
            d0 = np.sum(df0*s)
            if d0 > 0:
                s = -df0
                d0 = -np.sum(s*s)
            x3 = x3 * min(RATIO, d3/(d0-np.finfo(float).tiny))
            ls_failed = 0
        else:
            X = X0
            f0 = F0
            df0 = dF0
            if (ls_failed or i > np.abs(length)):
                break
            s = -df0
            d0 = -np.sum(s*s)
            x3 = 1/(1-d0)
            ls_failed = 1
    return X, fX  # , i


def learn_bocpd(X, useLogistic):

    max_minimize_iter = 30

    if useLogistic:
        model_f = 'gaussian1D'
        hazard_f = 'logistic_h'
        num_hazard_params = 3
        hazard_init = np.array([np.log(.01/(1-.01)), 0, 0])
        model_init = np.array([0, np.log(.1), np.log(.1), np.log(.1)])
        conversion = np.array([2, 0, 0, 0, 1, 1, 1])
    else:
        model_f = 'gaussian1D'
        hazard_f = 'constant_h'
        num_hazard_params = 1
        hazard_init = np.array(np.log(.01/(1-.01)))
        model_init = np.array([0, np.log(.1), np.log(.1), np.log(.1)])
        conversion = np.array([2, 0, 1, 1, 1])

    theta = np.concatenate((hazard_init, model_init))
    theta, nlml = rt_minimize(theta, bocpd_dwrap1D, -max_minimize_iter,
                              X, model_f, hazard_f, conversion, num_hazard_params)

    hazard_params = theta[:num_hazard_params]
    model_params = theta[num_hazard_params:]

    hazard_params[0] = 1 / (1 + np.exp(-hazard_params[0]))
    model_params[1:] = np.exp(model_params[1:])

    return hazard_params, model_params, nlml


def learn_IFM(X, useLogistic, init=None):

    D = X.shape[1]
    max_minimize_iter = 30

    if useLogistic:
        model_f = 'IFMfast'
        hazard_f = 'logistic_h'
        num_hazard_params = 3
        hazard_init = np.array([np.log(.01/(1-.01)), 0, 0])
        model_init = np.zeros(D*4)
        model_init[0:D] = 0
        model_init[D:] = np.log(1)
        conversion = np.zeros(3+4*D)
        conversion[:3] = [2, 0, 0]
        conversion[3:3+D] = 0
        conversion[3+D:] = 1
    else:
        model_f = 'IFMfast'
        hazard_f = 'constant_h'
        num_hazard_params = 1
        hazard_init = np.array(np.log(.01/(1-.01)))
        model_init = np.zeros(D*4)
        model_init[0:D] = 0
        model_init[D:] = np.log(1)

        conversion = np.zeros(1+4*D)
        conversion[0] = 2
        conversion[1:1+D] = 0
        conversion[1+D:] = 1

    if(init is not None):
        theta = init.copy()
        theta[conversion == 2] = np.log(
            theta[conversion == 2]/(1-theta[conversion == 2]))
        theta[conversion == 1] = np.log(theta[conversion == 1])
    else:
        theta = np.concatenate((hazard_init, model_init))

    theta, nlml = rt_minimize(theta, bocpd_dwrap_sp, -max_minimize_iter,
                              X, model_f, hazard_f, conversion, num_hazard_params)
    theta[conversion == 2] = 1 / (1 + np.exp(-theta[conversion == 2]))
    theta[conversion == 1] = np.exp(theta[conversion == 1])

    hazard_params = theta[:num_hazard_params]
    model_params = theta[num_hazard_params:]

    return hazard_params, model_params, nlml


def bocpd(X, model_f, theta_m, hazard_f, theta_h):

    T = len(X)
    D = 1
    R = np.zeros([T + 1, T + 1])
    S = np.zeros([T, T])
    R[0, 0] = 1

    Z = np.zeros(T)
    post_params = eval(model_f+'_init')(T + 1, D, theta_m)

    for t in range(0, T):

        predprobs = eval(model_f+'_predict')(post_params, X[t])
        H = eval(hazard_f)(np.arange(t+1)+1, theta_h)
        R[1:t + 2, t + 1] = R[:t+1, t] * predprobs * (1 - H)
        R[0, t + 1] = np.sum(R[:t+1, t] * predprobs * H)
        Z[t] = sum(R[:t + 2, t + 1])
        R[:t + 2, t + 1] = R[:t + 2, t + 1] / Z[t]
        S[:t+1, t] = R[:t+1, t] * predprobs
        S[:, t] = S[:, t] / np.sum(S[:, t])
        post_params = eval(model_f+'_update')(theta_m, post_params, X[t])
    nlml = -sum(np.log(Z))

    return R, S, nlml, Z


def bocpd_sparse(theta_h, theta_m, X, hazard_f, model_f, epsilon):

    num_hazard_params = len(theta_h)
    num_model_params = len(theta_m)

    [T, D] = X.shape

    R = [None]*(T+1)
    S = [None]*T

    R[0] = [1]
    Z = np.zeros([T])
    post_params = eval(model_f+'_init')(T + 1, D, theta_m)
    maxRunConsidered = 1
    for t in range(T):
        Rnew = np.zeros(maxRunConsidered + 1)
        predprobs = eval(model_f+'_predict')(post_params, X[t])
        H = eval(hazard_f)(np.arange(maxRunConsidered)+1, theta_h)
        Rnew[1:] = R[t] * predprobs * (1 - H)
        Rnew[0] = np.sum(R[t] * predprobs * H)

        Z[t] = np.sum(Rnew)
        Rnew = Rnew / Z[t]
        Rpruned, _ = pruneR(Rnew, epsilon)
        maxRunConsidered = Rpruned.shape[0]        
        R[t + 1] = Rpruned
        S[t] = R[t] * predprobs
        S[t] = S[t] / sum(S[t])

        post_params = eval(model_f+'_update')(theta_m,
                                              post_params, X[t], None, maxRunConsidered)
    print(maxRunConsidered)

    nlml = -np.sum(np.log(Z))
    R = convertRtoMatrix(R)
    S = convertRtoMatrix(S)
    return R, S, nlml, Z


def bocpd_dwrap1D(theta0, X, model_f, hazard_f, conversion, num_hazard_params):

    theta = theta0.copy()
    theta[conversion == 2] = 1 / (1 + np.exp(-theta[conversion == 2]))
    theta[conversion == 1] = np.exp(theta[conversion == 1])

    theta_h = theta[:num_hazard_params]
    theta_m = theta[num_hazard_params:]
    nlml, dnlml_h, dnlml_m = bocpd_deriv(
        theta_h, theta_m, X, hazard_f, model_f)

    dnlml = np.concatenate((dnlml_h, dnlml_m))

    dnlml[conversion == 2] = dnlml[conversion == 2] * \
        theta[conversion == 2] * (1 - theta[conversion == 2])
    dnlml[conversion == 1] = dnlml[conversion == 1] * theta[conversion == 1]
    return nlml, dnlml


def bocpd_deriv(theta_h, theta_m, X, hazard_f, model_f):

    num_hazard_params = len(theta_h)
    num_model_params = len(theta_m)

    T = len(X)
    D = 1
    R = np.zeros([T + 1, T + 1])
    dR_h = np.zeros([T + 1, T + 1, num_hazard_params])
    dR_m = np.zeros([T + 1, T + 1, num_model_params])
    R[0, 0] = 1

    Z = np.zeros(T)
    dZ_h = np.zeros([T, num_hazard_params])
    dZ_m = np.zeros([T, num_model_params])

    post_params, dmessage = eval(model_f+'_init')(T + 1, D, theta_m, 2)

    for t in range(0, T):

        predprobs, dpredprobs = eval(
            model_f+'_predict')(post_params, X[t], dmessage, 2)
        H, dH = eval(hazard_f)(np.arange(t+1)+1, theta_h, 2)
        R[1:t + 2, t + 1] = R[:t+1, t] * predprobs * (1 - H)
        for ii in range(num_hazard_params):
            dR_h[1:t + 2, t + 1, ii] = predprobs * \
                (dR_h[:t+1, t, ii] * (1 - H) - R[:t+1, t] * dH[:, ii])

        for ii in range(0, num_model_params):
            dR_m[1:t + 2, t + 1, ii] = (1 - H) * (dR_m[:t+1, t, ii]
                                                  * predprobs + R[:t+1, t] * dpredprobs[:, ii])
        R[0, t + 1] = np.sum(R[:t+1, t] * predprobs * H)
        for ii in range(num_hazard_params):
            dR_h[0, t + 1, ii] = np.sum(predprobs *
                                        (dR_h[:t+1, t, ii] * H + R[:t+1, t] * dH[:, ii]))
        for ii in range(num_model_params):
            dR_m[0, t + 1, ii] = np.sum(H * (dR_m[:t+1, t, ii]
                                             * predprobs + R[:t+1, t] * dpredprobs[:, ii]))
        Z[t] = sum(R[:t + 2, t + 1])
        for ii in range(num_hazard_params):
            dZ_h[t, ii] = sum(dR_h[:t + 2, t + 1, ii])
        for ii in range(num_model_params):
            dZ_m[t, ii] = sum(dR_m[:t + 2, t + 1, ii])
        for ii in range(num_hazard_params):
            dR_h[:t + 2, t + 1, ii] = (dR_h[:t + 2, t + 1, ii] / Z[t]) - \
                (dZ_h[t, ii] * R[:t + 2, t + 1]) / (Z[t] ** 2)
        for ii in range(num_model_params):
            dR_m[:t + 2, t + 1, ii] = (dR_m[:t + 2, t + 1, ii] / Z[t]) - \
                (dZ_m[t, ii] * R[:t + 2, t + 1]) / (Z[t] ** 2)
        R[:t + 2, t + 1] = R[:t + 2, t + 1] / Z[t]
        post_params, dmessage = eval(
            model_f+'_update')(theta_m, post_params, X[t], dmessage, 2)
    nlml = -sum(np.log(Z))
    dnlml_h = np.zeros(num_hazard_params)
    dnlml_m = np.zeros(num_model_params)
    for ii in range(num_hazard_params):
        dnlml_h[ii] = -np.sum(dZ_h[:, ii] / Z)
    for ii in range(num_model_params):
        dnlml_m[ii] = -np.sum(dZ_m[:, ii] / Z)

    return nlml, dnlml_h, dnlml_m  # , Z , dZ_h, dZ_m, R, dR_h, dR_m


def bocpd_dwrap_sp(theta0, X, model_f, hazard_f, conversion, num_hazard_params):

    theta = theta0.copy()
    
    theta[conversion == 2] = 1 / (1 + np.exp(-theta[conversion == 2]))
    theta[conversion == 1] = np.exp(theta[conversion == 1])
    
    theta_h = theta[:num_hazard_params]
    theta_m = theta[num_hazard_params:]
    nlml, dnlml_h, dnlml_m = bocpd_deriv_sparse(
        theta_h, theta_m, X, hazard_f, model_f, .001)

    
    dnlml = np.concatenate((dnlml_h, dnlml_m))

    dnlml[conversion == 2] = dnlml[conversion == 2] * \
        theta[conversion == 2] * (1 - theta[conversion == 2])
    dnlml[conversion == 1] = dnlml[conversion == 1] * theta[conversion == 1]    
    return nlml, dnlml


def bocpd_deriv_sparse(theta_h, theta_m, X, hazard_f, model_f, epsilon):

    num_hazard_params = len(theta_h)
    num_model_params = len(theta_m)

    [T, D] = X.shape

    R = [None]*(T+1)
    R[0] = 1

    Z = np.zeros(T)
    dZ_h = np.zeros([T, num_hazard_params])
    dZ_m = np.zeros([T, num_model_params])
    post_params, dmessage = eval(model_f+'_init')(1, D, theta_m, 2)
    maxRunConsidered = 1
    dR_h_last = np.zeros([maxRunConsidered, num_hazard_params])
    dR_m_last = np.zeros([maxRunConsidered, num_model_params])
    dR_h_curr = np.zeros([maxRunConsidered + 1, num_hazard_params])
    dR_m_curr = np.zeros([maxRunConsidered + 1, num_model_params])
    for t in range(0, T):
        Rnew = np.zeros([maxRunConsidered + 1, 1])
        predprobs, dpredprobs = eval(
            model_f+'_predict')(post_params, X[t], dmessage, 2)
        H, dH = eval(hazard_f)(np.arange(maxRunConsidered)+1, theta_h, 2)
        H = np.reshape(H, [len(H), 1])
        predprobs = np.reshape(predprobs, [len(predprobs), 1])

        Rnew[1:] = R[t] * predprobs * (1 - H)
        PRpart = dR_h_last * (1-H) - dH * R[t]
        dR_h_curr[1:, :] = PRpart * predprobs
        PRpart = dR_m_last * predprobs + dpredprobs * R[t]
        dR_m_curr[1:, :] = PRpart * (1-H)

        Rnew[0] = np.sum(R[t] * predprobs * H)

        PRpart = dR_h_last * H + dH * R[t]
        dR_h_curr[0, :] = np.sum(PRpart * predprobs, axis=0)
        PRpart = dR_m_last * predprobs + dpredprobs * R[t]
        dR_m_curr[0, :] = np.sum(PRpart * H, axis=0)
        Z[t] = np.sum(Rnew)

        dZ_h[t, :] = np.sum(dR_h_curr, axis=0)
        dZ_m[t, :] = np.sum(dR_m_curr, axis=0)
        dR_h_curr = dR_h_curr / Z[t] - Rnew * dZ_h[t, :] / (Z[t] ** 2)
        dR_m_curr = dR_m_curr / Z[t] - Rnew * dZ_m[t, :] / (Z[t] ** 2)
        Rnew = Rnew / Z[t]

        [Rpruned, renorm] = pruneR(Rnew, epsilon)
        maxRunConsidered = Rpruned.shape[0]
        R[t + 1] = Rpruned
        dR_h_curr = dR_h_curr[:maxRunConsidered, :]
        dR_m_curr = dR_m_curr[:maxRunConsidered, :]

        dR_h_curr = dR_h_curr / renorm - \
            Rnew[:maxRunConsidered] * np.sum(dR_h_curr, axis=0) / (renorm ** 2)
        dR_m_curr = dR_m_curr / renorm - \
            Rnew[:maxRunConsidered] * np.sum(dR_m_curr, axis=0) / (renorm ** 2)

        post_params, dmessage = eval(
            model_f+'_update')(theta_m, post_params, X[t], dmessage, maxRunConsidered, 2)

        dR_h_last = dR_h_curr
        dR_m_last = dR_m_curr
        dR_h_curr = np.zeros([maxRunConsidered + 1, num_hazard_params])
        dR_m_curr = np.zeros([maxRunConsidered + 1, num_model_params])

    nlml = -np.sum(np.log(Z))
    dnlml_h = np.zeros(num_hazard_params)
    dnlml_m = np.zeros(num_model_params)
    for ii in range(num_hazard_params):
        dnlml_h[ii] = -np.sum(dZ_h[:, ii] / Z)
    for ii in range(num_model_params):
        dnlml_m[ii] = -np.sum(dZ_m[:, ii] / Z)
    return nlml, dnlml_h, dnlml_m  # , Z , dZ_h, dZ_m, R, dR_h, dR_m


def whitten(x, N):
    min_lambda = 1e-10

    T = x.shape[0]
    x_bar = np.mean(x[:N, :], axis=0)
    sigma = np.cov(x[:N, :], rowvar=False)
    lamd, U = np.linalg.eig(sigma)
    lamd[lamd <= min_lambda] = min_lambda
    lamd_inv = np.diag(1 / np.sqrt(lamd))

    U = np.matrix(U)
    lamd_inv = np.matrix(lamd_inv)
    x_bar = np.matrix(x_bar)
    x = np.matrix(x)
    o = np.matrix(np.ones([T, 1]))

    Y = np.asarray(np.transpose(
        (U * lamd_inv * np.transpose(U) * np.transpose(x - o * x_bar))))

    return Y


def pruneR(R, epsilon):
    prunes = np.argwhere(R >= epsilon)
    Rpruned = R[:prunes[-1,0]+1]
    renorm = np.sum(Rpruned)
    Rpruned = Rpruned / renorm
    return Rpruned, renorm


def setupGcache(theta_prior, N, D):

    global Gcache
    global Pcache

    alpha_prior = theta_prior[2 * D:3 * D]
    nu = np.tile(np.reshape(np.arange(0, N+1),
                            [N+1, 1]), [1, D]) + 2 * alpha_prior
    Gcache = np.exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2))
    Pcache = psi(nu / 2 + .5) - psi(nu / 2) - (1 / nu)


def convertToAlert(Rs, thold):

    max_run, T = Rs.shape
    last_alarm = np.inf
    alert = np.zeros(T)
    for ii in range(T):
        alert[ii], last_alarm = convertToAlertSingle(
            Rs[:, ii], last_alarm, thold)
    return alert


def convertToAlertSingle(Rs, last_alarm, thold):
    if last_alarm > len(Rs):
        last_alarm = len(Rs)

    changePointProb = np.sum(Rs[:last_alarm+1])
    if changePointProb >= thold:
        alert = True
        last_alarm = 1
    else:
        alert = False
        last_alarm = last_alarm + 1

    return alert, last_alarm


def getMedianRunLength(S):
    T = S.shape[1]
    cdf = np.cumsum(S, 0)
    secondHalf = (cdf >= .5)
    Mrun = np.zeros(T)
    for ii in range(T):
        Mrun[ii] = np.amin(np.where(secondHalf[:, ii]))
    MchangeTime = np.arange(T) - Mrun + 1
    return Mrun  # , MchangeTime


def plotS(S, X):

    alertThold = .5

    plt.subplot(2, 1, 1)
    plt.plot(X)
    alert = convertToAlert(S, alertThold)
    plt.plot(np.where(alert), 0, 'rx')

    plt.axis('tight')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.imshow(np.cumsum(S, 0), cmap='gray')
    Mrun = getMedianRunLength(S)
    plt.plot(Mrun, 'r-')
    plt.axis('tight')


def convertRtoMatrix(R):

    T = len(R)
    maxRun = 0
    for t in range(T):
        if(maxRun < len(R[t])):
            maxRun = len(R[t])
    Rmat = np.zeros([maxRun, T])
    for t in range(T):
        Rmat[:len(R[t]), t] = R[t]
    return Rmat


def getMedianRunLength(S):

    T = S.shape[1]
    cdf = np.cumsum(S, axis=0)
    secondHalf = (cdf >= .5)
    Mrun = np.zeros(T)
    for ii in range(T):
        Mrun[ii] = np.argmax(secondHalf[:, ii])+1
    MchangeTime = np.arange(1, T+1) - Mrun + 1
    return Mrun, MchangeTime
