

import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import date
from functions import *
import matplotlib.dates as mdates

if __name__ == '__main__':
    print('Trying industry portfolio data')

    random.seed(4)

    Tlearn = 1000
    thirty_industry = np.genfromtxt(
        'data/thirty_industry.dat', delimiter=",")[:, :]
    year = np.floor(thirty_industry[:, 0] / 10000).astype('int')
    month = np.mod(np.floor(thirty_industry[:, 0] / 100), 100).astype('int')
    day = np.mod(thirty_industry[:, 0], 100).astype('int')
    time = np.array([date.toordinal(date(year[i], month[i], day[i])) for i in range(
        len(year))])   #note: python and matlab representations are different
    X = whitten(thirty_industry[:, 1:], Tlearn)

    [T, D] = X.shape
    Ttest = T - Tlearn

    print('learn the models independently')
    industry_hazard = np.zeros([3, D])
    industry_model = np.zeros([4, D])
    industry_learning = [None]*D
    Z = np.zeros([T, D])
    theta_init = None
    for ii in range(D):
        print(ii)
        industry_hazard[:, ii], industry_model[:, ii], industry_learning[ii] = learn_IFM(
            X[:Tlearn, ii:ii+1], True, theta_init)
        R, S, nlml, Z[:, ii] = bocpd_sparse(
            industry_hazard[:, ii], industry_model[:, ii], X[:, ii:ii+1], 'logistic_h', 'IFM', .001)
        theta_init = np.concatenate((np.mean(
            industry_hazard[:, :ii+1], axis=1), np.mean(industry_model[:, :ii+1], axis=1)))
    nlml_score = -np.sum(np.log(Z[Tlearn:, :])) / Ttest

    print('Learn the joint')
    tmp = industry_model
    theta_init = np.concatenate(
        (np.mean(industry_hazard, axis=1), np.ndarray.flatten(tmp)))
    industry_hazard_joint, industry_model_joint, industry_learning_joint = learn_IFM(
        X[:Tlearn, :], True, theta_init)

    print('Testing the joint')
    industry_R, industry_S, industry_nlml, Zjoint = bocpd_sparse(
        industry_hazard_joint, industry_model_joint, X, 'logistic_h', 'IFM', .001)

    nlml_score_joint = -np.sum(np.log(Zjoint[Tlearn:])) / Ttest
    TIM_nlml = -np.sum((-0.5 * X[Tlearn:, :]**2) -
                       np.log(np.sqrt(2*np.pi)))/Ttest
    
    df = 4
    Xprime = t.cdf(X, df)
    Xprime = norm.ppf(Xprime)

    industry_hazard_heavy, industry_model_heavy, industry_learning_heavy = learn_IFM(
        Xprime[:Tlearn, :], True, np.concatenate((industry_hazard_joint, industry_model_joint)))
    industry_RH, industry_SH, industry_nlml, Zheavy = bocpd_sparse(
        industry_hazard_heavy, industry_model_heavy, Xprime, 'logistic_h', 'IFM', .001)
    nlml_score_joint = -np.sum(np.log(Zheavy[Tlearn:])) / Ttest
    TIM_nlml = - \
        np.sum((-0.5 * Xprime[Tlearn:, :]**2) - np.log(np.sqrt(2*np.pi)))/Ttest

    industry_S=np.save('industry_S',industry_S)
    print('Array industry_S is saved in industry_S.npy')
   
    fig, ax = plt.subplots(figsize=(20,5))
    plt.imshow(np.cumsum(industry_S, axis=0)[
               ::-1], extent=[time[0], time[-1], 1, industry_S.shape[0]])
    plt.gca().invert_yaxis()
    Mrun, _ = getMedianRunLength(industry_S)
    plt.plot(time, Mrun, 'r-')
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    plt.xticks(rotation=90)
    ax.set_aspect('auto')
    plt.savefig('result.png')
    print('Plot is saved as result.pngArray')
    plt.show()
    
