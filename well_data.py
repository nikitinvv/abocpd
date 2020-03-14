

import numpy as np
import random
from scipy.special import psi, gammaln
from scipy.stats import norm
import matplotlib.pyplot as plt
from functions import *

if __name__ == '__main__':
    print('Trying well log data')

    random.seed(4)

    well = np.genfromtxt('data/well.dat')

    X = (well - np.mean(well, axis=0)) / \
        (np.finfo(float).tiny + np.std(well, axis=0))
    Tlearn = 2000
    Ttest = X.shape[0] - Tlearn

    useLogistic = True
    print('Starting learning')
    well_hazard, well_model, well_learning = learn_bocpd(
        X[:Tlearn], useLogistic)
    print('Learning Done')

    print('Testing')
    well_R, well_S, well_nlml, Z = bocpd(
        X, 'gaussian1D', well_model, 'logistic_h', well_hazard)
    print('Done Testing')
    nlml_score = -np.sum(np.log(Z[Tlearn:])) / Ttest

    rpa_hazard = np.array([1 / 250])
    rpa_mu0 = 1.15e5
    rpa_mu_sigma = 1e4
    rpa_mu0 = (rpa_mu0 - np.mean(well)) / np.std(well)
    rpa_mu_sigma = rpa_mu_sigma / np.std(well)

    rpa_kappa = 1 / rpa_mu_sigma ** 2

    rpa_alpha = 1
    rpa_beta = rpa_kappa
    rpa_model = np.array([rpa_mu0, 1, rpa_alpha, rpa_beta])
    well_R, well_S_rpa, well_nlml_rpa, Z_rpa = bocpd(
        X, 'gaussian1D', rpa_model, 'constant_h', rpa_hazard)

    nlml_score_rpa = -np.sum(np.log(Z_rpa[Tlearn:])) / Ttest
    TIM_nlml = -np.sum(norm.logpdf(X[Tlearn:])) / Ttest

    plt.figure(figsize=(8, 7))
    plotS(well_S, X)
    plt.title(['RDT '+str(nlml_score)])

    plt.figure(figsize=(8, 7))
    plotS(well_S_rpa, X)
    plt.title(['RPA '+str(nlml_score_rpa)])
    plt.show()
