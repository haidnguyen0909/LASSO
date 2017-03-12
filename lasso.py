
from numpy import linalg
import numpy as np
import matplotlib.pyplot as pl
import math


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def fista(X, y, l, maxit):
    list_loss = []
    beta = np.zeros(X.shape[1])
    alpha = np.zeros(X.shape[1])
    t = 0
    L = linalg.norm(X) ** 2
    for _ in range(maxit):
        beta_old = beta.copy()
        alpha_old = alpha.copy()
        t_old = t

        temp = beta_old - X.T.dot(X.dot(beta_old) - y)/L
        alpha = soft_thresh(temp, l/L)
        t = (1. + math.sqrt(1 + 4 * t_old * t_old))/2.
        sigma = (1-t_old)/t
        beta = (1 - sigma) * alpha + sigma * alpha_old

        loss = 0.5 * linalg.norm(X.dot(beta) - y) ** 2 + l * linalg.norm(beta, 1)
        list_loss.append(loss)
    return beta, list_loss


def ista(X, y, l, maxit):
    list_lost = []
    beta = np.zeros(X.shape[1])
    z = np.zeros(X.shape[1])
    L = linalg.norm(X) ** 2
    for _ in range(maxit):
        z = beta - np.dot(X.T, X.dot(beta) - y) / L
        beta = soft_thresh(z, l/L)
        loss = 0.5 * linalg.norm(X.dot(beta) - y) ** 2 + l * linalg.norm(beta, 1)
        list_lost.append(loss)
        print(loss)
    return beta, list_lost


def main():
    m,n = 500, 1000
    rng = np.random.RandomState(42)
    X = rng.randn(m , n)
    beta0 = rng.rand(n)
    y = np.dot(X, beta0)
    beta0[beta0 < 0.9] = 0
    print(beta0)
    l = 0.5 # regu. param

    maxit = 500
    beta_ista, list_loss_ista = ista(X, y, l, maxit)
    beta_fista, list_loss_fista = fista(X, y, l, maxit)

    pl.figure()
    niters = range(maxit)

    pl.plot(niters, list_loss_ista, label = 'ista')
    pl.plot(niters, list_loss_fista, label='fista')
    pl.legend(['ISTA', 'FISTA'], loc = 'upper right')
    pl.xlabel("iteration")
    pl.ylabel("loss")
    pl.title('Comparison of ISTA vs. FISTA in terms of convergence rate')
    pl.savefig('lasso')
    pl.show()






if __name__ == "__main__":
    main()


