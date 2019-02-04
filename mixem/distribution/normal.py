# coding=utf-8
import numpy as np
import scipy.stats

from mixem.distribution.distribution import Distribution


class NormalDistribution(Distribution):
    """Univariate normal distribution with parameters (mu, sigma)."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_density(self, data):
        assert(len(data.shape) == 1), "Expect 1D data!"

        return - (data - self.mu) ** 2 / (2 * self.sigma ** 2) - np.log(self.sigma) - 0.5 * np.log(2 * np.pi)

    def estimate_parameters(self, data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"

        wsum = np.sum(weights)

        self.mu = np.sum(weights * data) / wsum
        self.sigma = np.sqrt(np.sum(weights * (data - self.mu) ** 2) / wsum)

    def __repr__(self):
        return "Norm[μ={mu:.4g}, σ={sigma:.4g}]".format(mu=self.mu, sigma=self.sigma)


class MultivariateNormalDistribution(Distribution):
    """Multivariate normal distribution with parameters (mu, Sigma)."""

    def __init__(self, mu, sigma, beta=None):
        mu = np.array(mu)
        sigma = np.array(sigma)

        assert len(mu.shape) == 1, "Expect mu to be 1D vector!"
        assert len(sigma.shape) == 2, "Expect sigma to be 2D matrix!"

        assert sigma.shape[0] == sigma.shape[1], "Expect sigma to be a square matrix!"

        self.mu = mu
        self.sigma = sigma
        self.beta = beta

    def log_density(self, data):
        logP = scipy.stats.multivariate_normal.logpdf(data, self.mu, self.sigma)
        if self.beta is not None:
            logP = self.beta * logP
        return logP

    def estimate_parameters(self, data, weights):
        self.mu = np.sum(data * weights[:, np.newaxis], axis=0) / np.sum(weights)

        center_x = data - self.mu[np.newaxis, :]

        # sigma = (np.diag(weights) @ center_x).T @ center_x / np.sum(weights)
        self.sigma = np.dot(
            np.dot(
                np.diag(weights),
                center_x
            ).T,
            center_x
        ) / np.sum(weights)
        if self.beta is not None:
            s,logDet = np.linalg.slogdet(self.sigma)
            assert s > 0.
            self.sigma = self.sigma/np.exp(logDet/float(len(self.sigma)))

    def __repr__(self):
        po = np.get_printoptions()

        np.set_printoptions(precision=3)

        try:
            result = "MultiNorm[μ={mu}, σ={sigma}]".format(mu=self.mu, sigma=str(self.sigma).replace("\n", ","))
        finally:
            np.set_printoptions(**po)

        return result
    
class diagMVN(Distribution):
    """Multivariate normal distribution with parameters (mu, Sigma)."""

    def __init__(self, mu, sigma, beta=None):
        mu = np.array(mu)
        sigma = np.array(sigma)

        assert len(mu.shape) == 1, "Expect mu to be 1D vector!"
        assert len(sigma.shape) == 1, "Expect sigma to be 1D vector"

#         assert sigma.shape[0] == sigma.shape[1], "Expect sigma to be a square matrix!"

        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.eps = 1E-4
        
    @property
    def invSigma(self,):
        res = np.diag( 1./ (self.sigma + self.eps ))
        return res
    
    def log_density(self, data):
        center_x = data - self.mu[None,:]
        logP = center_x[:,:,None] * center_x[:,None,:] * self.invSigma[ None, :, :]
        logP = -np.sum(logP,axis=(1,2))
#         logP = np.dot( np.dot( center_x, self.invSigma), center_x.T)
#         logP = n
#         logP = scipy.stats.multivariate_normal.logpdf(data, self.mu, self.sigma)
        if self.beta is not None:
            logP = self.beta * logP
        assert not np.any(np.isnan(logP))
        return logP

    def estimate_parameters(self, data, weights):
        SUM = np.sum(weights)
        if SUM==0:
            return
        else:
            normW = weights / np.sum(weights)
        self.mu = np.sum(data * normW[:,None], axis=0) 

        center_x = data - self.mu[np.newaxis, :]
        
        # sigma = (np.diag(weights) @ center_x).T @ center_x / np.sum(weights)
        E2  = np.sum( normW[:,None] * center_x ** 2, axis=0) 
        E1  = np.sum( normW[:,None] * center_x     , axis=0) 
        self.sigma = (E2 - E1 **2)
        self.sigma = np.maximum( 0. , self.sigma)
        assert not np.any(np.isnan(self.sigma))
        assert not np.any(np.isnan(self.mu))
#         self.sigma = np.std(center_x, axis=0)        
#         self.sigma = np.dot(
#             np.dot(
#                 np.diag(weights),
#                 center_x
#             ).T,
#             center_x
#         ) / np.sum(weights)
        
        if self.beta is not None:
            logDet = np.sum(np.log(self.sigma + self.eps))
#             logDet = np.sum(np.log(1 + self.sigma) )
            assert np.all(self.sigma >= 0.)
#             print ('[logDet]',logDet)
#             np.prod(
#             s,logDet = np.linalg.slogdet(self.sigma)
#             assert s > 0.

#             self.sigma = self.sigma/np.exp(logDet/float(len(self.sigma)))
#             self.sigma = self.sigma + 0.001

    def __repr__(self):
        po = np.get_printoptions()

        np.set_printoptions(precision=3)

        try:
            result = "MultiNorm[μ={mu}, σ={sigma}]".format(mu=self.mu, sigma=str(self.sigma).replace("\n", ","))
        finally:
            np.set_printoptions(**po)

        return result