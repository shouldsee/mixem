# -*- coding: utf-8 -*-
from mixem.distribution import Distribution
import numpy as np
import scipy.special

def l2norm(x,axis=None,keepdims=0):
    res = np.sum(x**2,axis=axis,keepdims=keepdims)**0.5
    return res

# class vmfDistribution(mixem.distribution.Distribution):
class vmfDistribution(Distribution):
    """Von-mises Fisher distribution with parameters (mu, kappa).
    Ref: Clustering on the Unit Hypersphere using von Mises-Fisher Distributions
    http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
    """

    def __init__(self, mu, 
                 kappa = None,
                 beta = 1.,
                 normalizeSample = False,
                 sample_weights = None,
                ):
        mu = np.array(mu)
        
        assert len(mu.shape) == 1, "Expect mu to be 1D vector!"
        if all(mu==0):
            self.dummy = True
        else:
            self.dummy = False
            
        if kappa is not None:
            assert len(np.shape(kappa)) == 0,"Expect kappa to be 0D vector"
            kappa = float(kappa)
        self.kappa = kappa
        self.beta = beta
        self.sample_weights = sample_weights
        self.mu = mu
        self.radius = np.sum(self.mu ** 2) ** 0.5
        self.D = len(mu)
    @property
    def params(self):
        return {'mu':self.mu,'kappa':self.kappa,'beta':self.beta}
    def log_density(self, data):
#         L2 = np.sum(np.square(data),axis=1,keepdims=1)
#         return  np.dot(data, self.mu) * L2 / L2
        logP = np.dot( data, self.mu) * self.beta
        if self.kappa is not None:
            biv = scipy.special.iv(self.D/2. -1.,  self.kappa )
            normTerm = ( - np.log(biv)
                          + np.log(self.kappa) * (self.D/2. - 1.)
                          - np.log(2*np.pi) * self.D/2.
                        ) 
            assert not np.isnan(normTerm).any()
            logP = logP * self.kappa + normTerm
        logP = logP + np.log(self._get_fct(data))
        return  logP
    def _get_fct(self, data,keepdims=0):
#         L2 = np.sum(np.square(data),axis=1,keepdims=keepdims)
#         L2sqrt = np.sqrt(L2)
#         fct = np.exp(L2sqrt)
        if self.sample_weights is not None:
            fct = self.sample_weights
            if keepdims == 1:
                fct = fct[:,None]
        else:
            fct = 1.
        return fct
    
    def estimate_parameters(self, data, weights):
        if not self.dummy:
#             fct = 1.
#             wdata = data * fct
            weights =  weights[:, np.newaxis]
#             weights =  weights[:, np.newaxis] * self._get_fct(data,keepdims=1)
            #     * self._get_fct(data,keepdims=1)
            wwdata  =  data * weights
            rvct = np.sum(wwdata, axis=0) / np.sum(weights,axis=0)
            rnorm = l2norm(rvct)
            self.mu = rvct / rnorm * self.radius
            
            if self.kappa is not None:
                r = rnorm
                new_kappa =  (r * self.D - r **3 )/(1. - r **2)
                assert new_kappa > 0.
                self.kappa = new_kappa
            
    def __repr__(self):
        po = np.get_printoptions()

        np.set_printoptions(precision=3)

        try:
            result = "MultiNorm[μ={mu},]".format(mu=self.mu,)
        finally:
            np.set_printoptions(**po)

        return result