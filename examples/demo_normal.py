#!/usr/bin/env python

import numpy as np
import mixem
import mixem.distribution


def generate_data():
    dist_params = [
        (4, 1),
        (1, 0.5)
    ]

    weights = [0.3, 0.7]

    n_data = 5000
    data = np.zeros((n_data,))
    for i in range(n_data):
        dpi = np.random.choice(range(len(dist_params)), p=weights)
        mu, sigma = dist_params[dpi]
        data[i] = np.random.normal(loc=mu, scale=sigma)

    return data


def progress(iter, weights, params, ll):
    print("{0:4d}: ll={1:.5e} w={2:.2f}     mu0={3:.4e} mu1={4:.4e}   s0={5:.4e} s1={6:.5e}".format(iter, ll, weights[0], params[0][0], params[1][0], params[0][1], params[1][1]))


def recover(data):

    mu = np.mean(data)
    sigma = np.var(data)

    init_params = [
        (mu + 0.1, sigma),
        (mu - 0.1, sigma)
    ]

    init_weights = [0.1, 0.9]

    weight, param, ll = mixem.em(data, [mixem.distribution.NormalDistribution] * 2, init_weights, init_params, progress_callback=progress)

    print(weight, param, ll)


if __name__ == '__main__':
    data = generate_data()
    recover(data)