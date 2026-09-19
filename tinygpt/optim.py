# coding: utf-8
'''
Adam, the only optimizer here. SGD does not train transformers.
'''

import math

import numpy as np

from tinygpt.backend import vsqrt


class Adam(object):
    '''Adam with bias correction.'''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr

        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t

        sqrt_bc2 = math.sqrt(bc2)

        # Equivalent to:
        #
        #   mh = m / bc1
        #   vh = v / bc2
        #   prm -= lr * mh / (sqrt(vh) + eps)
        #
        # but avoids materialising mh and vh.
        step_scale = lr * sqrt_bc2 / bc1
        eps_scaled = self.eps * sqrt_bc2

        for k, prm in params.items():
            gr = grads[k]

            if self.wd and prm.ndim >= 2:
                gr = gr + self.wd * prm

            m = self.m[k]
            v = self.v[k]

            m *= self.b1
            m += (1.0 - self.b1) * gr

            v *= self.b2
            v += (1.0 - self.b2) * (gr * gr)

            # tmp = sqrt(v)
            #
            # Then recycle tmp for the rest of the update instead of creating
            # sqrt(v), denominator, normalized moment, and update arrays.
            tmp = vsqrt(v)
            tmp += eps_scaled

            np.divide(m, tmp, out=tmp)
            tmp *= step_scale

            prm -= tmp
