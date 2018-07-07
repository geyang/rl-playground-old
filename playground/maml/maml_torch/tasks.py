import numpy as np


class Sine:
    def __init__(self, npts=100, domain=(-5, 5), amp=None, phi0=None):
        self.npts = npts
        self.domain = domain  # this is the domain on which function is defined.
        self.f0 = 1
        self.amp = amp or np.random.uniform(0.1, 5.0)
        self.offset = 0
        self.phi0 = phi0 or np.random.uniform(0, np.pi)

    def __str__(self):
        return f"f0={self.f0:0.2f}, amp={self.amp:0.2f}"

    def fn(self, xs):
        return np.sin(self.f0 * xs + self.phi0) * self.amp

    def proper(self):
        """
        returns n evenly spaced points. Used for plotting.
        this method should be idempotent. should not mutate internal state.
        """
        xs = np.linspace(*self.domain, self.npts)
        return xs, self.fn(xs)

    def samples(self, n, domain=None, sort=False):
        """Generates n samples (uniformly) in domain.
        :param n: number of samples to generate
        :param domain: this is the domain on which to sample
        :return: xs, ys
        """
        xs = np.random.uniform(*(domain or self.domain), n)
        xs = np.sort(xs) if sort else xs  # return sorted so that it is easier to use
        ys = self.fn(xs)
        return xs, ys
        # xs, ys = self.proper()
        # i = np.random.randint(0, self.npts, size=n)
        # return xs[i], ys[i]


def Same(_ts=[Sine()]):
    return _ts[0]
