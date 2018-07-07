from .base import Baseline
import numpy as np


class LinearFeatureBaseline(Baseline):
    def __init__(self, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def get_param_values(self, **tags):
        return self._coeffs

    def set_param_values(self, val, **tags):
        self._coeffs = val

    @staticmethod
    def features(obs, rewards):
        o = np.clip(obs, -10, 10)  # hidden defaults are evil. -- Ge
        l = len(rewards)
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        """Here we fit each path separately."""
        obs = paths['obs'].swapaxes(0, 1)
        rewards = paths['rewards'].swapaxes(0, 1)
        featmat = np.concatenate([self.features(ob, r) for ob, r in zip(obs, rewards)])
        returns = paths['returns'].swapaxes(0, 1).reshape(-1)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs, *_ = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns), rcond=None, )
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, paths):
        if self._coeffs is None:
            return np.zeros(len(paths['']))

        n_timesteps, n_envs, *_ = paths['rewards'].shape
        obs = paths['obs'].swapaxes(0, 1)
        rewards = paths['rewards'].swapaxes(0, 1)
        featmat = np.concatenate([self.features(ob, r) for ob, r in zip(obs, rewards)])
        return featmat.dot(self._coeffs).reshape(n_envs, n_timesteps).swapaxes(0, 1)
