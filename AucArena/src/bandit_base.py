import numpy as np
import math
from scipy import linalg
import copy


# TS
class TS:
    def __init__(self, arms, sigmultiplier='fixed'):
        # upper bound coefficient
        self.alpha = 0.1
        # dimension of user features = d
        self.d = 2
        self._lambda_prior = 0.001
        self.R = 0.01
        self.epsilon = 0.1
        self.delta = 0.5
        self.t = 30

        self.x = None
        self.xT = None

        self.arms = arms
        self.sigmultiplier = sigmultiplier
        self.context_free = True

        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = []
        # AaI : store the inverse of all Aa matrix
        self.AaI = []
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = []
        self.mu_pos = []
        self.sigma_pos = []

        self.a_max = []
        self.features = []
        self.data_cumulative = {}
        self.set_arms()

    def set_arms(self):
        # init collection of matrix/vector Aa, Ba, ba
        self.Aa = [self._lambda_prior * np.identity(self.d) for _ in self.arms]
        self.cumA = [np.identity(self.d) for _ in self.arms]
        self.ba = [np.zeros((self.d, 1)) for _ in self.arms]
        self.cumb = [np.zeros((self.d, 1)) for _ in self.arms]

        self.AaI = [(1 / self._lambda_prior) * np.identity(self.d) for _ in self.arms]

        self.mu_pos = [self.AaI[key].dot(self.ba[key]) for key in self.arms]
        self.features = [np.array([[1, 0]]) for _ in self.arms]

        self.sig_sq = self.R * np.sqrt((24 / self.epsilon) * self.d * np.log(self.t / self.delta))

        self.sigma_pos = [self.sig_sq * self.AaI[key] for key in self.arms]

        for i in self.arms:
            self.data_cumulative[i] = {'context': np.array([]),
                                       'cumA': np.identity(self.d),
                                       'cumb': np.zeros((self.d, 1)),
                                       'time_step': 0,
                                       'cnt': 0}

    def update(self, reward, i):
        self.sig_sq = 5 / (i % 10 + 1)
        for idx in self.data_cumulative:
            if self.data_cumulative[idx]['time_step'] > 0:
                print(f"reward: {reward}")
                print(f"{idx}: {self.mu_pos[idx]}")
                print(f"{idx}: {self.sigma_pos[idx]}")
                self.Aa[idx] += self.data_cumulative[idx]['cumA']
                self.ba[idx] += reward * self.data_cumulative[idx]['cumb']
                self.AaI[idx] = linalg.solve(self.Aa[idx], np.identity(self.d))
                self.mu_pos[idx] = self.AaI[idx].dot(self.ba[idx])
                self.sigma_pos[idx] = self.sig_sq * self.AaI[idx]
                print(f"{idx}: {self.mu_pos[idx]}")
                print(f"{idx}: {self.sigma_pos[idx]}")

        for idx in self.arms:
            self.data_cumulative[idx]['time_step'] = 0
            self.data_cumulative[idx]['cumb'] = np.zeros((self.d, 1))
            self.data_cumulative[idx]['cumA'] = np.identity(self.d)

    def recommend(self):
        xaT = np.array([self.features[key].transpose() for key in self.arms])
        theta = np.array([
            np.random.multivariate_normal(self.mu_pos[idx].ravel(), self.sigma_pos[idx])
            for idx in self.arms
        ])
        pa = np.array([theta[idx].dot(xaT[idx])[0] for idx in self.arms])
        self.a_max = pa.argsort()[-5:]
        path = [self.arms[idx] for idx in self.a_max]
        return self.a_max, path

    def update_offline_data(self):
        for idx in self.a_max:
            xT = self.features[idx]
            x = np.transpose(xT)
            self.data_cumulative[idx]['cumA'] += x.dot(xT)
            self.data_cumulative[idx]['cumb'] += x
            self.data_cumulative[idx]['time_step'] += 1
            self.data_cumulative[idx]['cnt'] += 1

            if len(self.data_cumulative[idx]['context']) == 0:
                self.data_cumulative[idx]['context'] = copy.deepcopy(np.array(self.features[idx][0]))
            else:
                self.data_cumulative[idx]['context'] = np.vstack(
                    (self.data_cumulative[idx]['context'], np.array(self.features[idx][0])))


# LinTS
class LinTS:
    def __init__(self, arms, sigmultiplier='fixed'):
        # upper bound coefficient
        self.alpha = 0.1
        # dimension of user features = d
        self.d = 2
        self._lambda_prior = 0.001
        self.R = 0.01
        self.epsilon = 0.1
        self.delta = 0.5
        self.t = 30

        self.x = None
        self.xT = None

        self.arms = arms
        self.sigmultiplier = sigmultiplier
        self.context_free = False

        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = []
        # AaI : store the inverse of all Aa matrix
        self.AaI = []
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = []
        self.mu_pos = []
        self.sigma_pos = []

        self.a_max = []
        self.features = []
        self.data_cumulative = {}
        self.set_arms()

    def set_arms(self):
        # init collection of matrix/vector Aa, Ba, ba
        self.Aa = [self._lambda_prior * np.identity(self.d) for _ in self.arms]
        self.cumA = [np.identity(self.d) for _ in self.arms]
        self.ba = [np.zeros((self.d, 1)) for _ in self.arms]
        self.cumb = [np.zeros((self.d, 1)) for _ in self.arms]

        self.AaI = [(1 / self._lambda_prior) * np.identity(self.d) for _ in self.arms]

        self.mu_pos = [self.AaI[key].dot(self.ba[key]) for key in self.arms]
        self.features = [np.zeros((1, 2)) for _ in self.arms]

        self.sig_sq = self.R * np.sqrt((24 / self.epsilon) * self.d * np.log(self.t / self.delta))
        self.sigma_pos = [self.sig_sq * self.AaI[key] for key in self.arms]

        for i in self.arms:
            self.data_cumulative[i] = {'context': np.array([]),
                                       'cumA': np.identity(self.d),
                                       'cumb': np.zeros((self.d, 1)),
                                       'time_step': 0,
                                       'cnt': 0}

    def update(self, reward, i):
        self.sig_sq = 5 / (i % 10 + 1)
        for idx in self.data_cumulative:
            if self.data_cumulative[idx]['time_step'] > 0:
                print(f"reward: {reward}")
                print(f"{idx}: {self.mu_pos[idx]}")
                print(f"{idx}: {self.sigma_pos[idx]}")
                self.Aa[idx] += self.data_cumulative[idx]['cumA']
                self.ba[idx] += reward * self.data_cumulative[idx]['cumb']
                self.AaI[idx] = linalg.solve(self.Aa[idx], np.identity(self.d))
                self.mu_pos[idx] = self.AaI[idx].dot(self.ba[idx])
                self.sigma_pos[idx] = self.sig_sq * self.AaI[idx]
                print(f"{idx}: {self.mu_pos[idx]}")
                print(f"{idx}: {self.sigma_pos[idx]}")

        for idx in self.arms:
            self.data_cumulative[idx]['time_step'] = 0
            self.data_cumulative[idx]['cumb'] = np.zeros((self.d, 1))
            self.data_cumulative[idx]['cumA'] = np.identity(self.d)

    def get_features(self, ids):
        for k, _ in self.arms.items():
            if k in ids:
                self.features[k][0] = [1, 1]
            else:
                self.features[k][0] = [1, 0]

    def recommend(self, ids):
        self.get_features(ids)
        xaT = np.array([self.features[key].transpose() for key in self.arms])
        theta = np.array([
            np.random.multivariate_normal(self.mu_pos[idx].ravel(), self.sigma_pos[idx])
            for idx in self.arms
        ])
        pa = np.array([theta[idx].dot(xaT[idx])[0] for idx in self.arms])
        self.a_max = pa.argsort()[-5:]
        path = [self.arms[idx] for idx in self.a_max]
        return self.a_max, path

    def update_offline_data(self):
        for idx in self.a_max:
            xT = self.features[idx]
            x = np.transpose(xT)
            self.data_cumulative[idx]['cumA'] += x.dot(xT)
            self.data_cumulative[idx]['cumb'] += x
            self.data_cumulative[idx]['time_step'] += 1
            self.data_cumulative[idx]['cnt'] += 1

            if len(self.data_cumulative[idx]['context']) == 0:
                self.data_cumulative[idx]['context'] = copy.deepcopy(np.array(self.features[idx][0]))
            else:
                self.data_cumulative[idx]['context'] = np.vstack(
                    (self.data_cumulative[idx]['context'], np.array(self.features[idx][0])))
