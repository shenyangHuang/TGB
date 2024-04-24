"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer/blob/master/model/dirichlet.py
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""

"""Dirichlet.py
Maximum likelihood estimation and likelihood ratio tests of Dirichlet
distribution models of data.
Most of this package is a port of Thomas P. Minka's wonderful Fastfit MATLAB
code. Much thanks to him for that and his clear paper "Estimating a Dirichlet
distribution". See the following URL for more information:
    http://research.microsoft.com/en-us/um/people/minka/"""

import sys

import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.stats import dirichlet
from tqdm import tqdm
from numpy import (
    arange,
    array,
    asanyarray,
    asarray,
    diag,
    exp,
    isscalar,
    log,
    ndarray,
    ones,
    vstack,
    zeros,
)
from numpy.linalg import norm
from scipy.special import gammaln, polygamma, psi

MAXINT = sys.maxsize

__all__ = [
    "loglikelihood",
    "meanprecision",
    "mle",
    "pdf",
    "test",
]

euler = -1 * psi(1)  # Euler-Mascheroni constant


class NotConvergingError(Exception):
    """Error when a successive approximation method doesn't converge
    """
    pass


def test(D1, D2, method="meanprecision", maxiter=None):
    """Test for statistical difference between observed proportions.
    Parameters
    ----------
    D1 : (N1, K) shape array
    D2 : (N2, K) shape array
        Input observations. ``N1`` and ``N2`` are the number of observations,
        and ``K`` is the number of parameters for the Dirichlet distribution
        (i.e. the number of levels or categorical possibilities).
        Each cell is the proportion seen in that category for a particular
        observation. Rows of the matrices must add up to 1.
    method : string
        One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
        which to find MLE Dirichlet distribution. Default is
        ``'meanprecision'``, which is faster.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    D : float
        Test statistic, which is ``-2 * log`` of likelihood ratios.
    p : float
        p-value of test.
    a0 : (K,) shape array
    a1 : (K,) shape array
    a2 : (K,) shape array
        MLE parameters for the Dirichlet distributions fit to
        ``D1`` and ``D2`` together, ``D1``, and ``D2``, respectively."""

    N1, K1 = D1.shape
    N2, K2 = D2.shape
    if K1 != K2:
        raise ValueError("D1 and D2 must have the same number of columns")

    D0 = vstack((D1, D2))
    a0 = mle(D0, method=method, maxiter=maxiter)
    a1 = mle(D1, method=method, maxiter=maxiter)
    a2 = mle(D2, method=method, maxiter=maxiter)

    D = 2 * (loglikelihood(D1, a1) + loglikelihood(D2, a2) - loglikelihood(D0, a0))
    return (D, stats.chi2.sf(D, K1), a0, a1, a2)


def pdf(alphas):
    """Returns a Dirichlet PDF function
    Parameters
    ----------
    alphas : (K,) shape array
        The parameters for the distribution of shape ``(K,)``.
    Returns
    -------
    function
        The PDF function, takes an ``(N, K)`` shape input and gives an
        ``(N,)`` output.
    """
    alphap = alphas - 1
    c = np.exp(gammaln(alphas.sum()) - gammaln(alphas).sum())

    def dirichlet(xs):
        """Dirichlet PDF
        Parameters
        ----------
        xs : (N, K) shape array
            The ``(N, K)`` shape input matrix

        Returns
        -------
        (N,) shape array
            Point value for PDF
        """
        return c * (xs ** alphap).prod(axis=1)

    return dirichlet


def meanprecision(a):
    """Mean and precision of a Dirichlet distribution.
    Parameters
    ----------
    a : (K,) shape array
        Parameters of a Dirichlet distribution.
    Returns
    -------
    mean : (K,) shape array
        Means of the Dirichlet distribution. Values are in [0,1].
    precision : float
        Precision or concentration parameter of the Dirichlet distribution."""

    s = a.sum()
    m = a / s
    return (m, s)


def loglikelihood(D, a):
    """Compute log likelihood of Dirichlet distribution, i.e. log p(D|a).
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a : (K,) shape array
        Parameters for the Dirichlet distribution.
    Returns
    -------
    logl : float
        The log likelihood of the Dirichlet distribution"""
    N, K = D.shape
    logp = log(D).mean(axis=0)
    return N * (gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * logp).sum())


def mle(D, tol=1e-7, method="meanprecision", maxiter=None):
    """Iteratively computes maximum likelihood Dirichlet distribution
    for an observed data set, i.e. a for which log p(D|a) is maximum.
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    method : string
        One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
        which to find MLE Dirichlet distribution. Default is
        ``'meanprecision'``, which is faster.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    a : (K,) shape array
        Maximum likelihood parameters for Dirichlet distribution."""

    if method == "meanprecision":
        return _meanprecision(D, tol=tol, maxiter=maxiter)
    else:
        return _fixedpoint(D, tol=tol, maxiter=maxiter)


def _fixedpoint(D, tol=1e-7, maxiter=None):
    """Simple fixed point iteration method for MLE of Dirichlet distribution
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    a : (K,) shape array
        Fixed-point estimated parameters for Dirichlet distribution."""
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)

    # Start updating
    if maxiter is None:
        maxiter = MAXINT
    for i in range(maxiter):
        a1 = _ipsi(psi(a0.sum()) + logp)
        # Much faster convergence than with the more obvious condition
        # `norm(a1-a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        "Failed to converge after {} iterations, values are {}.".format(maxiter, a1)
    )


def _meanprecision(D, tol=1e-7, maxiter=None):
    """Mean/precision method for MLE of Dirichlet distribution
    Uses alternating estimations of mean and precision.
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    a : (K,) shape array
        Estimated parameters for Dirichlet distribution."""
    D = D + 1e-9
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)
    s0 = a0.sum()
    if s0 < 0:
        a0 = a0 / s0
        s0 = 1
    elif s0 == 0:
        a0 = ones(a0.shape) / len(a0)
        s0 = 1
    m0 = a0 / s0

    # Start updating
    if maxiter is None:
        maxiter = MAXINT
    for i in range(maxiter):
        a1 = _fit_s(D, a0, logp, tol=tol)
        s1 = sum(a1)
        a1 = _fit_m(D, a1, logp, tol=tol)
        m = a1 / s1
        # Much faster convergence than with the more obvious condition
        # `norm(a1-a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    return a1
    # raise NotConvergingError(
    #     f"Failed to converge after {maxiter} iterations, " f"values are {a1}."
    # )


def _fit_s(D, a0, logp, tol=1e-7, maxiter=1000):
    """Update parameters via MLE of precision with fixed mean
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a0 : (K,) shape array
        Current parameters for Dirichlet distribution
    logp : (K,) shape array
        Mean of log-transformed D across N observations
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 1000.
    Returns
    -------
    (K,) shape array
        Updated parameters for Dirichlet distribution."""
    s1 = a0.sum()
    m = a0 / s1
    mlogp = (m * logp).sum()
    for i in range(maxiter):
        s0 = s1
        g = psi(s1) - (m * psi(s1 * m)).sum() + mlogp
        h = _trigamma(s1) - ((m ** 2) * _trigamma(s1 * m)).sum()

        if g + s1 * h < 0:
            s1 = 1 / (1 / s0 + g / h / (s0 ** 2))
        if s1 <= 0:
            s1 = s0 * exp(-g / (s0 * h + g))  # Newton on log s
        if s1 <= 0:
            s1 = 1 / (1 / s0 + g / ((s0 ** 2) * h + 2 * s0 * g))  # Newton on 1/s
        if s1 <= 0:
            s1 = s0 - g / h  # Newton
        if s1 <= 0:
            raise NotConvergingError(f"Unable to update s from {s0}")

        a = s1 * m
        if abs(s1 - s0) < tol:
            return a

    return a
    # raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"s is {s1}")


def _fit_m(D, a0, logp, tol=1e-7, maxiter=1000):
    """Update parameters via MLE of mean with fixed precision s
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a0 : (K,) shape array
        Current parameters for Dirichlet distribution
    logp : (K,) shape array
        Mean of log-transformed D across N observations
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 1000.
    Returns
    -------
    (K,) shape array
        Updated parameters for Dirichlet distribution."""
    s = a0.sum()
    for i in range(maxiter):
        m = a0 / s
        a1 = _ipsi(logp + (m * (psi(a0) - logp)).sum())
        a1 = a1 / a1.sum() * s

        if norm(a1 - a0) < tol:
            return a1
        a0 = a1
    return a1
    # raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"s is {s}")


def _init_a(D):
    """Initial guess for Dirichlet alpha parameters given data D
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    Returns
    -------
    (K,) shape array
        Crude guess for parameters of Dirichlet distribution."""
    E = D.mean(axis=0)
    E2 = (D ** 2).mean(axis=0)
    return ((E[0] - E2[0]) / ((E2[0] - E[0] ** 2) + 1e-9 ) * E)


def _ipsi(y, tol=1.48e-9, maxiter=10):
    """Inverse of psi (digamma) using Newton's method. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).

    Parameters
    ----------
    y : (K,) shape array
        y-values of psi(x)
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 10.
    Returns
    -------
    (K,) shape array
        Approximate x for psi(x)."""
    y = asanyarray(y, dtype="float")
    x0 = np.piecewise(
        y,
        [y >= -2.22, y < -2.22],
        [(lambda x: exp(x) + 0.5), (lambda x: -1 / (x + euler))],
    )
    for i in range(maxiter):
        x1 = x0 - (psi(x0) - y) / _trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    return x1
    # raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"value is {x1}")


def _trigamma(x):
    return polygamma(1, x)


class MLE_Dirchlet(object):
    def __init__(self, trainQuads, num_r, k, timespan,
                 tol=1e-7, method="meanprecision", maxiter=10000):
        """
        num_r:int,  number of relations.
        k:int, statistics recent K historical snapshots.
        timespan:int, 24 for ICEWS, 1 for WIKI and YAGO
        tol : float, If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
        method : string, One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
        which to find MLE Dirichlet distribution. Default is ``'meanprecision'``, which is faster.
        maxiter : int, Maximum number of iterations to take calculations. Default is ``sys.maxint``.
        """
        self.num_r = num_r
        self.k = k
        self.timespan = timespan
        self.tol = tol
        self.method = method
        self.maxiter = maxiter
        self.entity_occ_times = self.get_entity_occ_times(trainQuads) # The number of occurrences of the entity at each time in the training set
        self.relations_observed_data = self.get_relations_observed_data(trainQuads)
        self.alphas = self.mle_dirchlet()

    def get_entity_occ_times(self, trainQuads):
        entity_occ_times = {}  # key -> entity, value -> dict [key: time, value: times]
        for quad in trainQuads:
            for entity in [quad[0], quad[2]]:
                if entity in entity_occ_times.keys():
                    if quad[3] in entity_occ_times[entity].keys():
                        entity_occ_times[entity][quad[3]] += 1
                    else:
                        entity_occ_times[entity][quad[3]] = 1
                else:
                    entity_occ_times[entity] = {quad[3]: 1, }
        return entity_occ_times

    def get_relations_observed_data(self, trainQuads):
        relations_observed_data = {}  # key: relation, value: list of observed data
        for quad in trainQuads:
            if quad[1] not in relations_observed_data.keys():
                relations_observed_data[quad[1]] = []
            observed = np.zeros([self.k+1])
            occ_times = self.entity_occ_times[quad[2]]
            for time in occ_times.keys():
                if time >= quad[3]:
                    continue
                observed[(quad[3] - time) // self.timespan] = occ_times[time]
            relations_observed_data[quad[1]].append(observed)

            # reversed_r = quad[1] + 1 + self.num_r
            # if reversed_r not in relations_observed_data.keys():
            #     relations_observed_data[reversed_r] = []
            # reversed_r_observed = np.zeros([self.k+1])
            # occ_times = self.entity_occ_times[quad[0]]
            # for time in occ_times.keys():
            #     if time >= quad[3]:
            #         continue
            #     reversed_r_observed[(quad[3] - time) // self.timespan] = occ_times[time]
            # relations_observed_data[reversed_r].append(reversed_r_observed)
        return relations_observed_data

    def mle_dirchlet(self):
        alphas = {}  # key: relation, value: alpha array
        with tqdm(total=len(self.relations_observed_data)) as bar:
            for r, observed in self.relations_observed_data.items():
                alphas[r] = mle(np.array(observed), tol=self.tol, method=self.method, maxiter=self.maxiter)
                bar.update(1)
        return alphas


class Dirichlet(object):
    def __init__(self, alphas, k):
        """alphas: Get from MLE_Dirchlet
        k: int, statistics recent K historical snapshots.
        """
        self.k = k
        self.distributions = {}
        for rel, alpha in alphas.items():
            self.distributions[rel] = dirichlet(alpha)

    def __call__(self, rel, dt):
        if dt >= self.k:
            return 0.0
        p_dt = self.distributions[rel].rvs(1)[0][dt]
        return p_dt