u"""
Created on 09/11/18
by fccoelho
license: GPL V3 or Later
"""
import numpy as np
from scipy.integrate import solve_ivp
from epimodels import BaseModel
import logging
from collections import OrderedDict
from functools import lru_cache
import numba

logging.basicConfig(filename='epimodels.log', filemode='w', level=logging.DEBUG)


class ContinuousModel(BaseModel):
    """
    Exposes a library of continuous time population models
    """

    def __init__(self) -> None:
        """
        Base class for Continuous models
        :param parallel: Boolean for parallel execution
        :param model_type: string identifying the model type
        """
        super().__init__()
        # try:
        #     assert model_type in model_types
        #     self.model_type = model_type
        # except AssertionError:
        #     logging.Error('Invalid model type: {}'.format(model_type))

    def __call__(self, inits: list, trange: list, totpop: float, params: dict, method: str = 'RK45', **kwargs):
        self.method = method
        self.kwargs = kwargs
        sol = self.run(inits, trange, totpop, params, **kwargs)
        res = {v: sol.y[s, :] for v, s in zip(self.state_variables.keys(), range(sol.y.shape[0]))}
        res['time'] = sol.t
        self.traces.update(res)

    def _model(self, t: float, y: list, params: list):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return len(self.state_variables)

    def run(self, inits, trange, totpop, params, **kwargs):
        # model = model_types[self.model_type]['function']
        params['N'] = totpop
        sol = solve_ivp(lambda t, y: self._model(t, y, params), trange, inits, self.method, **kwargs)
        return sol


class SIR(ContinuousModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'I': 'Infectious', 'R': 'Removed'})
        self.parameters = OrderedDict({'beta': r'$\beta$', 'gamma': r'$\gamma$'})
        self.model_type = 'SIR'
    @numba.jit
    def _model(self, t: float, y: list, params: dict) -> list:
        """
        SIR Model.
        :param t: time step
        :param y: state of the model at time t
        :param params: parameter dictionary
        :return:
        """
        S, I, R = y
        beta, gamma, N = params['beta'], params['gamma'], params['N']
        return [
            -beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I
        ]


class SIS(ContinuousModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'I': 'Infectious'})
        self.parameters = {'beta': r'\beta', 'gamma': r'\gamma'}
        self.model_type = 'SIS'

    # @lru_cache(1000)
    @numba.jit
    def _model(self, t: float, y: list, params: dict) -> list:
        """
        SIS Model.
        :param t:
        :param y:
        :param params:
        :return:
        """
        S, I = y
        beta, gamma, N = params['beta'], params['gamma'], params['N']
        return [
            -beta * S * I / N + gamma * I,
            beta * S * I / N - gamma * I,
        ]


class SIRS(ContinuousModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'I': 'Infectious', 'R': 'Removed'})
        self.parameters = OrderedDict({'beta': r'$\beta$', 'gamma': r'$\gamma$', 'xi': r'$\xi$'})
        self.model_type = 'SIRS'

    @numba.jit
    def _model(self, t: float, y: list, params: dict) -> list:
        """
        SIR Model.
        :param t:
        :param y:
        :param params:
        :return:
        """
        S, I, R = y
        beta, gamma, xi, N = params['beta'], params['gamma'], params['xi'], params['N']
        return [
            -beta * S * I / N + xi * R,
            beta * S * I / N - gamma * I,
            gamma * I - xi * R
        ]


class SEIR(ContinuousModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'E': 'Exposed', 'I': 'Infectious', 'R': 'Removed'})
        self.parameters = OrderedDict({'beta': r'$\beta$', 'gamma': r'$\gamma$', 'epsilon': r'$\epsilon$'})
        self.model_type = 'SEIR'

    @numba.jit
    def _model(self, t: float, y: list, params: dict) -> list:
        S, E, I, R = y
        beta, gamma, epsilon, N = params['beta'], params['gamma'], params['epsilon'], params['N']
        return [
            -beta * S * I / N,
            beta * S * I / N - epsilon * E,
            epsilon * E - gamma * I,
            gamma * I
        ]


class SEQIAHR(ContinuousModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {'S': 'Susceptible', 'E': 'Exposed', 'I': 'Infectious', 'A': 'Asymptomatic', 'H': 'Hospitalized',
             'R': 'Removed', 'C': 'Cumulative hospitalizations', 'D': 'Cumulative deaths'})
        self.parameters = OrderedDict({'chi': r'$\chi$', 'phi': r'$\phi$', 'beta': r'$\beta$',
                                       'rho': r'$\rho$', 'delta': r'$\delta$', 'gamma': r'$\gamma$',
                                       'alpha': r'$\alpha$', 'mu': r'$\mu$',
                                       'p': '$p$', 'q': '$q$', 'r': '$r$'
                                       })
        self.model_type = 'SEQIAHR'

    @numba.jit
    def _model(self, t: float, y: list, params: dict) -> list:
        S, E, I, A, H, R, C, D = y
        chi, phi, beta, rho, delta, gamma, alpha, mu, p, q, r, N = params.values()
        lamb = beta * (I + A)
        # Turns on Quarantine on day q and off on day q+r
        chi *= ((1 + np.tanh(t - q)) / 2) * ((1 - np.tanh(t - (q + r))) / 2)
        return [
            -lamb * ((1 - chi) * S),  # dS/dt
            lamb * ((1 - chi) * S) - alpha * E,  # dE/dt
            (1 - p) * alpha * E - delta * I - phi * I,  # dI/dt
            p * alpha * E - gamma * A,
            phi * I - (rho + mu) * H,  # dH/dt
            delta * I + rho * H + gamma * A,  # dR/dt
            phi * I,  # (1-p)*alpha*E+ p*alpha*E # Hospit. acumuladas
            mu * H  # Morte acumuladas
        ]
