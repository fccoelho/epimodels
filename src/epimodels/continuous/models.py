u"""
Created on 09/11/18
by fccoelho
license: GPL V3 or Later
"""
import numpy as np
from scipy.integrate import solve_ivp
from epimodels import BaseModel
import logging



class ContinuousModel(BaseModel):
    """
    Exposes a library of continuous time population models
    """
    def __init__(self, model_type: str):
        """
        defines which models a given site will use
        and set variable names accordingly.
        :param parallel: Boolean for parallel execution
        :param model_type: string identifying the model type
        """
        super().__init__()
        try:
            assert model_type in model_types
            self.model_type = model_type
        except AssertionError:
            logging.Error('Invalid model type: {}'.format(model_type))

        self.state_variables = model_types[model_type]['variables']
        self.parameters = model_types[model_type]['parameters']


    def __call__(self, inits: list, trange: list, totpop: float, params: dict, method: str='RK45', **kwargs):
        self.method = method
        sol = self.run(inits, trange, totpop, params, **kwargs)
        res = {'time': sol.t, 'S': sol.y[0,:], 'I':sol.y[1,:], 'R':sol.y[2,:]}
        self.traces.update(res)

    def run(self, inits, trange, totpop, params, **kwargs):
        model = model_types[self.model_type]['function']
        sol = solve_ivp(lambda t,y: model(t,y, params), trange, inits, self.method)
        return sol


def SIR(t: float, y: list, params) -> list:
    """
    SIR Model.
    :param t:
    :param y:
    :param params:
    :return:
    """
    S, I, R = y
    beta, gamma = params['beta'], params['gamma']
    return [
        -beta*S*I,
        beta*S*I - gamma*I,
        gamma*I
    ]


model_types = {
    'SIR': {'variables': {'R': 'Removed', 'I': 'Infectious', 'S': 'Susceptible'},
            'parameters': {'beta': r'\beta', 'gamma': r'\gamma'},
            'function': SIR
            },
    'SIS': {'variables': {'I': 'Infectious', "S": 'Susceptible'},
            'parameters': {'beta': r'\beta', 'gamma': r'\gamma'}
            },
    'SEIS': {'variables': {'I': 'Infectious', "S": 'Susceptible', 'E': 'Exposed'},
             'parameters': {'b': 'b', 'beta': r'\beta', 'e': 'e', 'r': 'r'}
             },
    'SEIR': {'variables': {'I': 'Infectious', "S": 'Susceptible', 'E': 'Exposed', 'R': 'Removed'},
             'parameters': {'b': 'b', 'beta': r'\beta', 'e': 'e', 'r': 'r', 'alpha': r'\alpha'}
             },
}
