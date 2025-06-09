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
from typing import List, Dict, Optional, Union, Callable, Tuple, Any
import copy

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

    def __call__(self, inits: List[float], trange: List[float], totpop: float, params: dict, method: str = 'RK45', **kwargs):
        """
        Run the model
        :param inits: initial contitions
        :param trange: time range: [t0, tf]
        :param totpop: total population size
        :param params: dictionary of parameters
        :param method: integration method. default is 'RK45'
        :param kwargs: Additional parameters passed on to solve_ivp
        """
        self.method = method
        self.kwargs = kwargs
        self.param_values = OrderedDict(zip(self.parameters.keys(), params.values()))
        sol = self.run(inits, trange, totpop, params, **kwargs)
        res = {v: sol.y[s, :] for v, s in zip(self.state_variables.keys(), range(sol.y.shape[0]))}
        res['time'] = sol.t
        self.traces.update(res)

    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:
        raise NotImplementedError

    def __repr__(self):
        f = copy.deepcopy(self._model)
        desc = f"""
# Model: {self.model_type}

```mermaid
{self.diagram}
```
"""
        return desc

    @property
    def diagram(self) -> str:
        return "A[Define a diagram for this model]"

    @property
    def dimension(self) -> int:
        return len(self.state_variables)

    def run(self, inits, trange, totpop, params, **kwargs):
        # model = model_types[self.model_type]['function']
        params['N'] = totpop
        sol = solve_ivp(lambda t, y: self._model(t, y, params), trange, inits, self.method, **kwargs)
        return sol


class SIR(ContinuousModel):
    '''
    SIR Model
    '''
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'I': 'Infectious', 'R': 'Removed'})
        self.parameters = OrderedDict({'beta': r'$\beta$', 'gamma': r'$\gamma$'})
        self.model_type = 'SIR'

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
"""

    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:
        S, I, R = y
        beta, gamma, N = params['beta'], params['gamma'], params['N']
        return [
            -beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I
        ]


class SIR1D(ContinuousModel):
    """
        One dimensional SIR model
    """
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'R': 'Recovered'})
        self.parameters = {'R0': r'{\cal R}_0', 'gamma': r'\gamma', 'S0': r'S_0'}
        self.model_type = 'SIR1D'

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Recovered)
"""

    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:

        N = params['N']
        R = y[0]
        R0, gamma, S0 = params['R0'], params['gamma'], params['S0']
        return [
            gamma * (N - R - (S0 * np.exp(-R0 * R)))
        ]


class SIS(ContinuousModel):
    '''
    SIS Model.
    '''
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'I': 'Infectious'})
        self.parameters = {'beta': r'\beta', 'gamma': r'\gamma'}
        self.model_type = 'SIS'

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| S
"""

    # @lru_cache(1000)
    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:
        S, I = y
        beta, gamma, N = params['beta'], params['gamma'], params['N']
        return [
            -beta * S * I / N + gamma * I,
            beta * S * I / N - gamma * I,
        ]


class SIRS(ContinuousModel):
    '''
    SIRS Model
    '''
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible', 'I': 'Infectious', 'R': 'Removed'})
        self.parameters = OrderedDict({'beta': r'$\beta$', 'gamma': r'$\gamma$', 'xi': r'$\xi$'})
        self.model_type = 'SIRS'

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
R -->|$$\xi$$| S
"""

    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:
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

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| E(Exposed)
E -->|$$\epsilon$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
"""

    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:
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

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| E(Exposed)
E -->|"$$\alpha(1-p)$$"| I(Infectious)
E -->|$$\alpha p$$| A(Asymptomatic)
I -->|$$\phi$$| H(Hospitalized)
I -->|$$\delta$$| R(Removed)
A -->|$$\gamma$$| R
H -->|$$\rho$$| R
H -->|$$\mu$$| D(Deaths)
"""

    def _model(self, t: float, y: List[float], params: dict[str, float]) -> List[object]:
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


class Dengue4Strain(ContinuousModel):
    """
    Dengue 4 strain model
    """
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({'S': 'Susceptible',
                                            'I_1': 'Infectious 1',
                                            'I_2': 'Infectious 2',
                                            'I_3': 'Infectious 3',
                                            'I_4': 'Infectious 4',
                                            'R_1': 'Removed 1',
                                            'R_2': 'Removed 2',
                                            'R_3': 'Removed 3',
                                            'R_4': 'Removed 4',
                                            'I_12': 'Infectious 1 and 2',
                                            'I_13': 'Infectious 1 and 3',
                                            'I_14': 'Infectious 1 and 4',
                                            'I_21': 'Infectious 2 and 1',
                                            'I_23': 'Infectious 2 and 3',
                                            'I_24': 'Infectious 2 and 4',
                                            'I_31': 'Infectious 3 and 1',
                                            'I_32': 'Infectious 3 and 2',
                                            'I_34': 'Infectious 3 and 4',
                                            'I_41': 'Infectious 4 and 1',
                                            'I_42': 'Infectious 4 and 2',
                                            'I_43': 'Infectious 4 and 3',
                                            'R_12': 'Removed 1 and 2',
                                            'R_13': 'Removed 1 and 3',
                                            'R_14': 'Removed 1 and 4',
                                            'R_23': 'Removed 2 and 3',
                                            'R_24': 'Removed 2 and 4',
                                            'R_34': 'Removed 3 and 4',
                                            'I_231': 'Infectious 2 and 3 and 1',
                                            'I_241': 'Infectious 2 and 4 and 1',
                                            'I_341': 'Infectious 3 and 4 and 1',
                                            'I_132': 'Infectious 1 and 3 and 2',
                                            'I_142': 'Infectious 1 and 4 and 2',
                                            'I_342': 'Infectious 3 and 4 and 2',
                                            'I_123': 'Infectious 1 and 2 and 3',
                                            'I_143': 'Infectious 1 and 4 and 3',
                                            'I_243': 'Infectious 2 and 4 and 3',
                                            'I_124': 'Infectious 1 and 2 and 4',
                                            'I_134': 'Infectious 1 and 3 and 4',
                                            'I_234': 'Infectious 2 and 3 and 4',
                                            'R_123': 'Removed 1 and 2 and 3',
                                            'R_124': 'Removed 1 and 2 and 4',
                                            'R_134': 'Removed 1 and 3 and 4',
                                            'R_234': 'Removed 2 and 3 and 4',
                                            'I_1234': 'Infectious 1 and 2 and 3 and 4',
                                            'I_1243':  'Infectious 1 and 2 and 4 and 3',
                                            'I_1342': 'Infectious 1 and 3 and 4 and 2',
                                            'I_2341': 'Infectious 2 and 3 and 4 and 1',
                                            'R_1234': 'Removed 1 and 2 and 3 and 4'
                                            })
        self.parameters = OrderedDict({'beta': r'$\beta$', #  transmission rate
                                       'N': r'$N$', #  total population
                                       'delta': r'$\delta$', #  cross-immunity protection
                                       'mu': r'$\mu', #  mortality rate
                                       'sigma': r'$\sigma$', #  recovery rate
                                       'im': r'$i_m$', #  imported cases
                                       })
        self.model_type = 'Dengue4Strain'

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
    S(Susceptible) -->|$$\beta$$| I1(I 1)
    S -->|$$\beta$$| I2(I 2) 
    S -->|$$\beta$$| I3(I 3)
    S -->|$$\beta$$| I4(I 4)
    
    I1 -->|$$\sigma$$| R1(R 1)
    I2 -->|$$\sigma$$| R2(R 2)
    I3 -->|$$\sigma$$| R3(R 3)
    I4 -->|$$\sigma$$| R4(R 4)
    
    R1 -->|$$\delta$$| I12(I 1+2)
    R1 -->|$$\delta$$| I13(I 1+3)
    R1 -->|$$\delta$$| I14(I 1+4)
    
    R2 -->|$$\delta$$| I21(I 2+1)
    R2 -->|$$\delta$$| I23(I 2+3)
    R2 -->|$$\delta$$| I24(I 2+4)
    
    R3 -->|$$\delta$$| I31(I 3+1)
    R3 -->|$$\delta$$| I32(I 3+2)
    R3 -->|$$\delta$$| I34(I 3+4)
    
    R4 -->|$$\delta$$| I41(I 4+1)
    R4 -->|$$\delta$$| I42(I 4+2)
    R4 -->|$$\delta$$| I43(I 4+3)
    
    I12 -->|$$\sigma$$| R12(R 1+2)
    I13 -->|$$\sigma$$| R13(R 1+3)
    I14 -->|$$\sigma$$| R14(R 1+4)
    
    I21 -->|$$\sigma$$| R12(R 1+2)
    I23 -->|$$\sigma$$| R23(R 2+3)
    I24 -->|$$\sigma$$| R24(R 2+4)
    
    I31 -->|$$\sigma$$| R13(R 1+3)
    I32 -->|$$\sigma$$| R23(R 2+3)
    I34 -->|$$\sigma$$| R34(R 3+4)
    
    I41 -->|$$\sigma$$| R14(R 1+4)
    I42 -->|$$\sigma$$| R24(R 2+4)
    I43 -->|$$\sigma$$| R34(R 3+4)
    
    R12 -->|$$\delta$$| I123(I 1+2+3)
    R13 -->|$$\delta$$| I132(I 1+3+2)
    R14 -->|$$\delta$$| I142(I 1+4+2)
    
    R12 -->|$$\delta$$| I213(I 2+1+3)
    R23 -->|$$\delta$$| I231(I 2+3+1)
    R24 -->|$$\delta$$| I241(I 2+4+1)
     
    R24 -->|$$\delta$$| I243(I 2+4+3)
    R34 -->|$$\delta$$| I341(I 3+4+1)
    R34 -->|$$\delta$$| I342(I 3+4+2)
     
    R23 -->|$$\delta$$| I234(I 2+3+4)
    R14 -->|$$\delta$$| I143(I 1+4+3)
    R13 -->|$$\delta$$| I134(I 1+3+4)
    
    R12 -->|$$\delta$$| I124(I 1+2+4)
    
    I123 -->|$$\sigma$$| R123(R 1+2+3)
    I132 -->|$$\sigma$$| R123(R 1+2+3)
    I124 -->|$$\sigma$$| R124(R 1+2+4)
    
    I142 -->|$$\sigma$$| R124(R 1+2+4)
    I143 -->|$$\sigma$$| R134(R 1+3+4)
    I134 -->|$$\sigma$$| R134(R 1+3+4)
    
    I234 -->|$$\sigma$$| R234(R 2+3+4)
    I243 -->|$$\sigma$$| R234(R 2+3+4)
    I341 -->|$$\sigma$$| R134(R 1+3+4)
    
    I342 -->|$$\sigma$$| R234(R 2+3+4)
    I231 -->|$$\sigma$$| R123(R 1+2+3)
    I241 -->|$$\sigma$$| R124(R 1+2+4)
    
    R123 -->|$$\delta$$| I1234(I 1+2+3+4)
    R124 -->|$$\delta$$| I1243(I 1+2+4+3)
    R134 -->|$$\delta$$| I1342(I 1+3+4+2)
    R234 -->|$$\delta$$| I2341(I 2+3+4+1)
    
    I1234 -->|$$\sigma$$| R1234(R 1+2+3+4)
    I1243 -->|$$\sigma$$| R1234(R 1+2+3+4)
    I1342 -->|$$\sigma$$| R1234(R 1+2+3+4)
    I2341 -->|$$\sigma$$| R1234(R 1+2+3+4)
    
    classDef strain1 fill:#ffcccc,stroke:#ff0000
    classDef strain2 fill:#ccffcc,stroke:#00ff00
    classDef strain3 fill:#ccccff,stroke:#0000ff
    classDef strain4 fill:#ffccff,stroke:#ff00ff
    
    class I1,I21,I31,I41,I231,I241,I341,I2341 strain1
    class I2,I12,I32,I42,I132,I213,I342,I142,I1342 strain2
    class I3,I13,I23,I43,I123,I143,I243,I1243 strain3
    class I4,I14,I24,I34,I142,I124,I134,I342,I124,I134,I234,I1234 strain4;
"""

    def _model(self, t: float, y: List[float], params: Dict[str, Any]) -> List[float]:
        (S, I_1, I_2, I_3, I_4, R_1, R_2, R_3, R_4, I_12, I_13, I_14, I_21,
         I_23, I_24, I_31, I_32, I_34, I_41, I_42, I_43, R_12, R_13, R_14,
         R_23, R_24, R_34, I_231, I_241, I_341, I_132, I_142, I_342, I_123,
         I_143, I_243, I_124, I_134, I_234, R_123, R_124, R_134, R_234, I_1234, I_1243, I_1342, I_2341, R_1234) = y

        beta, N, delta, mu, sigma, im = params['beta'], params['N'], params['delta'], params['mu'], params['sigma'], params['im']
        m1 = lambda t: 1 if (im[0] < t < (im[0]+5)) else 0
        m2 = lambda t: 1 if (im[1] < t < (im[1]+5)) else 0
        m3 = lambda t: 1 if (im[2] < t < (im[2]+5)) else 0
        m4 = lambda t: 1 if (im[3] < t < (im[3]+5)) else 0
        return [
            -beta * S * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
                         I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342 + \
                         I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243 + \
                         I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) + mu * N - mu * S, # S
            m1(t) + beta * S * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_1 - mu * I_1, # I_1
            m2(t) + beta * S * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_2 - mu * I_2, # I_2
            m3(t) + beta * S * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_3 - mu * I_3, # I_3
            m4(t) + beta * S * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_4 - mu * I_4, # I_4
            sigma * I_1 - beta * delta * R_1 * \
            (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342 + \
             I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243 + \
             I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_1, # R_1
            sigma * I_2 - beta * delta * R_2 * \
            (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
             I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243 + \
             I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_2, # R_2
            sigma * I_3 - beta * delta * R_3 * \
            (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
             I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342 + \
             I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_3, # R_3
            sigma * I_4 - beta * delta * R_4 * \
            (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
             I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342 + \
             I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) - mu * R_4, # R_4
            beta * delta * R_1 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_12 - mu * I_12, # I_12
            beta * delta * R_1 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_13 - mu * I_13, # I_13
            beta * delta * R_1 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_14 - mu * I_14, # I_14
            beta * delta * R_2 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_21 - mu * I_21, # I_21
            beta * delta * R_2 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_23 - mu * I_23, # I_23
            beta * delta * R_2 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_24 - mu * I_24, # I_24
            beta * delta * R_3 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_31 - mu * I_31, # I_31
            beta * delta * R_3 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_32 - mu * I_32, # I_32
            beta * delta * R_3 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_34 - mu * I_34, # I_34
            beta * delta * R_4 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_41 - mu * I_41, # I_41
            beta * delta * R_4 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_42 - mu * I_42, # I_42
            beta * delta * R_4 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_43 - mu * I_43, # I_43
            sigma * (I_12 + I_21) - beta * delta * \
            R_12 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243 + \
                    I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_12, # R_12
            sigma * (I_13 + I_31) - beta * delta * \
            R_13 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342 + \
                    I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_13, # R_13
            sigma * (I_14 + I_41) - beta * delta * \
            R_14 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342 + \
                    I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) - mu * R_14, # R_14
            sigma * (I_23 + I_32) - beta * delta * \
            R_23 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
                    I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_23, # R_23
            sigma * (I_24 + I_42) - beta * delta * \
            R_24 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
                    I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) - mu * R_24, # R_24
            sigma * (I_34 + I_43) - beta * delta * \
            R_34 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341 + \
                    I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) - mu * R_34, # R_34
            beta * delta * R_23 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_231 - mu * I_231, # I_231
            beta * delta * R_24 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_241 - mu * I_241, # I_241
            beta * delta * R_34 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_341 - mu * I_341, # I_341
            beta * delta * R_13 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_132 - mu * I_132, # I_132
            beta * delta * R_14 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_142 - mu * I_142, # I_142
            beta * delta * R_34 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_342 - mu * I_342, # I_342
            beta * delta * R_12 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_123 - mu * I_123, # I_123
            beta * delta * R_14 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_143 - mu * I_143, # I_143
            beta * delta * R_24 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_243 - mu * I_243, # I_243
            beta * delta * R_12 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_124 - mu * I_124, # I_124
            beta * delta * R_13 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_134 - mu * I_134, # I_134
            beta * delta * R_23 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_234 - mu * I_234, # I_234
            sigma * (I_123 + I_132 + I_231) - beta * delta * \
            R_123 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) - mu * R_123, # R_123
            sigma * (I_124 + I_241 + I_142) - beta * delta * \
            R_124 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) - mu * R_124, # R_124
            sigma * (I_134 + I_341 + I_143) - beta * delta * \
            R_134 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) - mu * R_134, # R_134
            sigma * (I_234 + I_342 + I_243) - beta * delta * \
            R_234 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) - mu * R_234, # R_234
            beta * delta * R_123 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234) \
            - sigma * I_1234 - mu * I_1234, # I_1234
            beta * delta * R_124 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243) \
            - sigma * I_1243 - mu * I_1243, # I_1243
            beta * delta * R_134 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342) \
            - sigma * I_1342 - mu * I_1342, # I_1342
            beta * delta * R_234 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341) \
            - sigma * I_2341 - mu * I_2341, # I_2341
            sigma * (I_1234 + I_1243 + I_1342 + I_2341) - mu * R_1234, # R_1234
        ]
