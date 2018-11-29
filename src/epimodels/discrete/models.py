"""
Library of discrete time Epidemic models

copyright 2012 Flávio Codeco Coelho
License: GPL-v3
"""

__author__ = 'fccoelho'

import numpy as np
from scipy.stats.distributions import poisson, nbinom
from numpy import inf, nan, nan_to_num
import sys
import logging
import cython
from epimodels import BaseModel

model_types = {
    'SIR': {'variables': {'R': 'Removed', 'I': 'Infectious', 'S': 'Susceptible'},
            'parameters': {'beta': r'\beta', 'gamma': r'\gamma'}
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
    'SIpRpS': ['Exposed', 'Infectious', 'Susceptible'],
    'SEIpRpS': ['Exposed', 'Infectious', 'Susceptible'],
    'SEIpR': ['Exposed', 'Infectious', 'Susceptible'],
    'SIpR': ['Exposed', 'Infectious', 'Susceptible'],
    'SIRS': ['Exposed', 'Infectious', 'Susceptible'],
    'Custom': ['Exposed', 'Infectious', 'Susceptible'],
    'Influenza': {'variables': {'S1': 'Susc_age1', 'E1': 'Incub_age1', 'Is1': 'Subc_age1', 'Ic1': 'Sympt_age1',
                                'Ig1': 'Comp_age1',
                                'S2': 'Susc_age2', 'E2': 'Incub_age2', 'Is2': 'Subc_age2', 'Ic2': 'Sympt_age2',
                                'Ig2': 'Comp_age2',
                                'S3': 'Susc_age3', 'E3': 'Incub_age3', 'Is3': 'Subc_age3', 'Ic3': 'Sympt_age3',
                                'Ig3': 'Comp_age3',
                                'S4': 'Susc_age4', 'E4': 'Incub_age4', 'Is4': 'Subc_age4', 'Ic4': 'Sympt_age4',
                                'Ig4': 'Comp_age4'},
                  'parameters': {
                      'beta': r'\beta', 'r': 'r', 'e': 'e', 'c': 'c', 'g': 'g', 'd': 'd',
                      'pc1': r'pc_1', 'pc2': r'pc_2', 'pc3': r'pc_3', 'pc4': r'pc_4',
                      'pp1': r'pp_1', 'pp2': r'pp_2', 'pp3': r'pp_3', 'pp4': r'pp_4', 'b': 'b'
                  }
                  },
}


class DiscreteModel(BaseModel):
    """
    Exposes a library of discrete time population models
    """

    def __init__(self):
        """
        defines which models a given site will use
        and set variable names accordingly.
        :param parallel: Boolean for parallel execution
        :param model_type: string identifying the model type
        """
        super().__init__()
        # try:
        #     assert model_type in model_types
        #     self.model_type = model_type
        # except AssertionError:
        #     logging.Error('Invalid model type: {}'.format(model_type))
        # self.run = selectModel(model_type)
        # self.state_variables = model_types[model_type]['variables']
        # self.parameters = model_types[model_type]['parameters']

    def run(self, *args):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # args = self.get_args_from_redis()
        res = self.run(*args)
        self.traces.update(res)
        # return res


@cython.locals(Type='bytes')
def selectModel(Type):
    """
    Sets the model engine
    """
    if Type == 'SIR':
        return stepSIR
    elif Type == 'SIS':
        return stepSIS
    elif Type == 'SEIS':
        return stepSEIS
    elif Type == 'SEIR':
        return stepSEIR
    elif Type == 'SIpRpS':
        return stepSIpRpS
    elif Type == 'SEIpRpS':
        return stepSEIpRpS
    elif Type == 'SIpR':
        return stepSIpR
    elif Type == 'SEIpR':
        return stepSEIpR
    elif Type == 'SIRS':
        return stepSIRS
    elif Type == 'Influenza':
        return None
    elif Type == 'Custom':
        # adds the user model as a method of instance self
        try:
            # TODO: move this import to the graph level
            import CustomModel

            return CustomModel.Model
        except ImportError:
            print("You have to Create a CustomModel.py file before you can select\nthe Custom model type")
    else:
        sys.exit('Model type specified in .epg file is invalid')


class Influenza(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'Influenza'
        self.state_variables = {'S1': 'Susc_age1', 'E1': 'Incub_age1', 'Is1': 'Subc_age1', 'Ic1': 'Sympt_age1',
                                'Ig1': 'Comp_age1',
                                'S2': 'Susc_age2', 'E2': 'Incub_age2', 'Is2': 'Subc_age2', 'Ic2': 'Sympt_age2',
                                'Ig2': 'Comp_age2',
                                'S3': 'Susc_age3', 'E3': 'Incub_age3', 'Is3': 'Subc_age3', 'Ic3': 'Sympt_age3',
                                'Ig3': 'Comp_age3',
                                'S4': 'Susc_age4', 'E4': 'Incub_age4', 'Is4': 'Subc_age4', 'Ic4': 'Sympt_age4',
                                'Ig4': 'Comp_age4'}
        self.parameters = {
            'beta': r'\beta', 'r': 'r', 'e': 'e', 'c': 'c', 'g': 'g', 'd': 'd',
            'pc1': r'pc_1', 'pc2': r'pc_2', 'pc3': r'pc_3', 'pc4': r'pc_4',
            'pp1': r'pp_1', 'pp2': r'pp_2', 'pp3': r'pp_3', 'pp4': r'pp_4', 'b': 'b'
        }
        self.run = self.model

    def model(self, inits: list, timesteps: int, totpop: int, params: dict) -> dict:
        """
        Flu model with classes S,E,I subclinical, I mild, I medium, I serious, deaths
        """
        S1 = np.zeros(timesteps)
        E1 = np.zeros(timesteps)
        Is1 = np.zeros(timesteps)
        Ic1 = np.zeros(timesteps)
        Ig1 = np.zeros(timesteps)
        S2 = np.zeros(timesteps)
        E2 = np.zeros(timesteps)
        Is2 = np.zeros(timesteps)
        Ic2 = np.zeros(timesteps)
        Ig2 = np.zeros(timesteps)
        S3 = np.zeros(timesteps)
        E3 = np.zeros(timesteps)
        Is3 = np.zeros(timesteps)
        Ic3 = np.zeros(timesteps)
        Ig3 = np.zeros(timesteps)
        S4 = np.zeros(timesteps)
        E4 = np.zeros(timesteps)
        Is4 = np.zeros(timesteps)
        Ic4 = np.zeros(timesteps)
        Ig4 = np.zeros(timesteps)
        tspan = np.arange(timesteps)

        S1[0], E1[0], Is1[0], Ic1[0], Ig1[0], S2[0], E2[0], Is2[0], Ic2[0], Ig2[0], S3[0], E3[0], Is3[0], Ic3[0], Ig3[
            0], S4[0], E4[0], Is4[0], Ic4[0], Ig4[0] = inits
        N = totpop

        # for k, v in params.items():
        #     exec ('%s = %s' % (k, v))
        beta = params['beta']  # Transmission
        r = params['r']  # recovery rate
        e = params['e']  # incubation rate
        c = params['c']  #
        g = params['g']  #
        d = params['d']
        pc1 = params['pc1']
        pc2 = params['pc2']
        pc3 = params['pc3']
        pc4 = params['pc4']
        pp1 = params['pp1']
        pp2 = params['pp2']
        pp3 = params['pp3']
        pp4 = params['pp4']
        b = params['b']  # birth rate
        for i in tspan[:-1]:
            # Vacination event

            # if 'vaccineNow' in params:  # TODO: add to params when creating model
            #     vaccineNow = params['vaccineNow']
            #     vaccov = params['vaccov']
            #     S1 -= vaccov * S1
            #     S2 -= vaccov * S2
            #     S3 -= vaccov * S3
            #     S4 -= vaccov * S4

            # New cases by age class
            # beta=eval(values[2])

            Infectantes = Ig1[i] + Ig2[i] + Ig3[i] + Ig4[i] + Ic1[i] + Ic2[i] + Ic3[i] + Ic4[i] + 0.5 * (
                        Is1[i] + Is2[i] + Is3[i] + Is4[i])
            L1pos = float(beta) * S1[i] * Infectantes / N
            L2pos = float(beta) * S2[i] * Infectantes / N
            L3pos = float(beta) * S3[i] * Infectantes / N
            L4pos = float(beta) * S4[i] * Infectantes / N

            ######################
            Lpos = L1pos + L2pos + L3pos + L4pos
            # Model
            # 0-2 years old
            E1[i + 1] = L1pos + (1 - e) * E1[i]
            Is1[i + 1] = (1 - (pc1 * c + (1 - pc1) * r)) * Is1[i] + e * E1[i]
            Ic1[i + 1] = (1 - (pp1 * g + (1 - pp1) * r)) * Ic1[i] + pc1 * c * Is1[i]
            Ig1[i + 1] = (1 - d) * Ig1[i] + pp1 * g * Ic1[i]
            S1[i + 1] = b + S1[i] - L1pos
            # 3-14 years old
            E2[i + 1] = L2pos + (1 - e) * E2[i]
            Is2[i + 1] = (1 - (pc2 * c + (1 - pc2) * r)) * Is2[i] + e * E2[i]
            Ic2[i + 1] = (1 - (pp2 * g + (1 - pp2) * r)) * Ic2[i] + pc2 * c * Is2[i]
            Ig2[i + 1] = (1 - d) * Ig2[i] + pp2 * g * Ic2[i]
            S2[i + 1] = b + S2[i] - L2pos
            # 15-59 years old
            E3[i + 1] = L3pos + (1 - e) * E3[i]
            Is3[i + 1] = (1 - (pc3 * c + (1 - pc3) * r)) * Is3[i] + e * E3[i]
            Ic3[i + 1] = (1 - (pp3 * g + (1 - pp3) * r)) * Ic3[i] + pc3 * c * Is3[i]
            Ig3[i + 1] = (1 - d) * Ig3[i] + pp3 * g * Ic3[i]
            S3[i + 1] = b + S3[i] - L3pos
            # >60 years old
            E4[i + 1] = L4pos + (1 - e) * E4[i]
            Is4[i + 1] = (1 - (pc4 * c + (1 - pc4) * r)) * Is4[i] + e * E4[i]
            Ic4[i + 1] = (1 - (pp4 * g + (1 - pp4) * r)) * Ic4[i] + pc4 * c * Is4[i]
            Ig4[i + 1] = (1 - d) * Ig4[i] + pp4 * g * Ic4[i]
            S4[i + 1] = b + S4[i] - L4pos

        # Return variable values

        return {'S1': S1, 'E1': E1, 'Is1': Is1, 'Ic1': Ic1, 'Igl': Ig1, 'S2': S2, 'E2': E2, 'Is2': Is2,
                'Ic2': Ic2, 'Ig2': Ig2, 'S3': S3, 'E3': E3, 'Is3': Is3, 'Ic3': Ic3, 'Ig3': Ig3, 'S4': S4,
                'E4': E4, 'Is4': Is4, 'Ic4': Ic4, 'Ig4': Ig4, 'time': tspan}


class SIS(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SIS'
        self.state_variables = {"S": 'Susceptible', 'I': 'Infectious'}
        self.parameters = {'beta': r'\beta', 'gamma': r'\gamma'}
        self.run = self.model

    def model(self, inits: list, timesteps: int, totpop: int, params: dict) -> dict:
        """
        calculates the model SIS, and return its values (no demographics)
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        :param timesteps:
        :param params:
        :param inits: tuple with initial conditions
        :param simstep: step of the simulation
        :param totpop: total population
        :return:
        """
        S: np.ndarray = np.zeros(timesteps)
        I: np.ndarray = np.zeros(timesteps)
        tspan = np.arange(timesteps)
        E, I[0], S[0] = inits
        N = totpop

        beta = params['beta']
        gamma = params['gamma']

        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * I[i] / N
            # Model
            I[i + 1] = I[i] + Lpos - gamma * I[i]
            S[i + 1] = S[i] - Lpos + gamma * I[i]

        return {'S': S, 'I': I, 'time': tspan}


class SIR(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SIR'
        self.state_variables = {'R': 'Removed', 'I': 'Infectious', 'S': 'Susceptible'}
        self. parameters =  {'beta': r'\beta', 'gamma': r'\gamma'}
        self.run = self.model

    def model(self, inits: list, timesteps: int, totpop: int, params: dict) -> dict:
        """
        calculates the model SIR, and return its values (no demographics)
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(timesteps)
        I: np.ndarray = np.zeros(timesteps)
        R: np.ndarray = np.zeros(timesteps)
        tspan = np.arange(timesteps)

        S[0], I[0], R[0] = inits
        N = totpop
        beta = params['beta']
        gamma = params['gamma']

        # Model
        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * I[i] / N  # Number of new cases
            I[i + 1] = I[i] + Lpos - gamma * I[i]
            S[i + 1] = S[i] - Lpos
            R[i + 1] = N - (S[i + 1] + I[i + 1])

        return {'time': tspan, 'S': S, 'I': I, 'R': R}


class SEIS(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SEIS'
        self.state_variables = {'I': 'Infectious', "S": 'Susceptible', 'E': 'Exposed'}
        self. parameters = {'b': 'b', 'beta': r'\beta', 'e': 'e', 'r': 'r'}
        self.run = self.model

    def model(self, inits, timesteps, totpop, params):
        """
        Defines the model SEIS:
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(timesteps)
        E: np.ndarray = np.zeros(timesteps)
        I: np.ndarray = np.zeros(timesteps)
        tspan = np.arange(timesteps)

        S[0], E[0], I[0] = inits
        N = totpop

        beta = params['beta'];
        e = params['e'];
        r = params['r'];
        b = params['b'];

        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * I[i] / N  # Number of new cases

            # Model
            E[i + 1] = (1 - e) * E[i] + Lpos
            I[i + 1] = e * E[i] + (1 - r) * I[i]
            S[i + 1] = S[i] + b - Lpos + r * I[i]

        return {'time': tspan, 'S': S, 'I': I, 'E': E}

class SEIR(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SEIR'
        self.state_variables = {'I': 'Infectious', "S": 'Susceptible', 'E': 'Exposed', 'R': 'Removed'}
        self. parameters = {'b': 'b', 'beta': r'\beta', 'e': 'e', 'r': 'r', 'alpha': r'\alpha'}
        self.run = self.model

    def model(self, inits, timesteps, totpop, params):
        """
        Defines the model SEIR:
        - inits = (E,I,S)
        - par = (Beta, alpha, E,r,delta,B,w,p) see docs.
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(timesteps)
        E: np.ndarray = np.zeros(timesteps)
        I: np.ndarray = np.zeros(timesteps)
        R: np.ndarray = np.zeros(timesteps)
        tspan = np.arange(timesteps)

        S[0], E[0], I[0], R[0] = inits
        N = totpop
        beta = params['beta'];
        alpha = params['alpha'];
        e = params['e'];
        r = params['r'];
        b = params['b'];

        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * I[i] / N  # Number of new cases

            # Model
            E[i + 1] = (1 - e) * E[i] + Lpos
            I[i + 1] = e * E[i] + (1 - r) * I[i]
            S[i + 1] = S[i] + b - Lpos
            R[i + 1] = N - (S[i + 1] + E[i + 1] + I[i + 1])

        return {'time': tspan, 'S': S, 'I': I, 'E': E, 'R': R}


@cython.locals(inits='object', simstep='long', totpop='long', theta='double', npass='double',
               beta='double', alpha='double', E='double', I='double', S='double', N='long',
               r='double', b='double', w='double', Lpos='double', Lpos_esp='double', R='double',
               Ipos='double', Spos='double', Rpos='double')
def stepSIpRpS(inits, simstep, totpop, theta=0, npass=0, bi=None, params=None, values=None):
    """
    calculates the model SIpRpS, and return its values (no demographics)
    - inits = (E,I,S)
    - theta = infectious individuals from neighbor sites
    """
    if simstep == 1:  # get initial values
        E, I, S = (bi['e'], bi['i'], bi['s'])
    else:
        E, I, S = inits
    N = totpop
    beta = params['beta'];
    alpha = params['alpha'];
    # e = params['e'];
    r = params['r'];
    delta = params['delta'];
    b = params['b'];
    # w = params['w'];
    # p = params['p']
    Lpos = float(beta) * S * ((I + theta) / (N + npass)) ** alpha  # Number of new cases

    # Model
    Ipos = (1 - r) * I + Lpos
    Spos = S + b - Lpos + (1 - delta) * r * I
    Rpos = N - (Spos + Ipos) + delta * r * I

    # Migrating infecctious
    migInf = Ipos

    return [0, Ipos, Spos], Lpos, migInf


@cython.locals(inits='object', simstep='long', totpop='long', theta='double', npass='double',
               beta='double', alpha='double', E='double', I='double', S='double', N='long',
               r='double', b='double', w='double', Lpos='double', Lpos_esp='double', R='double',
               Ipos='double', Spos='double', Rpos='double')
def stepSEIpRpS(inits, simstep, totpop, theta=0, npass=0, bi=None, params=None, values=None):
    """
    Defines the model SEIpRpS:
    - inits = (E,I,S)
    - theta = infectious individuals from neighbor sites
    """
    if simstep == 1:  # get initial values
        E, I, S = (bi['e'], bi['i'], bi['s'])
    else:
        E, I, S = inits
    N = totpop
    beta = params['beta'];
    alpha = params['alpha'];
    e = params['e'];
    r = params['r'];
    delta = params['delta'];
    b = params['b'];
    # w = params['w'];
    # p = params['p']

    Lpos = float(beta) * S * ((I + theta) / (N + npass)) ** alpha  # Number of new cases

    Epos = (1 - e) * E + Lpos
    Ipos = e * E + (1 - r) * I
    Spos = S + b - Lpos + (1 - delta) * r * I
    Rpos = N - (Spos + Epos + Ipos) + delta * r * I

    # Migrating infecctious
    migInf = Ipos

    return [Epos, Ipos, Spos], Lpos, migInf


@cython.locals(inits='object', simstep='long', totpop='long', theta='double', npass='double',
               beta='double', alpha='double', E='double', I='double', S='double', N='long',
               r='double', b='double', w='double', Lpos='double', Lpos_esp='double', R='double',
               Ipos='double', Spos='double', Rpos='double')
def stepSIpR(inits, simstep, totpop, theta=0, npass=0, bi=None, params=None, values=None):
    """
    calculates the model SIpR, and return its values (no demographics)
    - inits = (E,I,S)
    - theta = infectious individuals from neighbor sites
    """
    if simstep == 1:  # get initial values
        E, I, S = (bi['e'], bi['i'], bi['s'])
    else:
        E, I, S = inits
    N = totpop
    R = N - E - I - S
    beta = params['beta'];
    alpha = params['alpha'];
    # e = params['e'];
    r = params['r'];
    # delta = params['delta'];
    b = params['b'];
    # w = params['w'];
    p = params['p']
    Lpos = float(beta) * S * ((I + theta) / (N + npass)) ** alpha  # Number of new cases
    Lpos2 = p * float(beta) * R * ((I + theta) / (N + npass)) ** alpha  # number of secondary Infections

    # Model
    Ipos = (1 - r) * I + Lpos + Lpos2
    Spos = S + b - Lpos
    Rpos = N - (Spos + Ipos) - Lpos2

    # Migrating infecctious
    migInf = Ipos

    return [0, Ipos, Spos], Lpos + Lpos2, migInf


@cython.locals(inits='object', simstep='long', totpop='long', theta='double', npass='double',
               beta='double', alpha='double', E='double', I='double', S='double', N='long',
               r='double', b='double', w='double', Lpos='double', Lpos_esp='double', R='double',
               Ipos='double', Spos='double', Rpos='double')
def stepSEIpR(inits, simstep, totpop, theta=0, npass=0, bi=None, params=None, values=None):
    """
    calculates the model SEIpR, and return its values (no demographics)
    - inits = (E,I,S)
    - theta = infectious individuals from neighbor sites
    """
    if simstep == 1:  # get initial values
        E, I, S = (bi['e'], bi['i'], bi['s'])
    else:
        E, I, S = inits
    N = totpop
    R = N - E - I - S
    beta = params['beta']
    alpha = params['alpha']
    e = params['e']
    r = params['r']
    # delta = params['delta']
    b = params['b']
    # w = params['w']
    p = params['p']

    Lpos = float(beta) * S * ((I + theta) / (N + npass)) ** alpha  # Number of new cases
    Lpos2 = p * float(beta) * R * ((I + theta) / (N + npass)) ** alpha  # secondary infections

    # Model
    Epos = (1 - e) * E + Lpos + Lpos2
    Ipos = e * E + (1 - r) * I
    Spos = S + b - Lpos
    Rpos = N - (Spos + Ipos) - Lpos2

    # Migrating infecctious
    migInf = Ipos

    return [0, Ipos, Spos], Lpos + Lpos2, migInf


@cython.locals(inits='object', simstep='long', totpop='long', theta='double', npass='double',
               beta='double', alpha='double', E='double', I='double', S='double', N='long',
               r='double', b='double', w='double', Lpos='double', Lpos_esp='double', R='double',
               Ipos='double', Spos='double', Rpos='double')
def stepSIRS(inits, simstep, totpop, theta=0, npass=0, bi=None, params=None, values=None):
    """
    calculates the model SIRS, and return its values (no demographics)
    - inits = (E,I,S)
    - theta = infectious individuals from neighbor sites
    """
    if simstep == 1:  # get initial values
        E, I, S = (bi['e'], bi['i'], bi['s'])
    else:
        E, I, S = inits
    N = totpop
    R = N - (E + I + S)
    beta = params['beta'];
    alpha = params['alpha'];
    # e = params['e'];
    r = params['r'];
    # delta = params['delta'];
    b = params['b'];
    w = params['w'];
    # p = params['p']
    Lpos = float(beta) * S * ((I + theta) / (N + npass)) ** alpha  # Number of new cases

    # Model
    Ipos = (1 - r) * I + Lpos
    Spos = S + b - Lpos + w * R
    Rpos = N - (Spos + Ipos) - w * R

    # Migrating infecctious
    migInf = Ipos

    return [0, Ipos, Spos], Lpos, migInf
