"""
Library of discrete time Epidemic models

copyright 2012 Flávio Codeco Coelho
License: GPL-v3
"""

__author__ = 'fccoelho'

import numpy as np
# from scipy.stats.distributions import poisson, nbinom
# from numpy import inf, nan, nan_to_num
# import sys
# import logging
from collections import OrderedDict
# import cython
from typing import Dict, List, Iterable, Any
# import numba
# from numba.experimental import jitclass
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
    'SEQIAHR': {
        'variables': {'S': 'Susceptible', 'E': 'Exposed', 'I': 'Infectious', 'A': 'Asymptomatic', 'H': 'Hospitalized',
                      'R': 'Removed', 'C': 'Cumulative hospitalizations', 'D': 'Cumulative deaths'},
        'parameters': {'chi': r'$\chi', 'phi': r'$\phi$', 'beta': r'$\beta$',
                       'rho': r'$\rho$', 'delta': r'$\delta$', 'alpha': r'$\alpha$', 'mu': r'$\mu$',
                       'p': '$p$', 'q': '$q$', 'r': '$r$'
                       }

        }
}


class DiscreteModel(BaseModel):
    """
    Exposes a library of discrete time population models
    """

    def __init__(self):
        """
        Difference equation based model (discrete time)
        """
        super().__init__()

    def run(self, *args):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        res = self.run(*args)
        self.traces.update(res)
        # return res


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

    def model(self, inits: list, trange: list, totpop: int, params: dict) -> dict:
        """
        Flu model with classes S,E,I subclinical, I mild, I medium, I serious, deaths
        """
        S1 = np.zeros(trange[1] - trange[0])
        E1 = np.zeros(trange[1] - trange[0])
        Is1 = np.zeros(trange[1] - trange[0])
        Ic1 = np.zeros(trange[1] - trange[0])
        Ig1 = np.zeros(trange[1] - trange[0])
        S2 = np.zeros(trange[1] - trange[0])
        E2 = np.zeros(trange[1] - trange[0])
        Is2 = np.zeros(trange[1] - trange[0])
        Ic2 = np.zeros(trange[1] - trange[0])
        Ig2 = np.zeros(trange[1] - trange[0])
        S3 = np.zeros(trange[1] - trange[0])
        E3 = np.zeros(trange[1] - trange[0])
        Is3 = np.zeros(trange[1] - trange[0])
        Ic3 = np.zeros(trange[1] - trange[0])
        Ig3 = np.zeros(trange[1] - trange[0])
        S4 = np.zeros(trange[1] - trange[0])
        E4 = np.zeros(trange[1] - trange[0])
        Is4 = np.zeros(trange[1] - trange[0])
        Ic4 = np.zeros(trange[1] - trange[0])
        Ig4 = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

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

    def model(self, inits: list, trange: list, totpop: int, params: dict) -> dict:
        """
        calculates the model SIS, and return its values (no demographics)
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        :param trange:
        :param params:
        :param inits: tuple with initial conditions
        :param simstep: step of the simulation
        :param totpop: total population
        :return:
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)
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
        self.parameters = {'beta': r'\beta', 'gamma': r'\gamma'}
        self.run = self.model

    def model(self, inits: list, trange: list, totpop: int, params: dict) -> dict:
        """
        calculates the model SIR, and return its values (no demographics)
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

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
        self.parameters = {'b': 'b', 'beta': r'\beta', 'e': 'e', 'r': 'r'}
        self.run = self.model

    def model(self, inits, trange, totpop, params):
        """
        Defines the model SEIS:
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        E: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

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
        self.parameters = {'b': 'b', 'beta': r'\beta', 'e': 'e', 'r': 'r', 'alpha': r'\alpha'}
        self.run = self.model

    def model(self, inits, trange, totpop, params):
        """
        Defines the model SEIR:
        - inits = (E,I,S)
        - par = (Beta, alpha, E,r,delta,B,w,p) see docs.
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        E: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

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


class SIpRpS(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SIpRpS'
        self.state_variables = {'I': 'Infectious', "S": 'Susceptible', 'R': 'Removed'}
        self.parameters = {'b': 'b', 'beta': r'$\beta$', 'e': 'e', 'r': 'r', 'delta': r'$\delta$'}
        self.run = self.model

    def model(self, inits, trange, totpop, params):
        """
        calculates the model SIpRpS, and return its values (no demographics)
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], I[0], R[0] = inits
        N = totpop

        beta = params['beta'];
        r = params['r'];
        delta = params['delta'];
        b = params['b'];

        # Model
        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * (I[i] / N)  # Number of new cases
            I[i + 1] = (1 - r) * I[i] + Lpos
            S[i + 1] = S[i] + b - Lpos + (1 - delta) * r * I[i]
            R[i + 1] = N - (S[i + 1] + I[i + 1]) + delta * r * I[i]

        return {'time': tspan, 'S': S, 'I': I, 'R': R}


class SEIpRpS(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SEIpRpS'
        self.state_variables = {'I': 'Infectious', "S": 'Susceptible', 'E': "Exposed", 'R': 'Removed'}
        self.parameters = {'b': 'b', 'beta': r'$\beta$', 'e': 'e', 'r': 'r', 'delta': r'$\delta$'}
        self.run = self.model

    def model(self, inits, trange, totpop, params):
        """
        Defines the model SEIpRpS:
        - inits = (E,I,S)
        - theta = infectious individuals from neighbor sites
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        E: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], E[0], I[0], R[0] = inits
        N = totpop

        beta = params['beta'];
        e = params['e'];
        r = params['r'];
        delta = params['delta'];
        b = params['b'];

        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * (I[i] / N)  # Number of new cases

            E[i + 1] = (1 - e) * E[i] + Lpos
            I[i + 1] = e * E[i] + (1 - r) * I[i]
            S[i + 1] = S[i] + b - Lpos + (1 - delta) * r * I[i]
            R[i + 1] = N - (S[i + 1] + E[i + 1] + I[i + 1]) + delta * r * I[i]

        return {'time': tspan, 'S': S, 'I': I, 'E': E, 'R': R}


class SIpR(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SIpR'
        self.state_variables = {'I': 'Infectious', "S": 'Susceptible', 'R': 'Removed'}
        self.parameters = {'b': 'b', 'beta': r'$\beta$', 'r': 'r', 'p': 'p'}
        self.run = self.model

    def model(self, inits, trange, totpop, params):
        """
        calculates the model SIpR, and return its values (no demographics)
        - inits = (S,I,R)
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], I[0], R[0] = inits
        N = totpop

        beta = params['beta']
        r = params['r']
        b = params['b']
        p = params['p']

        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * (I[i] / N)  # Number of new cases
            Lpos2 = p * float(beta) * R[i] * (I[i] / N)  # number of secondary Infections

            # Model
            I[i + 1] = (1 - r) * I[i] + Lpos + Lpos2
            S[i + 1] = S[i] + b - Lpos
            R[i + 1] = N - (S[i + 1] + I[i + 1]) - Lpos2

        return {'time': tspan, 'S': S, 'I': I, 'R': R}


class SEIpR(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SEIpR'
        self.state_variables = {'I': 'Infectious', "S": 'Susceptible', 'E': "Exposed", 'R': 'Removed'}
        self.parameters = {'b': 'b', 'beta': r'$\beta$', 'e': 'e', 'r': 'r', 'alpha': r'$\alpha$', 'p': 'p'}
        self.run = self.model

    def model(self, inits, trange, totpop, params):
        """
        calculates the model SEIpR, and return its values (no demographics)
        - inits = (S,E,I,R)
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        E: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], E[0], I[0], R[0] = inits
        N = totpop

        beta = params['beta']
        e = params['e']
        r = params['r']
        b = params['b']
        p = params['p']
        # print(tspan)
        for i in tspan[:-1]:
            # print(i)
            Lpos = float(beta) * S[i] * (I[i] / N)  # Number of new cases
            Lpos2 = p * float(beta) * R[i] * (I[i] / N)  # secondary infections

            # Model
            E[i + 1] = (1 - e) * E[i] + Lpos + Lpos2
            I[i + 1] = e * E[i] + (1 - r) * I[i]
            S[i + 1] = S[i] + b - Lpos
            R[i + 1] = N - (S[i + 1] + I[i + 1]) - Lpos2

        return {'time': tspan, 'S': S, 'I': I, 'E': E, 'R': R}

# from numba.types import unicode_type, pyobject
# spec = [
#     ('model_type', unicode_type),
#     ('state_variables', pyobject),
#     ('parameters', pyobject),
#     ('run', pyobject)
# ]
#
# @jitclass(spec)
class SIRS(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'SIRS'
        self.state_variables = {'R': 'Removed', 'I': 'Infectious', 'S': 'Susceptible'}
        self.parameters = {'beta': r'$\beta$', 'b': 'b', 'w': 'w'}
        self.run = self.model


    # @numba.jit
    def model(self, inits: List, trange: List, totpop: int, params: Dict) -> Dict:
        """
        calculates the model SIRS, and return its values (no demographics)
        :param inits: (E,I,S)
        :param trange:
        :param totpop:
        :param params:
        :return:
        """
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], I[0], R[0] = inits
        N = totpop

        beta = params['beta'];
        r = params['r'];
        b = params['b'];
        w = params['w'];

        for i in tspan[:-1]:
            Lpos = float(beta) * S[i] * (I[i] / N)  # Number of new cases

            # Model
            I[i + 1] = (1 - r) * I[i] + Lpos
            S[i + 1] = S[i] + b - Lpos + w * R[i]
            R[i + 1] = N - (S[i + 1] + I[i + 1]) - w * R[i]

        return {'time': tspan, 'S': S, 'I': I, 'R': R}


class SEQIAHR(DiscreteModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {'S': 'Susceptible', 'E': 'Exposed', 'I': 'Infectious', 'A': 'Asymptomatic', 'H': 'Hospitalized',
             'R': 'Removed', 'C': 'Cumulative hospitalizations', 'D': 'Cumulative deaths'})
        self.parameters = OrderedDict({'chi': r'$\chi', 'phi': r'$\phi$', 'beta': r'$\beta$',
                                       'rho': r'$\rho$', 'delta': r'$\delta$', 'gamma': r'$\gamma$',
                                       'alpha': r'$\alpha$', 'mu': r'$\mu$',
                                       'p': '$p$', 'q': '$q$', 'r': '$r$'
                                       })
        self.model_type = 'SEQIAHR'

        self.run = self.model

    def model(self, inits, trange, totpop, params) -> dict:
        S: np.ndarray = np.zeros(trange[1] - trange[0])
        E: np.ndarray = np.zeros(trange[1] - trange[0])
        I: np.ndarray = np.zeros(trange[1] - trange[0])
        A: np.ndarray = np.zeros(trange[1] - trange[0])
        H: np.ndarray = np.zeros(trange[1] - trange[0])
        R: np.ndarray = np.zeros(trange[1] - trange[0])
        C: np.ndarray = np.zeros(trange[1] - trange[0])
        D: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], E[0], I[0], A[0], H[0], R[0], C[0], D[0] = inits

        N = totpop
        chi, phi, beta, rho, delta, gamma, alpha, mu, p, q, r = params.values()

        for i in tspan[:-1]:
            # Turns on Quarantine on day q and off on day q+r
            chi_t = chi * ((1 + np.tanh(i - q)) / 2) * ((1 - np.tanh(i - (q + r))) / 2)
            ##### Modeling the number of new cases (incidence function)
            Lpos = beta * ((1 - chi_t) * S[i]) * (I[i] + A[i])  # Number of new cases

            ##### Epidemiological model (SEQIAHR)
            S[i + 1] = S[i] - Lpos
            E[i + 1] = E[i] + Lpos - alpha * E[i]
            I[i + 1] = I[i] + (1 - p) * alpha * E[i] - delta * I[i] - phi* I[i]
            A[i + 1] = A[i] + p * alpha * E[i] - gamma * A[i]
            H[i + 1] = H[i] + phi *  I[i] - (rho + mu) * H[i]
            R[i + 1] = R[i] + delta * I[i] + rho * H[i] + gamma * A[i]
            C[i + 1] = C[i] + phi * delta * I[i] + (1 - p) * alpha * E[i]  # Cumulative cases Hospitalizations + I
            D[i + 1] = D[i] + mu * H[i]  # Cumulative deaths

        return {'time': tspan, 'S': S, 'E': E, 'I': I, 'A': A, 'H': H, 'R': R, 'C': C, 'D': D}
