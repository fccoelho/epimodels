__author__ = 'fccoelho'

import sys

sys.path.append('..')
from matplotlib import pyplot as P
import numpy as np
from epimodels.continuous import (SIS, SIR, SIR1D, SIRS, SEIR, SEQIAHR, Dengue4Strain)


def test_SIR():
    model = SIR()
    model([1000, 1, 0], [0, 500], 1001, {'beta': .2, 'gamma': .1})
    assert len(model.traces) == 4
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    # P.show()


def test_SIR_with_t_eval():
    model = SIR()
    model([1000, 1, 0], [0, 500], 1001, {'beta': .2, 'gamma': .1}, t_eval=range(0, 500))
    assert len(model.traces['S']) == 500
    # assert len(model.traces['time']) == 50


def test_SIR1D():
    model = SIR1D()
    model([0], [0, 500], 100, {'R0': 1.5, 'gamma': .1, 'S0': 98})
    # assert len(model.traces['R']) == 500
    assert len(model.traces) == 2
    model.plot_traces()
    # P.show()


def test_SIS():
    model = SIS()
    model([1000, 1], [0, 50], 1001, {'beta': 2, 'gamma': .1})
    assert len(model.traces) == 3
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    # P.show()


def test_SIRS():
    model = SIRS()
    model([1000, 1, 0], [0, 50], 1001, {'beta': 5, 'gamma': 1.9, 'xi': 0.05})
    assert len(model.traces) == 4
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    # P.show()


def test_SEIR():
    model = SEIR()
    model([1000, 0, 1, 0], [0, 50], 1001, {'beta': 5, 'gamma': 1.9, 'epsilon': 0.1})
    # print(model.traces)
    assert len(model.traces) == 5  # state variables plus time
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    # P.show()


def test_SEQIAHR():
    model = SEQIAHR()
    model([.99, 0, 1e-6, 0, 0, 0, 0, 0], [0, 300], 1, {'chi': .7, 'phi': .01, 'beta': .5,
                                                       'rho': .05, 'delta': .1, 'gamma': .1,
                                                       'alpha': .33, 'mu': .03,
                                                       'p': .75, 'q': 50, 'r': 40
                                                       })
    # print(model.traces)
    assert len(model.traces) == 9  # state variables plus time
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    # P.show()


def test_Dengue4Strain():
    model = Dengue4Strain()
    inits = [48000, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pars = {
        'beta': 500 / (50000 * 52),  # 500 cases per year
        'N': 50000,
        'delta': 0.2,  # Cross-immunity protection
        'mu': 1 / (1 * 52),  # Mortality rate
        'sigma': 1 / 1.5,  # recovery rate
        'im': 500
    }
    model(inits, [0, 200], 50000, pars)
    # model.plot_traces()
    pts = len(model.traces['time'])
    Ia1 = np.zeros(pts) # All infectious for strain 1
    Ia2 = np.zeros(pts) # All infectious for strain 2
    Ia3 = np.zeros(pts) # All infectious for strain 3
    Ia4 = np.zeros(pts) # All infectious for strain 4
    for v,tr in model.traces.items():
        if not v.startswith('I_'):
            continue
        if v.endswith('1'):
            Ia1 += tr
        elif v.endswith('2'):
            Ia2 += tr
        elif v.endswith('3'):
            Ia3 += tr
        elif v.endswith('4'):
            Ia4 += tr
    P.plot(model.traces['time'], Ia1, label='Infectious strain 1')
    P.plot(model.traces['time'], Ia2, label='Infectious strain 2')
    P.plot(model.traces['time'], Ia3, label='Infectious strain 3')
    P.plot(model.traces['time'], Ia4, label='Infectious strain 4')
    P.grid()
    P.legend(loc=0)


    P.show()

# def test_SIS_with_cache():
#     model = SIS()
#     model([1000, 1], [0, 50], 1001, {'beta': 2, 'gamma': .1})
#     tr1 = model.traces
#     model([1000, 1], [0, 50], 1001, {'beta': 2, 'gamma': .1})
#     tr2 = model.traces
#     assert (tr1 == tr2)
