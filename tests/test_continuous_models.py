__author__ = 'fccoelho'

import unittest
import pytest
from matplotlib import pyplot as P
# import pyximport; pyximport.install(pyimport=True)

from epimodels.continuous.models import *


def test_SIR():
    model = SIR()
    model([1000, 1, 0], [0, 500], 1001, {'beta': .2, 'gamma': .1})
    assert len(model.traces) == 4
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()

def test_SIR_with_t_eval():
    model = SIR()
    model([1000, 1, 0], [0, 500], 1001, {'beta': .2, 'gamma': .1}, t_eval=range(0, 500))
    assert len(model.traces['S']) == 500
    # assert len(model.traces['time']) == 50

def test_SIS():
    model = SIS()
    model([1000, 1], [0, 50], 1001, {'beta': 2, 'gamma': .1})
    assert len(model.traces) == 3
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()


def test_SIRS():
    model = SIRS()
    model([1000, 1, 0], [0, 50], 1001, {'beta': 5, 'gamma': 1.9, 'xi': 0.05})
    assert len(model.traces) == 4
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()


def test_SEIR():
    model = SEIR()
    model([1000, 0, 1, 0], [0, 50], 1001, {'beta': 5, 'gamma': 1.9, 'epsilon': 0.1})
    # print(model.traces)
    assert len(model.traces) == 5  # state variables plus time
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()


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
    P.show()

# def test_SIS_with_cache():
#     model = SIS()
#     model([1000, 1], [0, 50], 1001, {'beta': 2, 'gamma': .1})
#     tr1 = model.traces
#     model([1000, 1], [0, 50], 1001, {'beta': 2, 'gamma': .1})
#     tr2 = model.traces
#     assert (tr1 == tr2)
