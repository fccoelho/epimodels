__author__ = 'fccoelho'

import unittest
import pytest
from matplotlib import pyplot as P
# import pyximport; pyximport.install(pyimport=True)

from epimodels.continuous.models import *


def test_SIR():
    model = SIR()
    model([1000, 1, 0], [0, 50], 1001, {'beta': 2, 'gamma': .1})
    assert len(model.traces) == 4
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()


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
    model([1000, 0, 1, 0, 0, 0, 0], [0, 50], 1001, {'chi': .3, 'phi': .01, 'beta': .5,
                                                    'rho': 1, 'delta': .1, 'alpha': 2,
                                                    'p': .75, 'q': 30
                                                    })
    # print(model.traces)
    assert len(model.traces) == 8  # state variables plus time
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
