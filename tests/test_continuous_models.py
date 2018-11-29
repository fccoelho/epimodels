__author__ = 'fccoelho'

import unittest
import pytest
from matplotlib import pyplot as P
# import pyximport; pyximport.install(pyimport=True)

from epimodels.continuous.models import SIR, SIS, SIRS


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
