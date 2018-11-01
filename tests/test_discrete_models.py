__author__ = 'fccoelho'

import pytest
from epimodels.discrete.models import  Epimodel
import pyximport; pyximport.install(pyimport=True)
from matplotlib import pyplot as P


def test_SIS():
    model  = Epimodel('SIS')
    model([0, 1, 1000], 50, 1001, {'beta': 2, 'gamma': 1})
    assert len(model.traces) == 3
    assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()
    assert isinstance(model, Epimodel)

def test_SIR():
    model  = Epimodel('SIR')
    model([1000, 1, 0], 50, 1001, {'beta': 2, 'gamma': 1})
    assert len(model.traces) == 4
    assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()
    assert isinstance(model, Epimodel)



