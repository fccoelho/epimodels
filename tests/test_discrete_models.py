__author__ = 'fccoelho'

import pytest
from epimodels.discrete.models import Epimodel
import pyximport; pyximport.install(pyimport=True)
from matplotlib import pyplot as P


def test_SIS():
    modelsis  = Epimodel('SIS')
    modelsis([0, 1, 1000], 50, 1001, {'beta': 2, 'gamma': 1})
    assert len(modelsis.traces) == 3
    assert len(modelsis.traces['time']) == 50
    modelsis.plot_traces()
    P.show()
    assert isinstance(modelsis, Epimodel)


def test_SIR():
    modelsir  = Epimodel('SIR')
    modelsir([1000, 1, 0], 50, 1001, {'beta': 2, 'gamma': 1})
    assert len(modelsir.traces) == 4
    assert len(modelsir.traces['time']) == 50
    modelsir.plot_traces()
    P.show()
    assert isinstance(modelsir, Epimodel)

def test_FLU():
    modelflu = Epimodel('Influenza')
    modelflu([250,1,0,0,0,250,1,0,0,0,250,1,0,0,0,250,1,0,0,0], 50, 1004, {'beta': 2.0,
                                                                        'r': 0.25,
                                                                        'e': 0.5,
                                                                        'c': 0.5,
                                                                        'g': 1/3,
                                                                        'd': 1/7,
                                                                        'pc1': .67,
                                                                        'pc2': .67,
                                                                        'pc3': .67,
                                                                        'pc4': .67,
                                                                        'pp1': .32,
                                                                        'pp2': .112,
                                                                        'pp3': .13,
                                                                        'pp4': .38,
                                                                        'b': 0
                                                                        })
    assert len(modelflu.traces) == 21
    assert len(modelflu.traces['time']) == 50
    print(list(modelflu.traces.keys()))
    modelflu.plot_traces()
    P.show()
    assert isinstance(modelflu, Epimodel)

def test_SEIS():
    modelseis  = Epimodel('SEIS')
    modelseis([1000, 1, 0], 50, 1001, {'beta': 2, 'r': 1, 'e': 1, 'b': 0})
    assert len(modelseis.traces) == 4
    assert len(modelseis.traces['time']) == 50
    modelseis.plot_traces()
    P.show()
    assert isinstance(modelseis, Epimodel)

def test_SEIR():
    modelseir  = Epimodel('SEIR')
    tsteps = 50
    modelseir([1000, 1, 1, 0], tsteps, 1002, {'beta': 2, 'r': 1, 'e': 1, 'b': 0, 'alpha': 1})
    assert len(modelseir.traces) == 5
    assert len(modelseir.traces['time']) == tsteps
    modelseir.plot_traces()
    P.show()
    assert isinstance(modelseir, Epimodel)
