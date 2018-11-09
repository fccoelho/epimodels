__author__ = 'fccoelho'

import pytest
# import pyximport; pyximport.install(pyimport=True)
from epimodels.discrete.models import DiscreteModel
from matplotlib import pyplot as P


def test_SIS():
    modelsis  = DiscreteModel('SIS')
    modelsis([0, 1, 1000], 50, 1001, {'beta': 2, 'gamma': 1})
    assert len(modelsis.traces) == 3
    assert len(modelsis.traces['time']) == 50
    modelsis.plot_traces()
    P.show()
    assert isinstance(modelsis, DiscreteModel)


def test_SIR():
    modelsir  = DiscreteModel('SIR')
    modelsir([1000, 1, 0], 50, 1001, {'beta': 2, 'gamma': 1})
    assert len(modelsir.traces) == 4
    assert len(modelsir.traces['time']) == 50
    modelsir.plot_traces()
    P.show()
    assert isinstance(modelsir, DiscreteModel)

def test_FLU():
    modelflu = DiscreteModel('Influenza')
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
    assert isinstance(modelflu, DiscreteModel)

def test_SEIS():
    modelseis  = DiscreteModel('SEIS')
    modelseis([1000, 1, 0], 50, 1001, {'beta': 2, 'r': 1, 'e': 1, 'b': 0})
    assert len(modelseis.traces) == 4
    assert len(modelseis.traces['time']) == 50
    modelseis.plot_traces()
    P.show()
    assert isinstance(modelseis, DiscreteModel)

def test_SEIR():
    modelseir  = DiscreteModel('SEIR')
    tsteps = 50
    modelseir([1000, 1, 1, 0], tsteps, 1002, {'beta': 2, 'r': 1, 'e': 1, 'b': 0, 'alpha': 1})
    assert len(modelseir.traces) == 5
    assert len(modelseir.traces['time']) == tsteps
    modelseir.plot_traces()
    P.show()
    assert isinstance(modelseir, DiscreteModel)
