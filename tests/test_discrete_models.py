__author__ = 'fccoelho'

import pytest
# import pyximport; pyximport.install(pyimport=True)
from epimodels.discrete.models import (DiscreteModel, Influenza, SIS, SIR, SEIS, SEIR,
                                       SIpRpS, SEIpRpS, SIpR, SEIpR, SIRS, SEQIAHR)
from matplotlib import pyplot as P


def test_SIS():
    modelsis = SIS()
    modelsis([0, 1, 1000], [0,50], 1001, {'beta': 2, 'gamma': 1})
    assert len(modelsis.traces) == 3
    assert len(modelsis.traces['time']) == 50
    modelsis.plot_traces()
    P.show()
    assert isinstance(modelsis, DiscreteModel)


def test_SIR():
    modelsir = SIR()
    modelsir([1000, 1, 0], [0,500], 1001, {'beta': .2, 'gamma': .1})
    assert len(modelsir.traces) == 4
    assert len(modelsir.traces['time']) == 500
    modelsir.plot_traces()
    P.show()
    assert isinstance(modelsir, DiscreteModel)


def test_FLU():
    modelflu = Influenza()
    modelflu([250, 1, 0, 0, 0, 250, 1, 0, 0, 0, 250, 1, 0, 0, 0, 250, 1, 0, 0, 0],
             [0,50],
             1004,
             {'beta': 2.0,
              'r': 0.25,
              'e': 0.5,
              'c': 0.5,
              'g': 1 / 3,
              'd': 1 / 7,
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
    modelseis = SEIS()
    modelseis([1000, 1, 0], [0,50], 1001, {'beta': 2, 'r': 1, 'e': 1, 'b': 0})
    assert len(modelseis.traces) == 4
    assert len(modelseis.traces['time']) == 50
    modelseis.plot_traces()
    P.show()
    assert isinstance(modelseis, DiscreteModel)


def test_SEIR():
    modelseir = SEIR()
    tsteps = [0,50]
    modelseir([1000, 1, 1, 0], tsteps, 1002, {'beta': 2, 'r': 1, 'e': 1, 'b': 0, 'alpha': 1})
    assert len(modelseir.traces) == 5
    assert len(modelseir.traces['time']) == tsteps[1]
    modelseir.plot_traces()
    P.show()
    assert isinstance(modelseir, DiscreteModel)


def test_SIpRpS():
    modelsiprps = SIpRpS()
    tsteps = [0,50]
    modelsiprps([1000, 1, 0], tsteps, 1002, {'beta': 2, 'r': 1, 'e': 1, 'b': 0, 'delta': 0.5})
    assert len(modelsiprps.traces) == 4
    assert len(modelsiprps.traces['time']) == tsteps[1]
    modelsiprps.plot_traces()
    P.show()
    assert isinstance(modelsiprps, DiscreteModel)


def test_SEIpRpS():
    modelseiprps = SEIpRpS()
    tsteps = [0,50]
    modelseiprps([1000, 1, 0, 0], tsteps, 1002, {'beta': 2, 'r': 1, 'e': 1, 'b': 0, 'delta': 0.5})
    assert len(modelseiprps.traces) == 5
    assert len(modelseiprps.traces['time']) == tsteps[1]
    modelseiprps.plot_traces()
    P.show()
    assert isinstance(modelseiprps, DiscreteModel)


def test_SIpR():
    modelsipr = SIpR()
    modelsipr([1000, 1, 0], [0,50], 1001, {'beta': 2, 'gamma': 1, 'b': 0, 'r': .5, 'p': .5})
    assert len(modelsipr.traces) == 4
    assert len(modelsipr.traces['time']) == 50
    modelsipr.plot_traces()
    P.show()
    assert isinstance(modelsipr, DiscreteModel)

def test_SEIpR():
    modelseipr = SEIpR()
    tsteps = [0,50]
    modelseipr([1000, 1, 1, 0], tsteps, 1002, {'beta': 2, 'r': 1, 'e': 1, 'b': 0, 'p':0.5, 'alpha': 1})
    assert len(modelseipr.traces) == 5
    assert len(modelseipr.traces['time']) == tsteps[1]
    modelseipr.plot_traces()
    P.show()
    assert isinstance(modelseipr, DiscreteModel)

def test_SIRS():
    modelsirs = SIRS()
    modelsirs([1000, 1, 0], [0,50], 1001, {'beta': 2, 'b': 0, 'r': .5, 'w': .5})
    assert len(modelsirs.traces) == 4
    assert len(modelsirs.traces['time']) == 50
    modelsirs.plot_traces()
    P.show()
    assert isinstance(modelsirs, DiscreteModel)

def test_SEQIAHR():
    model = SEQIAHR()
    model([.99, 0, 1e-6, 0, 0, 0, 0, 0], [0, 300], 1, {'chi': .7, 'phi': .01, 'beta': .5,
                                                    'rho': .05, 'delta': .1, 'gamma': .1, 'alpha': .33, 'mu': .03,
                                                    'p': .75, 'q': 50, 'r': 40
                                                    })
    assert len(model.traces) == 9
    model.plot_traces()
    P.show()
    assert isinstance(model, DiscreteModel)