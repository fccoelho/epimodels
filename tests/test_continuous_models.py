__author__ = 'fccoelho'

import unittest
import pytest
from matplotlib import pyplot as P
# import pyximport; pyximport.install(pyimport=True)

from epimodels.continuous.models import ContinuousModel


def test_SIR():
    model = ContinuousModel('SIR')
    model([1000, 1, 0], [0,50], 1001, {'beta': .1, 'gamma': .09})
    assert len(model.traces) == 4
    # assert len(model.traces['time']) == 50
    model.plot_traces()
    P.show()
