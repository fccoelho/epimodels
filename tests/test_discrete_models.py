__author__ = 'fccoelho'

import pytest
from epimodels.discrete.models import  Epimodel



def test_SIS():
    model  = Epimodel('SIS')
    assert isinstance(model, Epimodel)



