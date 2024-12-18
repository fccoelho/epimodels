import unittest
import sys
sys.path.append('..')
from epimodels.continuous import models as cm
from epimodels.discrete import models as dm


class MyTestCase(unittest.TestCase):
    def test_latex_pars(self):
        model = cm.SIR()
        model([1000, 1, 0], [0, 500], 1001, {'beta': .2, 'gamma': .1})
        assert model.parameter_table(True).startswith("\\begin[l|c|c]{tabular}")
        assert model.parameter_table(True).endswith("\\end{tabular}")

    def test_latex_pars_discrete(self):
        model = dm.SIR()
        model([1000, 1, 0], [0, 500], 1001, {'beta': .2, 'gamma': .1})
        assert model.parameter_table(True).startswith("\\begin[l|c|c]{tabular}")
        assert model.parameter_table(True).endswith("\\end{tabular}")



if __name__ == '__main__':
    unittest.main()
