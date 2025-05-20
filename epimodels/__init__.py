import sys
import pandas as pd

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from matplotlib import pyplot as P


class BaseModel:
    """
    Base class for all models both discrete and continuous
    """

    def __init__(self):
        self.name = None
        self.model_type = None
        self.state_variables = {}
        self.parameters = {}
        self.param_values = {}
        self.traces = {}

    def plot_traces(self, vars: list = []):
        """
        Plots the simulations
        :param vars: variables to plot
        """
        for series, data in self.traces.items():
            if series in self.state_variables:
                P.plot(self.traces['time'], data, label=series)
        P.legend(loc=0)
        P.grid()
        P.title("{} model".format(self.model_type))

    def parameter_table(self, latex: bool = False):
        if self.parameters:
            tbl = {
                'Parameter': list(self.parameters.keys()),
                'Value': list(self.param_values.values()),
                'Symbol': list(self.parameters.values())
            }

            if latex == True:
                out = r"""\begin[l|c|c]{tabular}
                \hline
                Parameter & Value & Symbol \\
                \hline
                """ + \
                    r"\\".join([f"{p}&{v}&{s}" for p, v, s in zip(tbl['Parameter'], tbl['Value'], tbl['Symbol'])]) + \
                    r"""
                    \hline
                    \end{tabular}"""
            else:
                out = pd.DataFrame(tbl)
        else:
            out = pd.DataFrame()
        return out
