import sys

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
        self.traces = {}

    def plot_traces(self, vars: list=[]):
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